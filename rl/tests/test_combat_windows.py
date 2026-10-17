"""Original-time cutoff and visible-context mobility without trajectory edits."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from soku_rl.policy.hierarchical_control import command_catalog
from soku_rl.replay.combat_windows import post_opening_selection
from soku_rl.rl.semantic_sampling import (sampling_membership, choice_weight_mask,
                                         stratified_order)
from soku_rl.rl.hierarchical_data import semantic_training_dataset
from soku_rl.rl.factorized_training import validate_config


def fixture(frames):
    catalog = command_catalog()
    neutral = next(i for i, c in enumerate(catalog) if c['name'] == '5move')
    n = len(frames)
    data = dict(state=np.zeros((n, 185), np.float32),
        active_action=np.full(n, len(catalog), np.int64),
        target_action=np.full(n, neutral, np.int64), target_action_mask=np.ones(n, bool),
        target_termination_mask=np.zeros(n, bool), target_sequence_mask=np.zeros((n, 3), bool),
        supervision_strata=np.tile([True, False, False, False], (n, 1)))
    alignment = dict(samples=[dict(observation_frame=f) for f in frames])
    return data, alignment, catalog


def test_cutoff_uses_original_time_preserves_histories_and_all_attacks():
    data, alignment, catalog = fixture([299, 300, 355])
    attack = next(i for i, c in enumerate(catalog) if not c['continuous'])
    data['target_action'][:] = attack
    data['event_states'] = np.ones((3, 16, 185), np.float32)
    original = deepcopy(data)
    meta, stats = post_opening_selection(data, alignment, catalog)
    assert meta['training_eligible'].tolist() == [False, True, True]
    assert stats['eligible_attack_labels'] == 2
    assert stats['object_continuous_labels'] == 0
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


def test_both_owners_admit_mobility_without_own_attack_or_contact():
    data, alignment, catalog = fixture([300, 301, 302, 303])
    direction = next(i for i, c in enumerate(catalog)
                     if c['continuous'] and c['name'] != '5move')
    data['target_action'][1:3] = direction
    data['state'][:3, 20] = 1
    data['state'][1, 29] = 1
    # Past objects must NOT admit a plain current empty-screen movement.
    data['event_states'] = np.ones((4, 16, 185), np.float32)
    meta, stats = post_opening_selection(data, alignment, catalog)
    assert meta['training_eligible'].tolist() == [True, True, True, False]
    assert stats['object_mobility_without_active_attack'] == 3
    assert stats['enemy_object_mobility_without_active_attack'] == 1
    assert stats['object_neutral_or_release_labels'] == 1
    assert stats['object_direction_or_D_labels'] == 2


def test_no_fabricated_choice_and_keep_existing_link_and_continuation():
    data, alignment, catalog = fixture([300, 301, 302, 303])
    data['target_action_mask'][0:2] = False
    data['state'][0:2, 20] = 1
    data['target_termination_mask'][1] = True
    data['supervision_strata'][2, 2] = True
    data['active_action'][3] = next(i for i, c in enumerate(catalog) if not c['continuous'])
    data['target_termination_mask'][3] = True
    meta, _ = post_opening_selection(data, alignment, catalog)
    assert meta['training_eligible'].tolist() == [False, True, True, True]
    assert not data['target_action_mask'][1]  # Context-only stop supervision.


@pytest.mark.parametrize('frames', [[300, 300], [301, 300], [-1, 300]])
def test_invalid_original_chronology_rejected(frames):
    data, alignment, catalog = fixture(frames)
    with pytest.raises(ValueError, match='chronology'):
        post_opening_selection(data, alignment, catalog)


def test_sampler_and_class_weights_never_include_opening():
    data, alignment, catalog = fixture([299, 300, 301])
    data['state'][:, 20] = 1
    np.testing.assert_array_equal(sampling_membership(data), data['supervision_strata'])
    np.testing.assert_array_equal(choice_weight_mask(data), data['target_action_mask'])
    data.update(post_opening_selection(data, alignment, catalog)[0])
    assert choice_weight_mask(data).tolist() == [False, True, True]
    order, _, report = stratified_order(sampling_membership(data), np.arange(3),
        np.full(3, -1), np.random.default_rng(7), 4)
    assert report['status'] == 'PASS' and set(order.tolist()) == {1, 2}
    data['training_eligible'][:] = False
    order, _, report = stratified_order(sampling_membership(data), np.arange(3),
        np.full(3, -1), np.random.default_rng(7), 1)
    assert not len(order) and report['status'] == 'BLOCKED_CAPACITY'


@pytest.mark.parametrize('corruption', [None, 'missing', 'opening', 'integer_mask'])
def test_versioned_loader_fails_closed(tmp_path, corruption):
    data, alignment, catalog = fixture([299, 300])
    data['state'][:, 20] = 1
    data.update(post_opening_selection(data, alignment, catalog)[0])
    if corruption == 'missing':
        del data['training_eligible']
    elif corruption == 'opening':
        data['training_eligible'][0] = True
    elif corruption == 'integer_mask':
        data['training_eligible'] = data['training_eligible'].astype(np.int64)
    shard = tmp_path/'samples.npz'
    np.savez_compressed(shard, **data)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(format='semantic_post_opening_dataset_v3',
        status='PASS_OFFLINE_SELECTION_ORIGINAL_ARRAYS_EXACT', catalog=catalog,
        selection=dict(min_observation_frame=300), cases=[dict(partition='train',
            samples=2, path=shard.name, sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])))
    dataset = semantic_training_dataset(manifest, 'train')
    if corruption:
        with pytest.raises(ValueError, match='contract'):
            dataset.batch([0, 1])
    else:
        assert dataset.batch([0, 1])['training_eligible'].tolist() == [False, True]


def test_prepared_profile_changes_data_only_not_training_contract():
    old = OmegaConf.to_container(OmegaConf.load('rl/config/hierarchical_factorized_training.yaml'))
    new = OmegaConf.to_container(OmegaConf.load('rl/config/hierarchical_factorized_post_opening.yaml'))
    validate_config(new)
    assert new.pop('data') != old.pop('data')
    assert new == old
