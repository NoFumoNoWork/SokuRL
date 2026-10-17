"""Semantic hierarchy, independent label roles and causal feedback contracts."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium import spaces

from soku_rl.policy.hierarchical_control import command_catalog, program
from soku_rl.rl.animation_features import AnimationTokenFeatures
from soku_rl.rl.factorized_bc import (ChoiceHead, FactorizedActor, category_ids,
                                      factorized_loss, rare_weights, warm_start)
from soku_rl.rl.hierarchical_bc import HierarchicalActor
from soku_rl.rl.semantic_sampling import stratified_order
from soku_rl.replay.semantic_events import SemanticEventHistory, contact_feedback, event_strata
from test_hierarchical_bc import batch


@pytest.fixture
def actor():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    source = AnimationTokenFeatures(spaces.Box(-np.inf, np.inf, (1544,), np.float32), timing_vocab_size=1024)
    return FactorizedActor(source, command_catalog()).eval()


def test_catalog_partition_and_interpreter():
    catalog = command_catalog()
    ids = category_ids(catalog)
    assert len(catalog) == len(ids) == 56 and set(ids) == set(range(6))
    for item in catalog:
        assert program(item)  # All labels use the existing native command compiler.


def test_joint_normalization_masks_and_no_nan(actor):
    x = batch(actor.actions)
    x['candidates'][0] = False
    x['candidates'][0, 9] = True  # One locomotion candidate, five empty categories.
    y = actor(x)
    torch.testing.assert_close(y['action'].exp().sum(-1), torch.ones(2))
    assert y['action'][0, 9] == 0 and y['action'][0].argmax() == 9
    assert not torch.isnan(y['action']).any()
    assert torch.isneginf(y['category'][0]).sum() == 5
    assert len(actor.events.layers) == 3
    assert actor.events.layers[0].self_attn.num_heads == 4
    x['candidates'][:] = False
    with pytest.raises(ValueError, match='empty'):
        actor(x)


def test_loss_denominators_weights_and_unsupported_target():
    logits = torch.zeros(3, 4, requires_grad=True).log_softmax(-1)
    outputs = dict(action=logits, termination=torch.zeros(3, requires_grad=True),
                   sequence=logits[:, None].expand(-1, 3, -1))
    targets = dict(action=torch.tensor([0, 1, -1]), action_mask=torch.tensor([1, 1, 0]),
        termination=torch.tensor([0., 1., 0.]), termination_mask=torch.tensor([1, 0, 0]),
        sequence=torch.tensor([[0, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
        sequence_mask=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    loss = factorized_loss(outputs, targets, torch.tensor([4., 1., 1., 1.]))
    torch.testing.assert_close(loss['choice'], torch.tensor(2.5 * np.log(4), dtype=torch.float32))
    torch.testing.assert_close(loss['future'], torch.tensor(4 * np.log(4), dtype=torch.float32))
    torch.testing.assert_close(loss['total'], loss['choice'] + .3 * loss['termination'] + .15 * loss['future'])
    loss['total'].backward()
    bad = dict(outputs, action=logits.masked_fill(torch.arange(4) == 0, float('-inf')))
    with pytest.raises(ValueError, match='excluded'):
        factorized_loss(bad, targets, torch.ones(4))
    with pytest.raises(ValueError, match='bounded'):
        factorized_loss(outputs, targets, torch.full((4,), 5.))


def test_rare_weights_training_only_bounded():
    w = rare_weights(torch.tensor([0] * 100 + [1, 2]), torch.ones(102, dtype=torch.bool), 4)
    assert w.tolist() == [1., 4., 4., 1.]


def test_future_targets_cannot_change_choice_or_first_future(actor):
    x = batch(actor.actions)
    teacher = torch.tensor([[1, 2, 3], [2, 3, 4]])
    a = actor(x, selected=torch.tensor([0, 1]), teacher=teacher)
    teacher[:, 0] = 8
    b = actor(x, selected=torch.tensor([0, 1]), teacher=teacher)
    torch.testing.assert_close(a['action'], b['action'], rtol=0, atol=0)
    torch.testing.assert_close(a['sequence'][:, 0], b['sequence'][:, 0], rtol=0, atol=0)
    assert not torch.equal(a['sequence'][:, 1], b['sequence'][:, 1])


def test_shared_backbone_warm_start_exact(actor):
    source = AnimationTokenFeatures(spaces.Box(-np.inf, np.inf, (1544,), np.float32), timing_vocab_size=1024)
    old = HierarchicalActor(source, len(command_catalog()))
    audit = warm_start(actor, old)
    assert audit['optimizer'] == 'FRESH'
    assert 'events.layers.2.linear1.weight' in audit['copied']
    assert 'action_head.weight' in audit['discarded']
    assert 'choice.category.weight' in audit['initialized']
    for key in audit['copied']:
        torch.testing.assert_close(actor.state_dict()[key], old.state_dict()[key], rtol=0, atol=0)


def test_inference_selects_category_before_member(actor):
    with torch.no_grad():
        actor.choice.category.weight.zero_()
        actor.choice.category.bias.fill_(-100)
        actor.choice.category.bias[0] = .4
        actor.choice.category.bias[2] = 0
        actor.choice.conditional.weight.zero_()
        actor.choice.conditional.bias.zero_()
        melee = torch.where(actor.choice.mapping == 2)[0][0]
        actor.choice.conditional.bias[melee] = 100
    y = actor(batch(actor.actions))
    assert (actor.choice.mapping[y['choice_id']] == 0).all()
    assert (actor.choice.mapping[y['action'].argmax(-1)] == 2).all()


def test_sampler_overlap_missing_strata_caps_and_reproducibility():
    strata = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0], [0, 1, 1, 0]], bool)
    args = (strata, [1, 1, 2, 2], [-1, 0, -1, 0])
    a, roles, report = stratified_order(*args, np.random.default_rng(5), 8,
                                       row_cap=2, replay_cap=4, chain_cap=2)
    b, _, _ = stratified_order(*args, np.random.default_rng(5), 8,
                              row_cap=2, replay_cap=4, chain_cap=2)
    assert np.array_equal(a, b) and report['status'] == 'PASS'
    assert (strata[a, roles]).all() and 'defense_reaction' in report['depleted_strata']
    assert report['max_chain_draws'] <= 2 and report['max_replay_draws'] <= 4
    _, _, blocked = stratified_order(*args, np.random.default_rng(5), 9,
                                     row_cap=1, replay_cap=2, chain_cap=1)
    assert blocked['status'] == 'BLOCKED_CAPACITY' and blocked['actual'] == 4


def test_feedback_time_gate_and_snapshot_not_retroactive():
    h = SemanticEventHistory()
    state = np.zeros(185, np.float32)
    h.observe(state, 0)
    before = h.tensors()
    e = dict(kind='BLOCK_HIGH_RESOLUTION', hit_or_block='BLOCK', frame=1, order=2,
             segment=0, attacker_seat=1)
    feedback = contact_feedback([e], 0, 1)
    h.observe(state, 1, feedback=feedback)
    assert not before['event_mask'].any()
    assert h.tensors()['event_ids'][-1].tolist() == [6, 0, 1]
    with pytest.raises(ValueError, match='future'):
        contact_feedback([dict(e, frame=3)], 0, 2)
    with pytest.raises(ValueError, match='confirmed'):
        h.observe(state, 2, feedback=[dict(feedback[0], available_frame=2, evidence='UNKNOWN')])


def test_feedback_kinds_require_independent_confirmed_evidence():
    for kind, result in [('WHIFF', 1), ('CANCEL', 2), ('RECOVERY', 3)]:
        h = SemanticEventHistory(); state = np.zeros(185, np.float32)
        h.observe(state, 0, feedback=[dict(kind=kind, occurred_frame=0, available_frame=0,
            owner=0, evidence='CONFIRMED', source='synthetic_test_NOT_GAME_PROOF')])
        assert h.tensors()['event_numbers'][-1, -1] == result


def test_event_roles_not_action_categories_and_passive_not_choice():
    data = dict(target_action_mask=np.array([1, 0], bool), target_termination_mask=np.array([1, 0], bool),
        target_sequence_mask=np.zeros((2, 3), bool), state=np.zeros((2, 185)))
    data['state'][1, 5] = 158
    audit = dict(samples=[dict(row=3, observation_frame=3, selection=dict(acceptance=4)),
                          dict(row=4, observation_frame=4, selection=None)])
    chains = dict(decisions=[dict(observation_row=2, next_decision=dict(id=1)),
                             dict(observation_row=4, next_decision=None)], combos=[])
    strata, _ = event_strata(data, audit, chains, {4})
    assert strata[0].tolist() == [False, True, True, False]
    assert not strata[1].any()  # Forced reaction without labels never fabricates a decision.
