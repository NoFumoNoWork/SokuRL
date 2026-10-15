"""Verify noisy-teacher identity and replay independently seeded input replacement."""
import hashlib
import json
import math

import numpy as np


def validate_noise_manifest(manifest, contract):
    probability = manifest['random_probability']
    actions = manifest['num_actions']
    if (manifest['control'] != 'teacher_noise'
            or manifest['supervision'] != 'unperturbed_teacher_labels_all_frames'
            or type(probability) not in (int, float) or not math.isfinite(probability)
            or not 0 <= probability <= 1 or type(actions) is not int or actions < 1):
        raise ValueError('invalid noisy-teacher demonstration identity')
    identity = ['action-noise-v1', manifest['teacher_fingerprint'], actions, float(probability)]
    if hashlib.sha256(json.dumps(identity).encode()).hexdigest() != manifest['behavior_fingerprint']:
        raise ValueError('noisy-teacher fingerprint differs')
    behavior = contract['behavior']
    if (set(behavior) != {'kind', 'policy', 'random_probability'} or behavior['kind'] != 'action_noise'
            or type(behavior['random_probability']) not in (int, float)
            or behavior['random_probability'] != probability or behavior['policy'] != contract['teacher']):
        raise ValueError('noisy-teacher behavior contract differs')


def validate_noise_episode(data, row, manifest):
    mask = data['noise_selected']
    count = row['steps']
    if (mask.shape != (count,) or mask.dtype != np.bool_ or type(row['noise_decisions']) is not int
            or row['noise_decisions'] != int(mask.sum())):
        raise ValueError('invalid noisy-teacher replacement mask/count')
    gate, actions = np.random.SeedSequence(row['behavior_seed']).spawn(2)
    gate_rng, action_rng = np.random.default_rng(gate), np.random.default_rng(actions)
    expected = data['actions'].copy()
    selected = np.zeros(count, dtype=bool)
    for frame in range(count):
        if gate_rng.random() < manifest['random_probability']:
            selected[frame] = True
            expected[frame] = int(action_rng.integers(manifest['num_actions']))
    if not np.array_equal(mask, selected) or not np.array_equal(data['executed_actions'], expected):
        raise ValueError('noisy-teacher inputs do not match the recorded noise seed')
