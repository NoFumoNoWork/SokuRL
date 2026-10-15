"""Training-only expert windows mixed into the online teacher's existing SGD budget."""
import numpy as np
import torch

from soku_rl.rl.demonstration_sets import load_demonstration_sets
from soku_rl.rl.demonstration_supervision import supervision_mask
from soku_rl.rl.recurrent_cloning import demonstration_episodes


def validate_teacher_replay(config, total_sequences):
    if not isinstance(config, dict) or set(config) != {'datasets', 'sequences', 'seed'}:
        raise ValueError('teacher demonstration_replay requires datasets, sequences and seed')
    if (not isinstance(config['datasets'], list) or not config['datasets']
            or any(not isinstance(path, str) or not path for path in config['datasets'])):
        raise ValueError('teacher replay datasets must be a nonempty list of paths')
    if type(config['sequences']) is not int or not 0 < config['sequences'] < total_sequences:
        raise ValueError('teacher replay sequences must leave at least one online window')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('teacher replay seed must be a nonnegative integer')


def load_teacher_replay(config, interface, teacher_fingerprint):
    samples, _, identities = load_demonstration_sets(config['datasets'], interface, 0.)
    if any(row['control'] != 'teacher' or row['teacher_fingerprint'] != teacher_fingerprint
           for row in identities):
        raise ValueError('teacher replay requires original trajectories of the same rule teacher')
    # The loader verifies held-out shards, but the sampler receives only train.
    return TeacherReplay(samples['train']), identities


class TeacherReplay:
    def __init__(self, samples):
        episodes = demonstration_episodes(samples)
        if not episodes or any(not supervision_mask(rows).all() for rows in episodes):
            raise ValueError('teacher replay requires nonempty, fully supervised original episodes')
        self.episodes = [{'observations': [row[0] for row in rows], 'labels': [row[1] for row in rows]}
                         for rows in episodes]
        self.ends = np.cumsum([len(row['labels']) for row in self.episodes])

    def sample(self, rng, count, sequence_length, device):
        windows = []
        for index in rng.integers(int(self.ends[-1]), size=count):
            episode_index = int(np.searchsorted(self.ends, index, side='right'))
            offset = int(index - (0 if episode_index == 0 else self.ends[episode_index - 1]))
            episode = self.episodes[episode_index]
            length = min(sequence_length, len(episode['labels']) - offset)
            labels = torch.tensor(episode['labels'][offset:offset + length], device=device)
            windows.append((episode['observations'], offset, length, labels, 'replay'))
        return windows
