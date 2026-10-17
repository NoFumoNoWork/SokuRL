"""Causal multi-rate windows over hash-verified, dense human replay exports."""
import hashlib

import numpy as np

from sokurl_platform.replay_inputs import decode_keys
from soku_rl.env.observation.animation import frame_width


RECENT = (20, 15, 10, 5, 0)
DISTANT = (60, 45, 30)
FRAME_WIDTH = frame_width(16)
LATEST = slice(7 * FRAME_WIDTH, 8 * FRAME_WIDTH)
TOKEN_SCALE = np.array([1, 1, 1, 1, 19, 2047, 255, 8191, 8191, 8191], np.float32)


def detailed(frames):
    result = np.asarray(frames, np.float32).copy()
    result[..., :180] = (result[..., :180].reshape(*result.shape[:-1], 18, 10)
                         / TOKEN_SCALE).reshape(*result.shape[:-1], 180)
    # Only the current resource values are needed; do not duplicate past gauges.
    result[..., :-1, -5:-1] = 0
    return result


def coarse(frames, previous):
    """40 public features: players, relative geometry, displacement, objects.

    Displacement is an observed position difference, NOT native velocity.
    Object summaries are per owner; no identity/tracking across sorted slots.
    """
    frames, previous = np.asarray(frames), np.asarray(previous)
    players = frames[..., :20].reshape(*frames.shape[:-1], 2, 10)
    past = previous[..., :20].reshape(*frames.shape[:-1], 2, 10)
    relative = np.concatenate((players[..., 1, 1:3] - players[..., 0, 1:3],
                               players[..., 0, 1:3]), axis=-1)
    movement = ((past[..., 1:3] - players[..., 1:3]) *
                (players[..., :1] * past[..., :1])).reshape(*frames.shape[:-1], 4)
    objects = frames[..., 20:180].reshape(*frames.shape[:-1], 16, 10)
    summaries = []
    for owner in range(2):
        valid = (objects[..., 0] > 0) & (objects[..., 9] == owner)
        count = valid.sum(-1)
        mean = (objects[..., 1:3] * valid[..., None]).sum(-2) / np.maximum(count[..., None], 1)
        action_mean = (objects[..., 5] * valid).sum(-1) / np.maximum(count, 1) / 2047
        nearest = np.min(np.where(valid, np.abs(objects[..., 1] - players[..., 0, 1, None]),
                                 np.inf), axis=-1)
        summaries.extend([count[..., None]/16, mean, action_mean[..., None],
                          np.where(count, nearest, 0)[..., None], (count > 0)[..., None]])
    return np.concatenate([(players/TOKEN_SCALE).reshape(*frames.shape[:-1], 20),
                           relative, movement, *summaries], axis=-1).astype(np.float32)


def window_inputs(observations, own_inputs, anchors):
    """Does not accept labels; t's input and all future inputs are excluded."""
    anchors = np.asarray(anchors, np.int64)
    frames = observations[:, LATEST]
    recent = frames[anchors[:, None] - np.array(RECENT)]
    distant_indices = anchors[:, None] - np.array(DISTANT)
    return dict(observations=observations[anchors].astype(np.float32),
        recent=detailed(recent),
        distant=coarse(frames[distant_indices], frames[distant_indices + 1]),
        inputs=own_inputs[anchors[:, None] + np.arange(-60, 0)].astype(np.float32))


def prepare_replay(arrays):
    """Keep time rows intact; select anchors, never stitch masked frames."""
    n = len(arrays['actions'])
    if (arrays['observations'].shape != (n, 1544)
            or any(len(v) != n for v in arrays.values())
            or not np.array_equal(arrays['consumption_frames'], arrays['observation_frames'] + 1)
            or not np.isfinite(arrays['observations']).all()):
        raise ValueError('invalid dense causal observation contract')
    own = np.zeros((n, 8), np.float32)
    for i in np.flatnonzero(arrays['raw_words_present']):
        own[i] = tuple(decode_keys(int(arrays['raw_words'][i, arrays['seats'][i]])).values())
    anchors = []
    for i in range(60, n - 4):
        whole = slice(i - 60, i + 5)
        if (not arrays['supervised'][i] or not arrays['compatible'][whole].all()
                or arrays['boundary'][whole].any()
                or not arrays['raw_words_present'][i-60:i].all()
                or not np.all(np.diff(arrays['observation_frames'][whole]) == 1)
                or not np.all(arrays['seats'][whole] == arrays['seats'][i])
                or not np.all(arrays['round_before'][whole] == arrays['round_before'][i])
                or not np.all(arrays['round_after'][whole] == arrays['round_before'][i])):
            continue
        anchors.append(i)
    anchors = np.asarray(anchors, np.int64)
    output = window_inputs(arrays['observations'], own, anchors)
    targets = anchors[:, None] + np.arange(5)
    mask = arrays['supervised'][targets].copy()
    labels = arrays['actions'][targets].copy()
    if np.any(mask & ((labels < 0) | (labels >= 90))):
        raise ValueError('invalid supervised action')
    output.update(labels=labels, mask=mask, changed=arrays['changed'][targets],
                  previous_action=arrays['previous_action'][anchors], anchors=anchors,
                  frames=arrays['observation_frames'][anchors])
    return output


class WindowDataset:
    def __init__(self, accepted):
        self.accepted, self.cases = accepted, []
        parts = []
        shard_index = 0
        for case in accepted.manifest['replays']:
            dense = []
            for _ in case['shards']:
                with np.load(accepted.shards[shard_index], allow_pickle=False) as data:
                    dense.append({key: data[key] for key in data.files})
                shard_index += 1
            arrays = {key: np.concatenate([d[key] for d in dense]) for key in dense[0]}
            data = prepare_replay(arrays)
            self.cases.append(dict(replay_id=case['replay_id'], retained_rows=len(arrays['actions']),
                                   anchors=len(data['labels']), labels=int(data['mask'].sum())))
            parts.append(data)
        self.data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        self.count = len(self.data['labels'])
        if not self.count:
            raise ValueError('no complete causal60F windows')
        digest = hashlib.sha256()
        for key in sorted(self.data):
            digest.update(key.encode()); digest.update(self.data[key].tobytes())
        self.identity = digest.hexdigest()

    def batch(self, indices):
        return {key: value[indices] for key, value in self.data.items()}
