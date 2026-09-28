from dataclasses import replace

import numpy as np
from pettingzoo.test import parallel_api_test
import pytest

from test_env_timing import RecordingBackend, VISIBILITY
from soku_rl.env import EpisodeConfig, HisoutenParallelEnv, TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS, decode_action
from soku_rl.learning_features import relative_features
from soku_rl.learning_wrappers import (LearningConfig, LearningEpisode, LearningInterface,
                                      LearningParallelEnv, LearningVectorEnv)


def config(mode):
    return EpisodeConfig(17, 4, 3, 12, mode, VISIBILITY)


def gauges(own, enemy):
    frame = np.zeros(400, dtype=np.float32)
    frame[[0, 8, 4, 12]] = 1
    frame[5], frame[13] = own, enemy
    return np.tile(frame, 4)


def test_shaping_telescopes_at_ko_and_timeout():
    for timeout in (False, True):
        interface = LearningInterface(config("state"), LearningConfig("combat", True, 8, 2.))
        episode = LearningEpisode(interface)
        infos = {a: {"frame": 0} for a in AGENTS}
        episode.reset({a: gauges(1, 1) for a in AGENTS}, infos)
        total = dict.fromkeys(AGENTS, 0.)
        for index, (hp0, hp1) in enumerate(((1, .9), (.8, .5), (.8, 0))):
            ended = index == 2
            observations = dict(zip(AGENTS, (gauges(hp0, hp1), gauges(hp1, hp0))))
            rewards = dict(zip(AGENTS, (1., -1.))) if ended and not timeout else dict.fromkeys(AGENTS, 0.)
            flags = (dict.fromkeys(AGENTS, ended and not timeout), dict.fromkeys(AGENTS, ended and timeout))
            result = episode.step(dict.fromkeys(AGENTS, 256), (observations, rewards, *flags, infos))
            assert sum(result[1].values()) == pytest.approx(0)
            for agent in AGENTS:
                assert interface.observation_space.contains(result[0][agent])
                total[agent] += result[1][agent]
        assert total == pytest.approx(dict.fromkeys(AGENTS, 0.) if timeout else dict(zip(AGENTS, (1., -1.))))


def test_wrappers_do_not_advance_extra_frames_or_bypass_latency():
    backend = RecordingBackend()
    env = LearningVectorEnv(TwoPlayerVectorEnv(backend, 2, config("diagnostic_state")),
                            LearningConfig("combat", False, 8, 1.))
    env.reset({0: 1, 1: 2})
    command = env.interface.command(89)
    for _ in range(5):
        env.step({s: dict.fromkeys(AGENTS, 89) for s in (0, 1)})
    assert backend.inputs[0][:12] == [(decode_action(256),) * 2] * 12
    assert backend.inputs[0][12:] == [(decode_action(command),) * 2] * 3
    before = list(env.transforms[1].history[AGENTS[0]])
    observations, _ = env.reset({0: 3})
    assert list(env.transforms[1].history[AGENTS[0]]) == before
    assert (observations[0][AGENTS[0]][-64:] == 0).all()
    assert backend.frames == {0: 0, 1: 15}


def test_pettingzoo_and_vector_wrappers_match():
    single = LearningParallelEnv(HisoutenParallelEnv(RecordingBackend(), config("diagnostic_state")),
                                 LearningConfig("combat", False, 8, 1.))
    vector = LearningVectorEnv(TwoPlayerVectorEnv(RecordingBackend(), 1, config("diagnostic_state")),
                               LearningConfig("combat", False, 8, 1.))
    left, _ = single.reset(seed=3)
    right, _ = vector.reset({0: 3})
    for a in AGENTS:
        np.testing.assert_array_equal(left[a], right[0][a])
    for _ in range(6):
        actions = dict.fromkeys(AGENTS, 41)
        left, right = single.step(actions), vector.step({0: actions})
        for a in AGENTS:
            np.testing.assert_array_equal(left[0][a], right[0][0][a])
        assert left[1:] == tuple(value[0] for value in right[1:])
    parallel_api_test(single, num_cycles=1000)


def test_relative_features_have_masks_and_public_sources_only():
    episode = config("state")
    value = gauges(1, .8)
    value[-392] = 0  # Opponent is hidden in the current frame.
    features = relative_features(value, episode, 3)
    assert (features[:4] == 0).all()
    assert features[-3] == pytest.approx(.2)
    assert features[-1] == pytest.approx(3 / 17)


def test_action_vocabulary_is_reversible_and_invalid_inputs_fail():
    interface = LearningInterface(config("state"), LearningConfig("combat", False, 0, 0.))
    assert interface.action_space.n == 90
    assert interface.command(interface.action(256)) == 256
    for action in range(90):
        assert interface.action(interface.command(action)) == action
    for invalid in (-1, 90, True, 1.5):
        with pytest.raises(ValueError):
            interface.command(invalid)
    with pytest.raises(ValueError, match="unavailable"):
        interface.action(575)
    with pytest.raises(ValueError, match="image"):
        LearningInterface(config("image"), LearningConfig("full", False, 8, 1.))
