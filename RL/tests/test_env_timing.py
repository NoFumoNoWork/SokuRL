"""Exercise the public single and vector interfaces with a deterministic backend."""
import unittest
from dataclasses import replace
import numpy as np
from pettingzoo.test import parallel_api_test

from soku_rl.env.observation.diagnostic import Fighter, Observation
from soku_rl.env import EpisodeConfig, HisoutenParallelEnv, TwoPlayerVectorEnv
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.encoding import AGENTS, decode_action
from soku_rl.pomg import Outcome, TimeStep
from soku_rl.env.observation.pixels import RGBFrame
from soku_rl.env.observation.visibility import VisibilityConfig

VISIBILITY = VisibilityConfig(8, .5, .02, .1, 48., 96., 16., .25)


class RecordingBackend:
    def __init__(self):
        self.frames = {}
        self.inputs = {}

    def _state(self, slot):
        fighter = Fighter(400, 0, 10000, 1, 0, False, 0, 0, 1)
        obs = Observation(self.frames[slot], fighter, fighter, ())
        return TimeStep(self.frames[slot], (obs, obs), (0., 0.), Outcome.ONGOING, {})

    def reset_slots(self, seeds):
        for slot in seeds:
            self.frames[slot], self.inputs[slot] = 0, []
        return {s: self._state(s) for s in seeds}

    def step(self, actions):
        for slot, action in actions.items():
            self.inputs[slot].append(action)
            self.frames[slot] += 1
        return {s: self._state(s) for s in actions}

    def close(self):
        pass


class EnvTimingTests(unittest.TestCase):
    def test_pettingzoo_parallel_contract_for_images(self):
        class ImageBackend(RecordingBackend):
            def _state(self, slot):
                state = super()._state(slot)
                image = RGBFrame(state.frame, 320, 240, bytes([state.frame % 256]) * (320 * 240 * 3))
                return replace(state, observations=(image, image))

        env = HisoutenParallelEnv(ImageBackend(), EpisodeConfig(60, 4, 3, 12, "image", VISIBILITY, LEGACY_MATCH))
        parallel_api_test(env, num_cycles=1000)
        observations, infos = env.reset(seed=123)
        self.assertTrue(env.observation_space(AGENTS[0]).contains(observations[AGENTS[0]]))
        self.assertEqual(observations[AGENTS[0]].shape, (13, 240, 320))
        self.assertTrue((observations[AGENTS[0]][-1] == 0).all())
        self.assertTrue((observations[AGENTS[1]][-1] == 255).all())
        self.assertNotIn("seed", infos[AGENTS[0]])
        self.assertNotIn("diagnostics", infos[AGENTS[0]])
        env.close()

    def test_single_and_vector_have_identical_transitions(self):
        config = EpisodeConfig(17, 4, 3, 5, "diagnostic_state", VISIBILITY, LEGACY_MATCH)
        single_backend, vector_backend = RecordingBackend(), RecordingBackend()
        single = HisoutenParallelEnv(single_backend, config)
        vector = TwoPlayerVectorEnv(vector_backend, 2, config)
        single.reset(seed=1)
        vector.reset({0: 1, 1: 2})
        for _ in range(6):
            actions = dict.fromkeys(AGENTS, 0)
            left = single.step(actions)
            right = vector.step({0: actions})
            self.assertEqual(left[1:], tuple(value[0] for value in right[1:]))
        self.assertEqual(single_backend.frames[0], 17)
        self.assertEqual(vector_backend.frames[1], 0)
        self.assertEqual(single_backend.inputs[0], vector_backend.inputs[0])
        self.assertEqual(single_backend.inputs[0][:5], [(decode_action(256),) * 2] * 5)
        self.assertEqual(single_backend.inputs[0][5:], [(decode_action(0),) * 2] * 12)
        self.assertTrue(left[3][AGENTS[0]])
        self.assertEqual(single.agents, [])

    def test_partial_reset_clears_only_selected_episode(self):
        backend = RecordingBackend()
        env = TwoPlayerVectorEnv(backend, 2, EpisodeConfig(100, 4, 3, 12, "diagnostic_state", VISIBILITY, LEGACY_MATCH))
        env.reset({0: 1, 1: 2})
        for _ in range(5):
            env.step({s: dict.fromkeys(AGENTS, 0) for s in (0, 1)})
        env.reset({0: 3})
        env.step({s: dict.fromkeys(AGENTS, 575) for s in (0, 1)})
        self.assertEqual(backend.frames, {0: 3, 1: 18})
        self.assertEqual(backend.inputs[0], [(decode_action(256),) * 2] * 3)
        self.assertEqual(backend.inputs[1][-3:], [(decode_action(0),) * 2] * 3)


if __name__ == "__main__":
    unittest.main()
