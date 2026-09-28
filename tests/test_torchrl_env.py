"""Check tensor and task adapters when the optional training packages exist."""
import importlib.util
import unittest

from test_env_timing import RecordingBackend, VISIBILITY
from soku_rl.env import EpisodeConfig, HisoutenParallelEnv


@unittest.skipUnless(importlib.util.find_spec("torchrl"), "install the marl extra")
class TorchRLContractTests(unittest.TestCase):
    def test_numeric_specs_and_episode_boundaries(self):
        from torchrl.envs.utils import check_env_specs
        from soku_rl.torchrl_env import wrap_torchrl
        from soku_rl.benchmarl_task import SokuTask

        config = EpisodeConfig(30, 4, 3, 12, "diagnostic_state", VISIBILITY)
        env = wrap_torchrl(HisoutenParallelEnv(RecordingBackend(), config), 123, "cpu")
        try:
            check_env_specs(env)
            rollout = env.rollout(20, break_when_any_done=False)
            self.assertEqual(rollout.batch_size[0], 20)
            self.assertEqual(set(env.group_map), {"player_0", "player_1"})
            task = SokuTask("VS", {"episode": {"max_frames": 30, "decision_frames": 3}})
            self.assertEqual(task.max_steps(env), 10)
            self.assertEqual(set(task.observation_spec(env).keys()), set(env.group_map))
            self.assertFalse(task.has_state())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
