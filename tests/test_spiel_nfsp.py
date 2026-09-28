"""Check OpenSpiel collection boundaries and both learning losses on CUDA."""
import importlib.util
import unittest

import numpy as np

from soku_rl.env.encoding import AGENTS


@unittest.skipUnless(importlib.util.find_spec("open_spiel"), "install the nfsp extra")
class SpielNFSPTests(unittest.TestCase):
    def test_explicit_transitions_and_learning(self):
        import torch
        from soku_rl.spiel_nfsp import VectorNFSP

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this algorithm integration check")
        config = dict(hidden_layers_sizes=[32], reservoir_buffer_capacity=64,
                      anticipatory_param=1., batch_size=16, learn_every=16,
                      rl_learning_rate=.001, sl_learning_rate=.001,
                      min_buffer_size_to_learn=16, replay_buffer_capacity=64,
                      update_target_network_every=16, weight_update_coeff=.995,
                      discount_factor=1., epsilon_start=.1, epsilon_end=.01,
                      epsilon_decay_duration=100, epsilon_decay_schedule_str="linear",
                      optimizer_str="adam", loss_str="huber", gradient_clipping=10.)
        learner = VectorNFSP((4,), config, torch.device("cuda:0"), 127)
        learner.begin((0, 1))
        initial = {name: next(agent._avg_network.parameters()).detach().clone()
                   for name, agent in learner.agents.items()}
        for frame in range(32):
            previous = {slot: {name: np.full(4, slot * 100 + frame, np.float32)
                               for name in AGENTS} for slot in (0, 1)}
            current = {slot: {name: value + 1 for name, value in players.items()}
                       for slot, players in previous.items()}
            actions = learner.act(previous)
            rewards = {slot: dict(zip(AGENTS, (1., -1.))) for slot in previous}
            terms = {slot: dict.fromkeys(AGENTS, frame == 31) for slot in previous}
            truncs = {slot: dict.fromkeys(AGENTS, False) for slot in previous}
            learner.feed(previous, actions, current, rewards, terms, truncs)
        for name, agent in learner.agents.items():
            replay = agent._rl_agent.replay_buffer.experience
            np.testing.assert_array_equal(replay.next_info_state - replay.info_state, 1.)
            self.assertEqual(int(replay.is_final_step.sum()), 2)
            self.assertEqual(learner.updates[name], {"rl": 4, "sl": 4})
            self.assertFalse(torch.equal(initial[name], next(agent._avg_network.parameters())))
            self.assertIsNone(agent._prev_timestep)
            self.assertIsNone(agent._rl_agent.prev_timestep)
        mode = learner.modes[1, AGENTS[0]]
        learner.begin((0,))
        self.assertEqual(learner.modes[1, AGENTS[0]], mode)


if __name__ == "__main__":
    unittest.main()
