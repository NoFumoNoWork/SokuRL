"""Simulation-time input limits and independent episode reset."""
import unittest

from soku_rl.env.control import ControlConfig, DelayedControls
from soku_rl.env.encoding import AGENTS, decode_action


class ControlTimingTests(unittest.TestCase):
    def test_commands_arrive_after_twelve_frames(self):
        controls = DelayedControls(ControlConfig(3, 12))
        neutral = (decode_action(256),) * 2
        for frame in range(30):
            if frame % 3 == 0:
                controls.submit(frame, dict(zip(AGENTS, (frame, frame + 1))))
            actual = controls.inputs(frame)
            source = ((frame - 12) // 3) * 3
            expected = neutral if frame < 12 else (decode_action(source), decode_action(source + 1))
            self.assertEqual(actual, expected)

    def test_reset_clears_held_and_queued_inputs(self):
        controls = DelayedControls(ControlConfig(3, 12))
        for frame in range(15):
            if frame % 3 == 0:
                controls.submit(frame, dict.fromkeys(AGENTS, 0))
            controls.inputs(frame)
        controls.reset()
        controls.submit(0, dict.fromkeys(AGENTS, 575))
        self.assertEqual(controls.inputs(0), (decode_action(256),) * 2)

    def test_rejects_extra_decisions_and_skipped_frames(self):
        controls = DelayedControls(ControlConfig(3, 12))
        controls.submit(0, dict.fromkeys(AGENTS, 0))
        with self.assertRaises(ValueError):
            controls.submit(0, dict.fromkeys(AGENTS, 1))
        with self.assertRaises(ValueError):
            controls.inputs(1)
        with self.assertRaises(ValueError):
            ControlConfig(0, 12)


if __name__ == "__main__":
    unittest.main()
