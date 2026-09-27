from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import bridge_shared


class BridgeProtocolTests(unittest.TestCase):
    def test_control_block_layout(self) -> None:
        self.assertEqual(ctypes.sizeof(bridge_shared.ControlBlock), 80)
        self.assertEqual(bridge_shared.ControlBlock.commandSeq.offset, 12)
        self.assertEqual(bridge_shared.ControlBlock.horizontalAxis.offset, 24)
        self.assertEqual(bridge_shared.ControlBlock.durationFrames.offset, 48)
        self.assertEqual(bridge_shared.ControlBlock.gameFrame.offset, 56)
        self.assertEqual(bridge_shared.ControlBlock.resultCode.offset, 72)

    def test_required_actions(self) -> None:
        expected = {
            "NEUTRAL", "LEFT", "RIGHT", "UP", "DOWN",
            "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
            "A", "B", "C", "D",
            "LEFT_A", "RIGHT_A", "DOWN_A", "UP_A",
            "LEFT_B", "RIGHT_B", "DOWN_B", "UP_B",
            "LEFT_C", "RIGHT_C", "DOWN_C", "UP_C",
        }
        self.assertEqual(set(bridge_shared.ACTION_INPUTS), expected)

    def test_axis_convention_matches_sokulib(self) -> None:
        self.assertEqual(bridge_shared.ACTION_INPUTS["LEFT"][:2], (-1, 0))
        self.assertEqual(bridge_shared.ACTION_INPUTS["RIGHT"][:2], (1, 0))
        self.assertEqual(bridge_shared.ACTION_INPUTS["UP"][:2], (0, -1))
        self.assertEqual(bridge_shared.ACTION_INPUTS["DOWN"][:2], (0, 1))


if __name__ == "__main__":
    unittest.main()
