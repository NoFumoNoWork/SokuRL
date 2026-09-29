from __future__ import annotations

import ctypes
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import bridge_shared
from frame_runtime import (
    CheckpointFingerprint,
    RingAccounting,
    SimulationAccounting,
    first_hash_divergence,
    goto_plan,
    step_back_target,
)
from raw_recorder import FRAME_FIELDS, INPUT_FIELDS, RawSessionWriter, frame_row, input_row
from scenario_runner import compile_steps, token_input


class BridgeProtocolTests(unittest.TestCase):
    def test_cpp_python_layout_agreement(self) -> None:
        self.assertEqual(bridge_shared.CONTROL_VERSION, 7)
        self.assertEqual(ctypes.sizeof(bridge_shared.LogicalInput), 32)
        self.assertEqual(ctypes.sizeof(bridge_shared.PlayerState), 140)
        self.assertEqual(ctypes.sizeof(bridge_shared.ObjectState), 80)
        self.assertEqual(ctypes.sizeof(bridge_shared.SimplePlayerState), 40)
        self.assertEqual(ctypes.sizeof(bridge_shared.SimpleStatePatch), 96)
        self.assertEqual(ctypes.sizeof(bridge_shared.ReconstructionFrame), 168)
        self.assertEqual(ctypes.sizeof(bridge_shared.RawFrameState), 10596)
        self.assertEqual(ctypes.sizeof(bridge_shared.ControlBlock), 10884)
        self.assertEqual(bridge_shared.ControlBlock.commandSeq.offset, 16)
        self.assertEqual(bridge_shared.ControlBlock.commandInput.offset, 32)
        self.assertEqual(bridge_shared.ControlBlock.currentFrame.offset, 144)
        self.assertEqual(bridge_shared.ControlBlock.latest.offset, 288)
        header = (ROOT / "native" / "SokuRLBridge" / "ControlBlock.hpp").read_text(encoding="utf-8")
        for assertion in (
            "sizeof(LogicalInput) == 32", "sizeof(PlayerState) == 140",
            "sizeof(ObjectState) == 80", "sizeof(SimpleStatePatch) == 96",
            "sizeof(ReconstructionFrame) == 168",
            "sizeof(RawFrameState) == 10596", "sizeof(ControlBlock) == 10884",
            "offsetof(ControlBlock, currentFrame) == 144",
        ):
            self.assertIn(assertion, header)
        launcher = (ROOT / "tools" / "sokurl.py").read_text(encoding="utf-8")
        self.assertIn("version in (4, 5, 6, 7)", launcher)

    def test_spirit_uses_signed_game_semantics(self) -> None:
        player_fields = dict(bridge_shared.PlayerState._fields_)
        simple_fields = dict(bridge_shared.SimplePlayerState._fields_)
        self.assertIs(player_fields["spirit"], ctypes.c_int32)
        self.assertIs(player_fields["maxSpirit"], ctypes.c_int32)
        self.assertIs(simple_fields["spirit"], ctypes.c_int32)
        for raw, expected in ((0, 0), (32767, 32767), (0x8000, -32768), (0xFFA8, -88)):
            state = bridge_shared.PlayerState()
            state.spirit = ctypes.c_int16(raw).value
            self.assertEqual(state.spirit, expected)

    def test_simple_patch_rejects_spirit_outside_game_storage(self) -> None:
        patch = bridge_shared.SimpleStatePatch()
        for value in (-32768, 32767):
            patch.p1.spirit = value
            patch.p1.maxSpirit = value
            bridge_shared.validate_simple_patch(patch)
        for value in (-32769, 32768):
            patch.p1.spirit = value
            with self.assertRaises(ValueError):
                bridge_shared.validate_simple_patch(patch)

    def test_command_protocol_values(self) -> None:
        self.assertEqual(bridge_shared.COMMAND_RUN, 3)
        self.assertEqual(bridge_shared.COMMAND_PAUSE, 4)
        self.assertEqual(bridge_shared.COMMAND_STEP_FRAMES, 5)
        self.assertEqual(bridge_shared.COMMAND_ESTABLISH_CHECKPOINT, 6)
        self.assertEqual(bridge_shared.COMMAND_GOTO_FRAME, 7)
        self.assertEqual(bridge_shared.COMMAND_MENU_CONFIRM, 8)
        self.assertEqual(bridge_shared.COMMAND_STEP_WITH_INPUTS, 9)
        self.assertEqual(bridge_shared.COMMAND_APPLY_SIMPLE_STATE, 10)
        self.assertEqual(bridge_shared.RESULT_NAMES[12], "CHECKPOINT_RESTORE_UNSUPPORTED")

    def test_required_actions_and_axis_convention(self) -> None:
        self.assertEqual(bridge_shared.ACTION_INPUTS["LEFT"][:2], (-1, 0))
        self.assertEqual(bridge_shared.ACTION_INPUTS["RIGHT"][:2], (1, 0))
        self.assertEqual(bridge_shared.ACTION_INPUTS["UP"][:2], (0, -1))
        self.assertEqual(bridge_shared.ACTION_INPUTS["DOWN"][:2], (0, 1))
        self.assertTrue(all(len(value) == 8 for value in bridge_shared.ACTION_INPUTS.values()))

    def test_pause_and_step_accounting(self) -> None:
        runtime = SimulationAccounting(frame=20, paused=True)
        self.assertEqual(runtime.run_update(), 20)
        self.assertEqual(runtime.step(1), 21)
        self.assertEqual(runtime.step(10), 31)
        runtime.paused = False
        self.assertEqual(runtime.run_update(), 32)

    def test_monotonic_frame_ids(self) -> None:
        runtime = SimulationAccounting(paused=False)
        values = [runtime.run_update() for _ in range(20)]
        self.assertEqual(values, list(range(1, 21)))

    def test_goto_and_step_back_calculations(self) -> None:
        self.assertEqual(goto_plan(100, 50, 301, True), "restart-and-resimulate")
        self.assertEqual(goto_plan(100, 100, 301, True), "already-there")
        self.assertEqual(step_back_target(100), 99)
        with self.assertRaises(ValueError):
            goto_plan(100, 400, 301, True)
        with self.assertRaises(ValueError):
            step_back_target(0)

    def test_checkpoint_invalidation(self) -> None:
        baseline = CheckpointFingerprint(1, 2, 3, 42, (0, 1, 2, 3, 4, 5))
        self.assertTrue(baseline.remains_valid(baseline))
        changed = CheckpointFingerprint(1, 2, 4, 42, (0, 1, 2, 3, 4, 5))
        self.assertFalse(baseline.remains_valid(changed))

    def test_state_hash_is_stable_and_sensitive(self) -> None:
        state = bridge_shared.RawFrameState()
        state.frameId = 7
        state.p1.x = 123.5
        first = bridge_shared.calculate_state_hash(state)
        self.assertEqual(first, bridge_shared.calculate_state_hash(state))
        state.p1.hp = 9999
        self.assertNotEqual(first, bridge_shared.calculate_state_hash(state))

    def test_state_hash_includes_projectile_state(self) -> None:
        state = bridge_shared.RawFrameState()
        first = bridge_shared.calculate_state_hash(state)
        state.p1ObjectCount = 1
        state.p1Objects[0].typeId = 0x12345678
        state.p1Objects[0].x = 42.5
        self.assertNotEqual(first, bridge_shared.calculate_state_hash(state))

    def test_simple_patch_layout_covers_only_documented_scalars(self) -> None:
        fields = {name for name, _ in bridge_shared.SimpleStatePatch._fields_}
        self.assertEqual(fields, {
            "timeElapsedRaw", "activeWeather", "displayedWeather", "weatherCounter",
            "p1", "p2",
        })
        player_fields = {name for name, _ in bridge_shared.SimplePlayerState._fields_}
        self.assertNotIn("actionId", player_fields)
        self.assertNotIn("objectCount", player_fields)
        self.assertIn("hp", player_fields)
        self.assertIn("spirit", player_fields)

    def test_divergence_harness_reports_first_mismatch(self) -> None:
        self.assertIsNone(first_hash_divergence([10, 20, 30], [10, 20, 30]))
        self.assertEqual(first_hash_divergence([10, 20, 30], [10, 99, 30]), 1)
        self.assertEqual(first_hash_divergence([10, 20, 30], [10, 20]), 2)

    def test_input_log_serialization_has_both_players(self) -> None:
        state = bridge_shared.RawFrameState()
        state.frameId = 12
        state.p1.input.horizontalAxis = 1
        state.p2.input.b = 1
        row = input_row(state)
        self.assertEqual(row["p1_horizontalAxis"], 1)
        self.assertEqual(row["p2_b"], 1)
        self.assertEqual(set(row), set(INPUT_FIELDS))

    def test_csv_schema_is_fixed(self) -> None:
        state = bridge_shared.RawFrameState()
        self.assertEqual(set(frame_row(state)), set(FRAME_FIELDS))
        self.assertEqual(set(input_row(state)), set(INPUT_FIELDS))
        self.assertIn("p1_frameFlags", FRAME_FIELDS)
        self.assertIn("p2_hand_4", FRAME_FIELDS)

    def test_writer_outputs_manifest_and_zero_drop_validity(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            writer = RawSessionWriter(Path(directory))
            state = bridge_shared.RawFrameState()
            state.frameId = 0
            writer.write([state])
            writer.close(dropped_frames=0, validation="UNKNOWN")
            manifest = json.loads((writer.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frames"], 1)
            self.assertTrue(manifest["valid_lossless_recording"])
            self.assertTrue((writer.path / "frames_000.csv").exists())
            self.assertTrue((writer.path / "inputs_000.csv").exists())

    def test_ring_overflow_detection(self) -> None:
        ring = RingAccounting(capacity=2)
        self.assertTrue(ring.push())
        self.assertTrue(ring.push())
        self.assertFalse(ring.push())
        self.assertEqual(ring.dropped, 1)
        self.assertEqual(ring.drain(1), 1)
        self.assertTrue(ring.push())

    def test_scenario_inputs_are_facing_relative(self) -> None:
        self.assertEqual(token_input("4C", 1)[:5], (-1, 0, 0, 0, 1))
        self.assertEqual(token_input("4C", -1)[:5], (1, 0, 0, 0, 1))
        self.assertEqual(token_input("5B", 1)[:5], (0, 0, 0, 1, 0))

    def test_scenario_raw_and_repeat_compile_per_frame(self) -> None:
        steps = [
            {"wait": 2},
            {"raw": [{"input": "2", "frames": 2}, "4C"]},
            {"repeat": {"times": 2, "steps": [{"input": "5B"}]}},
        ]
        compiled = compile_steps(steps, 1)
        self.assertEqual(len(compiled), 7)
        self.assertEqual(compiled[2][:2], (0, 1))
        self.assertEqual(compiled[4][:5], (-1, 0, 0, 0, 1))
        self.assertEqual(compiled[5][:5], (0, 0, 0, 1, 0))


if __name__ == "__main__":
    unittest.main()
