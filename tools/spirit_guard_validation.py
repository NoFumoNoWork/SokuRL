from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from bridge_shared import ACTION_INPUTS, BridgeClient
from frame_validation import InputPair, PracticeInstance, copy_state
from headless_validation import wait_for_frame_zero
import sokurl


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"
NEUTRAL = ACTION_INPUTS["NEUTRAL"]


def pair(p1: tuple[int, ...] = NEUTRAL, p2: tuple[int, ...] = NEUTRAL) -> InputPair:
    return InputPair(p1, p2)


def pressure_trace() -> list[InputPair]:
    trace: list[InputPair] = []
    trace.extend(pair(ACTION_INPUTS["RIGHT"], ACTION_INPUTS["LEFT"]) for _ in range(90))
    trace.extend(pair(NEUTRAL, ACTION_INPUTS["RIGHT"]) for _ in range(20))
    for button in ("B", "C", "B", "C", "B", "C"):
        trace.extend(pair(ACTION_INPUTS[button], ACTION_INPUTS["RIGHT"]) for _ in range(3))
        trace.extend(pair(NEUTRAL, ACTION_INPUTS["RIGHT"]) for _ in range(90))
    trace.extend(pair(NEUTRAL, ACTION_INPUTS["RIGHT"]) for _ in range(240))
    return trace


def run_once(seed: int, injected_spirit: int, patch_after_frame: int = 253) -> dict[str, object]:
    process = sokurl._launch_vs_from_title(
        35.0, headless=True, unlimited=True, seed=seed, pause_at_start=True
    )
    client: BridgeClient | None = None
    try:
        client = BridgeClient(process.pid)
        initial = wait_for_frame_zero(client, process.pid)
        instance = PracticeInstance(process, client, 0)
        states = [initial]
        for inputs in pressure_trace():
            states.append(instance.step(inputs))
            if states[-1].frameId == 203:
                patched = copy_state(states[-1])
                patched.p1.spirit = injected_spirit
                states[-1] = instance.apply_simple(patched)
            if states[-1].frameId == patch_after_frame:
                patched = copy_state(states[-1])
                patched.p2.spirit = 50
                states[-1] = instance.apply_simple(patched)

        negative = [state for state in states if state.p1.spirit < 0 or state.p2.spirit < 0]
        p1_minimum = min(states, key=lambda state: state.p1.spirit)
        p2_minimum = min(states, key=lambda state: state.p2.spirit)
        spirit_changes = [
            {
                "frame": current.frameId,
                "before": previous.p2.spirit,
                "after": current.p2.spirit,
                "p1_action": current.p1.actionId,
                "p2_action": current.p2.actionId,
            }
            for previous, current in zip(states, states[1:])
            if current.p2.spirit < previous.p2.spirit
        ]
        return {
            "pid": process.pid,
            "frames": len(states),
            "patch_after_frame": patch_after_frame,
            "initial_hash": f"{states[0].stateHash:016X}",
            "final_hash": f"{states[-1].stateHash:016X}",
            "trace_hashes": [state.stateHash for state in states],
            "p1_minimum_spirit": p1_minimum.p1.spirit,
            "p1_minimum_frame": p1_minimum.frameId,
            "p1_minimum_action": p1_minimum.p1.actionId,
            "p2_minimum_spirit": p2_minimum.p2.spirit,
            "p2_minimum_frame": p2_minimum.frameId,
            "p2_minimum_action": p2_minimum.p2.actionId,
            "negative_frames": [
                {"frame": state.frameId, "p1": state.p1.spirit, "p2": state.p2.spirit}
                for state in negative
            ],
            "maximum_p1_objects": max(state.p1ObjectCount for state in states),
            "p2_hp_range": [min(state.p2.hp for state in states), max(state.p2.hp for state in states)],
            "p1_x_range": [min(state.p1.x for state in states), max(state.p1.x for state in states)],
            "p2_x_range": [min(state.p2.x for state in states), max(state.p2.x for state in states)],
            "p1_actions": sorted({state.p1.actionId for state in states}),
            "p2_actions": sorted({state.p2.actionId for state in states}),
            "spirit_decreases": spirit_changes,
            "p1_final_spirit": states[-1].p1.spirit,
            "p2_final_spirit": states[-1].p2.spirit,
            "dropped_frames": client.snapshot().dropped_frames,
        }
    finally:
        if client is not None:
            client.close()
        if process.is_running():
            sokurl.shutdown(5.0, process.pid)


def run(seed: int, injected_spirit: int, runs: int) -> dict[str, object]:
    results = [run_once(seed, injected_spirit) for _ in range(runs)]
    baseline = results[0]["trace_hashes"]
    trace_matches = [result["trace_hashes"] == baseline for result in results]
    for result in results:
        result.pop("trace_hashes")
    return {
        "schema": "SokuRLSpiritGuardValidation/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "injected_p1_spirit": injected_spirit,
        "p1_injection_frame": 203,
        "p2_guard_probe_spirit": 50,
        "p2_guard_probe_frame": 253,
        "runs": results,
        "trace_matches": trace_matches,
        "success": (
            all(trace_matches)
            and all(result["p1_minimum_spirit"] == injected_spirit for result in results)
            and all(result["negative_frames"] for result in results)
            and all(result["p1_final_spirit"] >= 0 for result in results)
            and all(result["maximum_p1_objects"] > 0 for result in results)
            and all(result["dropped_frames"] == 0 for result in results)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate signed spirit under deterministic projectile guard pressure"
    )
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x534F4B55)
    parser.add_argument("--injected-spirit", type=int, default=-88)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not -32768 <= args.injected_spirit < 0:
        parser.error("--injected-spirit must be a negative signed 16-bit value")
    if args.runs < 1:
        parser.error("--runs must be positive")

    try:
        report = run(args.seed, args.injected_spirit, args.runs)
    except Exception as error:
        report = {"success": False, "error": str(error)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"spirit-guard-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
