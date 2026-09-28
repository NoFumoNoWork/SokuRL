from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bridge_shared import ACTION_INPUTS
from frame_validation import InputPair, launch_checkpoint
from scenario_runner import parse_script, run_script, save_anchor, load_anchor


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"


def create_anchor(name: str, frame: int) -> dict[str, object]:
    instance = launch_checkpoint()
    try:
        sequence = instance.client.run()
        acknowledged = instance.client.wait_for_ack(sequence, timeout=5.0)
        if acknowledged.ack_seq != sequence:
            raise RuntimeError("anchor setup run was not acknowledged")
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            snapshot = instance.client.snapshot()
            if snapshot.game_frame >= frame:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(f"anchor setup run stopped at frame {instance.client.snapshot().game_frame}")
        sequence = instance.client.pause()
        paused = instance.client.wait_for_ack(sequence, timeout=5.0)
        if paused.ack_seq != sequence or paused.run_state_name != "PAUSED":
            raise RuntimeError("anchor setup pause was not acknowledged")
        return save_anchor(name, instance.pid)
    finally:
        instance.close()


def validate(name: str, script_path: Path, loads: int, runs: int) -> dict[str, object]:
    anchor = create_anchor(name, 900)
    expected_hash = anchor["target_hash"]
    load_results = []
    for attempt in range(1, loads + 1):
        instance = load_anchor(name)
        try:
            snapshot = instance.client.snapshot()
            load_results.append({
                "attempt": attempt,
                "pid": instance.pid,
                "frame": snapshot.game_frame,
                "hash": f"{snapshot.latest.stateHash:016X}",
                "match": f"{snapshot.latest.stateHash:016X}" == expected_hash,
            })
        finally:
            instance.close()

    script = parse_script(script_path)
    script_results = []
    baseline_trace = None
    for attempt in range(1, runs + 1):
        instance = load_anchor(name)
        try:
            before = instance.client.snapshot()
            actor = before.latest.p1 if script.opponent == 0 else before.latest.p2
            baseline_objects = before.latest.p1ObjectCount if script.opponent == 0 else before.latest.p2ObjectCount
            result = run_script(script, instance.pid)
            compact_trace = [
                (
                    item["script_frame"], item["action"], item["sequence"],
                    item["subsequence"], item["animation_frame"], item["hitstop"],
                    item["captured_objects"], item["state_hash"],
                )
                for item in result["trace"]
            ]
            if baseline_trace is None:
                baseline_trace = compact_trace
            script_results.append({
                "attempt": attempt,
                "pid": instance.pid,
                "initial_action": actor.actionId,
                "final_hash": result["final_hash"],
                "max_objects": result["max_objects"],
                "projectiles_created": result["max_objects"] > baseline_objects,
                "trace_match": compact_trace == baseline_trace,
                "trace": result["trace"],
            })
        finally:
            instance.close()

    first_trace = script_results[0]["trace"] if script_results else []
    five_b_seen = any(item["action"] == 400 for item in first_trace)
    raw_24c_seen = any(500 <= item["action"] < 600 for item in first_trace)
    success = (
        all(item["match"] for item in load_results)
        and all(item["trace_match"] for item in script_results)
        and all(item["projectiles_created"] for item in script_results)
        and len({item["final_hash"] for item in script_results}) == 1
        and five_b_seen
    )
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": {
            "name": name,
            "target_frame": anchor["target_frame"],
            "target_hash": expected_hash,
            "setup_frames": len(anchor["setup"]),
        },
        "load_attempts": loads,
        "load_successes": sum(item["match"] for item in load_results),
        "load_results": load_results,
        "script": str(script_path),
        "script_runs": runs,
        "script_results": script_results,
        "five_b_seen": five_b_seen,
        "raw_24c_seen": raw_24c_seen,
        "success": success,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ScenarioRunner v1 runtime acceptance test")
    parser.add_argument("--loads", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    try:
        report = validate("graze_test", ROOT / "scenarios" / "graze_test.yaml", args.loads, args.runs)
    except Exception as error:
        report = {"success": False, "error": str(error)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"scenario-validation-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    summary = dict(report)
    if "script_results" in summary:
        summary["script_results"] = [
            {key: value for key, value in result.items() if key != "trace"}
            for result in summary["script_results"]
        ]
    if "load_results" in summary:
        summary.pop("load_results")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
