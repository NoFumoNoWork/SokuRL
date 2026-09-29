from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

from bridge_shared import ACTION_INPUTS, BridgeClient
from headless_validation import step_group, wait_for_frame_zero
import sokurl


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"
NEUTRAL = ACTION_INPUTS["NEUTRAL"]


def weather_values(state) -> tuple[int, int, int, int]:
    return (
        state.activeWeather,
        state.displayedWeather,
        state.weatherCounter,
        state.timeElapsedRaw,
    )


def run_once(seed: int | None, frames: int, run_index: int) -> dict[str, object]:
    process = sokurl._launch_vs_from_title(
        35.0,
        headless=True,
        unlimited=True,
        seed=seed,
        pause_at_start=True,
    )
    client = BridgeClient(process.pid)
    weather_digest = hashlib.sha256()
    state_digest = hashlib.sha256()
    trajectory: list[tuple[int, int, int, int]] = []
    transitions: list[dict[str, int]] = []
    started = time.perf_counter()
    try:
        initial = wait_for_frame_zero(client, process.pid)
        previous_pair = (initial.activeWeather, initial.displayedWeather)
        for frame in range(frames + 1):
            state = initial if frame == 0 else step_group(
                [client], NEUTRAL, NEUTRAL, frame
            )[0]
            values = weather_values(state)
            trajectory.append(values)
            weather_digest.update(struct.pack("<QIIII", frame, *values))
            state_digest.update(struct.pack("<QQ", frame, state.stateHash))
            pair = (state.activeWeather, state.displayedWeather)
            if pair != previous_pair:
                transitions.append({
                    "frame": frame,
                    "active_weather": state.activeWeather,
                    "displayed_weather": state.displayedWeather,
                    "weather_counter": state.weatherCounter,
                    "time_elapsed_raw": state.timeElapsedRaw,
                })
                previous_pair = pair
            if frame and frame % 128 == 0:
                client.drain_frames()
        snapshot = client.snapshot()
        return {
            "run": run_index,
            "requested_seed": seed,
            "initial_random_seed": initial.randomSeed,
            "pid": process.pid,
            "frames": frames,
            "elapsed_seconds": time.perf_counter() - started,
            "initial_state_hash": f"{initial.stateHash:016X}",
            "final_state_hash": f"{state.stateHash:016X}",
            "weather_trace_sha256": weather_digest.hexdigest().upper(),
            "state_trace_sha256": state_digest.hexdigest().upper(),
            "initial_weather": list(trajectory[0]),
            "final_weather": list(trajectory[-1]),
            "weather_transitions": transitions,
            "dropped_frames": snapshot.dropped_frames,
            "trajectory": trajectory,
        }
    finally:
        client.close()
        if process.is_running():
            sokurl.shutdown(5.0, process.pid)


def first_difference(
    expected: list[tuple[int, int, int, int]],
    actual: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    for frame, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left != right:
            return {
                "frame": frame,
                "expected": list(left),
                "actual": list(right),
            }
    return None


def compact_run(run: dict[str, object]) -> dict[str, object]:
    result = dict(run)
    result.pop("trajectory")
    return result


def validate(
    runs: int,
    frames: int,
    seed: int,
    control_seed: int,
    unseeded_runs: int,
) -> dict[str, object]:
    sokurl._validate_game()
    same_seed_runs = [run_once(seed, frames, index + 1) for index in range(runs)]
    baseline = same_seed_runs[0]["trajectory"]
    assert isinstance(baseline, list)
    differences = []
    for run in same_seed_runs[1:]:
        trajectory = run["trajectory"]
        assert isinstance(trajectory, list)
        difference = first_difference(baseline, trajectory)
        if difference is not None:
            difference["run"] = run["run"]
            differences.append(difference)

    control = run_once(control_seed, frames, runs + 1)
    control_trajectory = control["trajectory"]
    assert isinstance(control_trajectory, list)
    control_difference = first_difference(baseline, control_trajectory)

    unseeded = [
        run_once(None, frames, runs + index + 2)
        for index in range(unseeded_runs)
    ]
    unseeded_weather_hashes = {
        str(run["weather_trace_sha256"]) for run in unseeded
    }
    unseeded_initial_seeds = {
        int(run["initial_random_seed"]) for run in unseeded
    }

    return {
        "schema": "SokuRLWeatherResetValidation/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reset_model": "fresh th123 process recreated at the same frame-zero seed",
        "in_process_rl_reset_tested": False,
        "input_trace": "P1/P2 neutral for every simulation frame",
        "frames_per_run": frames,
        "nominal_seconds_per_run": frames / 60.0,
        "same_seed": seed,
        "same_seed_runs": [compact_run(run) for run in same_seed_runs],
        "same_seed_weather_equal": not differences,
        "same_seed_first_differences": differences,
        "different_seed": control_seed,
        "different_seed_run": compact_run(control),
        "different_seed_first_weather_difference": control_difference,
        "unseeded_runs": [compact_run(run) for run in unseeded],
        "unseeded_unique_initial_seeds": len(unseeded_initial_seeds),
        "unseeded_unique_weather_traces": len(unseeded_weather_hashes),
        "success": not differences and all(
            run["dropped_frames"] == 0
            for run in same_seed_runs + [control] + unseeded
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate weather determinism across fresh-process resets"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--frames", type=int, default=3600)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x534F4B55)
    parser.add_argument("--control-seed", type=lambda value: int(value, 0), default=0x534F4B56)
    parser.add_argument("--unseeded-runs", type=int, default=0)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if not 1 <= args.frames <= 12000:
        parser.error("--frames must be between 1 and 12000")
    if args.seed == args.control_seed:
        parser.error("--control-seed must differ from --seed")
    if args.unseeded_runs < 0:
        parser.error("--unseeded-runs must not be negative")

    try:
        report = validate(
            args.runs,
            args.frames,
            args.seed,
            args.control_seed,
            args.unseeded_runs,
        )
    except Exception as error:
        report = {"success": False, "error": str(error)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"weather-reset-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
