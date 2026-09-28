from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from bridge_shared import BridgeClient
import sokurl


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"


def wait_stable(pid: int, frames: int, timeout: float = 30.0) -> dict[str, int]:
    client = BridgeClient(pid)
    try:
        deadline = time.monotonic() + timeout
        start = None
        while time.monotonic() < deadline:
            snapshot = client.snapshot()
            if (
                snapshot.in_gameplay
                and snapshot.latest.sceneId == sokurl.SCENE_BATTLE
                and snapshot.latest.battleMode == sokurl.BATTLE_MODE_VSPLAYER
            ):
                if start is None:
                    start = snapshot.game_frame
                if snapshot.game_frame >= start + frames:
                    return {
                        "start_frame": start,
                        "end_frame": snapshot.game_frame,
                        "p1": snapshot.latest.p1.characterId,
                        "p2": snapshot.latest.p2.characterId,
                    }
            time.sleep(0.02)
        raise RuntimeError(f"PID {pid} did not sustain {frames} VS frames")
    finally:
        client.close()


def run(count: int, frames: int, headless: bool = False) -> dict[str, object]:
    results = []
    for index in range(1, count + 1):
        process = sokurl._launch_vs_from_title(30.0, headless=headless)
        started = time.monotonic()
        try:
            stable = wait_stable(process.pid, frames)
            result = {
                "run": index,
                "pid": process.pid,
                "seconds": round(time.monotonic() - started, 3),
                **stable,
                "success": stable["p1"] == 1 and stable["p2"] == 0,
            }
            if not result["success"]:
                raise RuntimeError(f"run {index} loaded wrong characters: {result}")
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            if process.is_running():
                sokurl.shutdown(5.0, process.pid)
    return {
        "runs": count,
        "headless": headless,
        "frames_per_run": frames,
        "successful_runs": sum(bool(item["success"]) for item in results),
        "results": results,
        "success": len(results) == count and all(item["success"] for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    report = run(args.runs, args.frames, args.headless)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "headless" if args.headless else "rendered"
    path = REPORT_DIR / f"vs-stress-{mode}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))
    print(f"report={path}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
