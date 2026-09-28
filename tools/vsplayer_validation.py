from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from bridge_shared import ACTION_INPUTS, BridgeClient
import sokurl


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"
VSPLAYER = 3
NEUTRAL = ACTION_INPUTS["NEUTRAL"]
B = ACTION_INPUTS["B"]


def step(client: BridgeClient, p1: tuple[int, ...], p2: tuple[int, ...]):
    before = client.snapshot().game_frame
    sequence = client.step_with_inputs(p1, p2)
    client.wait_for_ack(sequence)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        snapshot = client.snapshot()
        if snapshot.game_frame == before + 1 and snapshot.run_state_name == "PAUSED":
            return snapshot
        time.sleep(0.005)
    raise RuntimeError(f"frame did not advance from {before}")


def wait_frames(client: BridgeClient, count: int):
    snapshot = None
    for _ in range(count):
        snapshot = step(client, NEUTRAL, NEUTRAL)
    return snapshot


def run() -> dict[str, object]:
    sokurl._validate_game()
    process = sokurl._launch_vs_from_title(35.0)
    client: BridgeClient | None = None
    try:
        deadline = time.monotonic() + 35.0
        confirmations = 0
        transitions: list[dict[str, int | None]] = []
        while time.monotonic() < deadline:
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError(
                    f"th123 exited after {confirmations} confirmations; transitions={transitions}"
                )
            scene, mode, _, _, left_stage, right_stage = sokurl._read_process_values(process.pid)
            if scene == sokurl.SCENE_BATTLE and mode == VSPLAYER:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("timed out entering local VS Player battle")

        if client is None:
            client = BridgeClient(process.pid)
        while time.monotonic() < deadline:
            snapshot = client.snapshot()
            if snapshot.in_gameplay and snapshot.game_frame >= 600:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("VS Player bridge did not reach an active battle")

        sequence = client.pause()
        baseline = client.wait_for_ack(sequence)
        p2_actions: list[int] = []
        p2_objects: list[int] = []
        hp_samples = [{"label": "baseline", "p1": baseline.latest.p1.hp, "p2": baseline.latest.p2.hp}]

        for _ in range(3):
            for _ in range(3):
                snapshot = step(client, NEUTRAL, B)
                p2_actions.append(snapshot.latest.p2.actionId)
                p2_objects.append(snapshot.latest.p2ObjectCount)
            for _ in range(75):
                snapshot = step(client, NEUTRAL, NEUTRAL)
                p2_actions.append(snapshot.latest.p2.actionId)
                p2_objects.append(snapshot.latest.p2ObjectCount)
        hp_samples.append({"label": "after_p2_attacks", "p1": snapshot.latest.p1.hp, "p2": snapshot.latest.p2.hp})

        p1_actions: list[int] = []
        p1_objects: list[int] = []
        for _ in range(3):
            for _ in range(3):
                snapshot = step(client, B, NEUTRAL)
                p1_actions.append(snapshot.latest.p1.actionId)
                p1_objects.append(snapshot.latest.p1ObjectCount)
            for _ in range(75):
                snapshot = step(client, NEUTRAL, NEUTRAL)
                p1_actions.append(snapshot.latest.p1.actionId)
                p1_objects.append(snapshot.latest.p1ObjectCount)
        hp_samples.append({"label": "after_p1_attacks", "p1": snapshot.latest.p1.hp, "p2": snapshot.latest.p2.hp})

        damaged = snapshot.latest.p1.hp < 10000 or snapshot.latest.p2.hp < 10000
        damaged_hp = (snapshot.latest.p1.hp, snapshot.latest.p2.hp)
        snapshot = wait_frames(client, 180)
        hp_samples.append({"label": "after_wait_180", "p1": snapshot.latest.p1.hp, "p2": snapshot.latest.p2.hp})
        hp_stayed_damaged = damaged and (snapshot.latest.p1.hp, snapshot.latest.p2.hp) == damaged_hp

        return {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "pid": process.pid,
            "mode": baseline.latest.battleMode,
            "confirmations": confirmations,
            "transitions": transitions,
            "p1_character": baseline.latest.p1.characterId,
            "p2_character": baseline.latest.p2.characterId,
            "p1_attack_seen": 400 in p1_actions,
            "p2_attack_seen": 400 in p2_actions,
            "max_p1_objects": max(p1_objects, default=0),
            "max_p2_objects": max(p2_objects, default=0),
            "hp_samples": hp_samples,
            "damage_observed": damaged,
            "hp_stayed_damaged_for_180_frames": hp_stayed_damaged,
            "success": (
                baseline.latest.battleMode == VSPLAYER
                and 400 in p1_actions
                and 400 in p2_actions
                and max(p1_objects, default=0) > 0
                and max(p2_objects, default=0) > 0
                and hp_stayed_damaged
            ),
        }
    finally:
        if client is not None:
            client.close()
        if process.is_running():
            sokurl.shutdown(5.0, process.pid)


def main() -> int:
    try:
        report = run()
    except Exception as error:
        report = {"success": False, "error": str(error)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"vsplayer-validation-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
