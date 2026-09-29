from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from frame_validation import InputPair, input_tuple
from replay_validation import play_full_replay, replay_to_target


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"


def hand(player) -> list[int]:
    return [player.handIds[index] for index in range(min(player.cardCount, 5))]


def player_summary(state, player_index: int) -> dict[str, Any]:
    player = state.p1 if player_index == 0 else state.p2
    return {
        "frame": state.frameId,
        "state_hash": f"{state.stateHash:016X}",
        "action": player.actionId,
        "sequence": player.sequenceId,
        "subsequence": player.subsequenceId,
        "spirit": player.spirit,
        "max_spirit": player.maxSpirit,
        "card_gauge": player.cardGauge,
        "card_count": player.cardCount,
        "hand": hand(player),
        "spellcard_input": player.input.spellcard,
    }


def contiguous_segments(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    segments: list[tuple[int, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            segments.append((start, previous))
            start = frame
        previous = frame
    segments.append((start, previous))
    return segments


def negative_spirit_segments(states, player_index: int) -> list[dict[str, Any]]:
    negative = []
    for state in states:
        player = state.p1 if player_index == 0 else state.p2
        if player.spirit < 0:
            negative.append(state.frameId)
    result = []
    for start, end in contiguous_segments(negative):
        values = [
            (states[frame].p1 if player_index == 0 else states[frame].p2).spirit
            for frame in range(start, end + 1)
        ]
        result.append({
            "start": start,
            "end": end,
            "duration": end - start + 1,
            "minimum": min(values),
            "before": player_summary(states[max(0, start - 1)], player_index),
            "first": player_summary(states[start], player_index),
            "minimum_frame": player_summary(states[start + values.index(min(values))], player_index),
            "last": player_summary(states[end], player_index),
            "after": player_summary(states[min(len(states) - 1, end + 1)], player_index),
        })
    return result


def spell_segments(states, player_index: int) -> list[dict[str, Any]]:
    spell_frames: list[tuple[int, int]] = []
    for state in states:
        player = state.p1 if player_index == 0 else state.p2
        if 600 <= player.actionId <= 619:
            spell_frames.append((state.frameId, player.actionId))
    raw_segments: list[tuple[int, int, int]] = []
    for frame, action in spell_frames:
        if not raw_segments or frame != raw_segments[-1][1] + 1 or action != raw_segments[-1][2]:
            raw_segments.append((frame, frame, action))
        else:
            start, _, previous_action = raw_segments[-1]
            raw_segments[-1] = (start, frame, previous_action)
    result = []
    for start, end, action in raw_segments:
        player = states[start].p1 if player_index == 0 else states[start].p2
        expected_card = action - 400
        before = states[max(0, start - 1)]
        after = states[min(len(states) - 1, end + 1)]
        before_player = before.p1 if player_index == 0 else before.p2
        input_window = []
        for frame in range(max(0, start - 5), min(len(states), start + 6)):
            current = states[frame].p1 if player_index == 0 else states[frame].p2
            if current.input.spellcard:
                input_window.append(frame)
        result.append({
            "start": start,
            "end": end,
            "duration": end - start + 1,
            "action": action,
            "expected_card_id": expected_card,
            "expected_card_was_selected": bool(hand(before_player) and hand(before_player)[0] == expected_card),
            "spellcard_input_frames_near_start": input_window,
            "before": player_summary(before, player_index),
            "first": player_summary(states[start], player_index),
            "last": player_summary(states[end], player_index),
            "after": player_summary(after, player_index),
        })
    return result


def card_count_drops(states, player_index: int) -> list[dict[str, Any]]:
    result = []
    for frame in range(1, len(states)):
        previous = states[frame - 1].p1 if player_index == 0 else states[frame - 1].p2
        current = states[frame].p1 if player_index == 0 else states[frame].p2
        if current.cardCount < previous.cardCount:
            result.append({
                "frame": frame,
                "amount": previous.cardCount - current.cardCount,
                "before": player_summary(states[frame - 1], player_index),
                "after": player_summary(states[frame], player_index),
            })
    return result


def replay_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_lossless_contiguous(states, playback: dict[str, object]) -> None:
    if playback["dropped_frames"]:
        raise RuntimeError(
            f"replay capture dropped {playback['dropped_frames']} frames; resource analysis aborted"
        )
    if playback["first_non_contiguous_index"] is not None:
        raise RuntimeError(
            "replay capture is non-contiguous at index "
            f"{playback['first_non_contiguous_index']}; resource analysis aborted"
        )
    for index, state in enumerate(states):
        if state.frameId != index:
            raise RuntimeError(
                f"frame/index mismatch at index {index}: frameId={state.frameId}"
            )


def run(replay: Path, reconstruction_limit: int) -> dict[str, Any]:
    previous_headless = os.environ.get("SOKURL_HEADLESS_RENDER")
    previous_unlimited = os.environ.get("SOKURL_UNLIMITED_PACING")
    os.environ["SOKURL_HEADLESS_RENDER"] = "1"
    os.environ.pop("SOKURL_UNLIMITED_PACING", None)
    try:
        states, playback = play_full_replay(replay)
        require_lossless_contiguous(states, playback)
        inputs = [
            InputPair(input_tuple(state.p1.input), input_tuple(state.p2.input))
            for state in states[1:]
        ]
        players = []
        targets = []
        for player_index in (0, 1):
            negatives = negative_spirit_segments(states, player_index)
            spells = spell_segments(states, player_index)
            drops = card_count_drops(states, player_index)
            targets.extend(drop["frame"] for drop in drops if 600 <= drop["after"]["action"] <= 619)
            for segment in negatives:
                targets.extend((segment["start"], segment["end"]))
            targets.extend(segment["start"] for segment in spells)
            players.append({
                "player": player_index + 1,
                "character": (states[0].p1 if player_index == 0 else states[0].p2).characterId,
                "minimum_spirit_signed": min(
                    (state.p1 if player_index == 0 else state.p2).spirit
                    for state in states
                ),
                "maximum_spirit": max(
                    (state.p1 if player_index == 0 else state.p2).spirit for state in states
                ),
                "negative_spirit_segments": negatives,
                "spell_segments": spells,
                "card_count_drops": drops,
            })

        unique_targets = list(dict.fromkeys(int(target) for target in targets))[:reconstruction_limit]
        reconstruction = []
        for target in unique_targets:
            actual, divergence, differences = replay_to_target(replay, states, inputs, target)
            reconstruction.append({
                "target": target,
                "expected_hash": f"{states[target].stateHash:016X}",
                "actual_hash": f"{actual.stateHash:016X}",
                "first_divergent_frame": divergence,
                "differences": differences[:20],
            })

        return {
            "schema": "SokuRLReplayResourcesValidation/v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "replay_sha256": replay_sha256(replay),
            "playback": playback,
            "spirit_decode": "bridge ABI v7 sign-extends the game's signed int16 value",
            "players": players,
            "reconstruction": reconstruction,
            "success": (
                playback["dropped_frames"] == 0
                and playback["first_non_contiguous_index"] is None
                and all(item["first_divergent_frame"] is None for item in reconstruction)
            ),
        }
    finally:
        if previous_headless is None:
            os.environ.pop("SOKURL_HEADLESS_RENDER", None)
        else:
            os.environ["SOKURL_HEADLESS_RENDER"] = previous_headless
        if previous_unlimited is None:
            os.environ.pop("SOKURL_UNLIMITED_PACING", None)
        else:
            os.environ["SOKURL_UNLIMITED_PACING"] = previous_unlimited


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate replay spell-card and spirit resource state"
    )
    parser.add_argument("replay", type=Path)
    parser.add_argument("--reconstruction-limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.reconstruction_limit < 0:
        parser.error("--reconstruction-limit must not be negative")

    try:
        report = run(args.replay.resolve(), args.reconstruction_limit)
    except Exception as error:
        report = {"success": False, "error": str(error)}
    path = args.report
    if path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = REPORT_DIR / f"replay-resources-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
