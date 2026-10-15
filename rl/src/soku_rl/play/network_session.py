"""Drive one local policy through full original network matches."""
import time

from soku_rl.env.encoding import decode_action
from soku_rl.play.live_policy import LivePolicy


def run_session(connection, policy, interface, seat, seed, matches, timeout, record):
    if (type(matches) is not int or matches < 1 or timeout <= 0
            or type(seed) is not int or not 0 <= seed < 0xFFFFFFFF):
        raise ValueError("positive match count, timeout and supported uint32 seed are required")
    live = LivePolicy(policy, interface, seat)
    completed, rounds, decisions, skipped, accepted = 0, 0, 0, 0, 0
    interrupted = 0
    dropped = {"late": 0, "wrong_round": 0}
    started = time.monotonic()
    scores = (0, 0)

    def result(termination):
        return {"matches": completed, "rounds": rounds, "decisions": decisions,
                "accepted_commands": accepted, "dropped_commands": dropped,
                "skipped_decisions": skipped, "termination": termination,
                "interrupted_matches": interrupted,
                "seconds": time.monotonic()-started, "last_scores": scores}

    try:
        while time.monotonic()-started < timeout:
            batch = connection.request("poll", {})
            if batch["closed"]:
                return result("game_closed")
            for event in batch["input_events"]:
                record({"kind": "input_event", **event})
                if event["result"] == "expired" and event["command_type"] == 1:
                    raise RuntimeError(f"network input {event['request']} expired before injection")
            if batch["menu_reply"] != "not_requested":
                record({"kind": "menu", "reply": batch["menu_reply"]})
            latest = {}
            for frame in batch["records"]:
                latest[frame["match"], frame["round"]] = frame["frame"]
            for frame in batch["records"]:
                scores = frame["scores"]
                metadata = {key: value for key, value in frame.items() if key != "observations"}
                record({"kind": "frame", **metadata})
                events = {event["kind"] for event in frame["events"]}
                if "match_interrupted" in events:
                    if frame["phase"] != "menu":
                        raise ConnectionError(f"network match {frame['match']} was interrupted")
                    interrupted += 1
                if "match_finished" in events:
                    completed += 1
                    if completed == matches:
                        return result("matches_completed")
                if frame["phase"] != "battle":
                    if live.active:
                        live.stop()
                    continue
                if frame["seat"] != seat:
                    raise RuntimeError("network seat differs from the configured policy seat")
                if "round_started" in events:
                    live.start_round(frame["frame"], frame["observations"], (seed+rounds) % 0xFFFFFFFF)
                    rounds += 1
                else:
                    live.observe(frame["frame"], frame["observations"])
                if live.decision_due:
                    if frame["frame"]+interface.episode.decision_frames <= latest[frame["match"], frame["round"]]:
                        live.skip_decision()
                        skipped += 1
                        record({"kind": "decision_skipped", "match": frame["match"], "round": frame["round"],
                                "frame": frame["frame"], "reason": "newer_observation_available"})
                        continue
                    inference_started = time.perf_counter_ns()
                    command = live.act()
                    inference_ms = (time.perf_counter_ns()-inference_started)/1e6
                    submit_started = time.perf_counter_ns()
                    response = connection.request("submit", {"match": frame["match"],
                        "frame": frame["frame"], "keys": decode_action(command).inputs,
                        "duration": interface.episode.decision_frames})
                    record({"kind": "command", "match": frame["match"], "round": frame["round"],
                            "command": command, "inference_ms": inference_ms,
                            "submit_ms": (time.perf_counter_ns()-submit_started)/1e6, **response})
                    decisions += 1
                    if response["reply"] in dropped:
                        dropped[response["reply"]] += 1
                    elif response["reply"] == "accepted":
                        accepted += 1
                    else:
                        raise RuntimeError(f"network action rejected: {response['reply']} at {frame['frame']}")
            if not batch["records"]:
                time.sleep(.001)
        raise TimeoutError(f"network session completed {completed}/{matches} matches before its deadline")
    finally:
        live.stop()
