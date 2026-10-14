"""Run full local matches with common policies, observations and replay delivery."""
from dataclasses import asdict
import time

from soku_rl.play.match_policies import MatchPolicies


def run_session(connection, policies, interface, seed, matches, timeout, record):
    if type(matches) is not int or matches < 1 or timeout <= 0:
        raise ValueError("local play requires a positive match count and timeout")
    controller = MatchPolicies(policies, interface, seed, 2)
    completed, frames, scores = 0, 0, (0, 0)
    started = time.monotonic()

    def result(termination):
        return {"matches": completed, "frames": frames, "last_scores": scores,
                "termination": termination, "seconds": time.monotonic() - started}

    try:
        reply = connection.request("start", {"episode": asdict(interface.episode), "seed": seed})
        while time.monotonic() - started < timeout:
            if reply["closed"]:
                record({"kind": "game_closed", "replay_saved": reply["replay_saved"]})
                return result("game_closed")
            current = reply["frame"]
            step = controller.advance(current.match, current.observations)
            scores = current.match.scores
            record({"kind": "frame", "match": current.match.match, "round": current.match.round,
                    "frame": current.match.frame, "scores": scores, "phase": step.phase,
                    "inputs": step.inputs, "events": tuple(asdict(event) for event in step.events)})
            if step.phase == "match_finished":
                connection.request("finish", {})
                completed += 1
                if completed == matches:
                    return result("matches_completed")
                reply = connection.request("reset", {"seed": (seed + completed) % 0xFFFFFFFF})
            else:
                reply = connection.request("step", step.inputs)
                frames += 1
        raise TimeoutError(f"local session completed {completed}/{matches} matches before its deadline")
    finally:
        controller.stop()
