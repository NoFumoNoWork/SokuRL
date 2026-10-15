"""Run live policies with explicit gap recovery, without advancing the game."""
from dataclasses import asdict
import time

from soku_rl.env.encoding import decode_action
from soku_rl.play.live_policy import LivePolicy
from soku_rl.play.match import MatchLifecycle
from soku_rl.play.match_policies import PlayStep


class RealtimePolicy:
    def __init__(self, policy, interface, seat, seed):
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("play seed must be a supported uint32")
        self.live = LivePolicy(policy, interface, seat)
        self.lifecycle = MatchLifecycle(2)
        self.seat, self.seed, self.instances = seat, seed, 0
        self.match, self.frame = 0, -1
        self.live.prepare((seed + seat) % 0xFFFFFFFF)

    def advance(self, frame):
        state = frame.match
        if state.phase == "battle":
            if state.match == self.match:
                if state.frame != self.frame + 1:
                    raise RuntimeError("real-time policy lost a simulation frame")
            elif state.frame not in (0, 1):
                raise RuntimeError("real-time policy missed the start of a match")
            self.match, self.frame = state.match, state.frame
        events = self.lifecycle.update(state)
        kinds = {event.kind for event in events}
        if "match_started" in kinds or state.phase != "battle":
            self.live.stop()
        if state.phase != "battle":
            self.live.prepare((self.seed + 2 * self.instances + self.seat) % 0xFFFFFFFF)
        continuous = self.live.active and not self.live.reset_each_round
        if state.phase == "battle":
            if self.lifecycle.can_act:
                if not self.live.active or "round_started" in kinds and not continuous:
                    seed = (self.seed + 2 * self.instances + self.seat) % 0xFFFFFFFF
                    self.live.start_round(state.frame, frame.observations, seed)
                    self.instances += 1
                else:
                    self.live.observe(state.frame, frame.observations)
            elif continuous:
                self.live.observe(state.frame, frame.observations)
            else:
                self.live.stop()
        inputs = {}
        if self.live.decision_due:
            inputs[self.seat] = decode_action(self.live.act()).inputs
        return PlayStep(self.lifecycle.phase, events, inputs)

    def stop(self):
        self.live.stop()
        self.live.prepared.clear()

    def catch_up(self, frame):
        """Reset private policy memory after a gap without resetting the match."""
        state = frame.match
        if state.phase != "battle":
            return 0
        expected = self.frame + 1 if state.match == self.match else 1
        skipped = state.frame - expected
        if skipped <= 0:
            return 0
        self.stop()
        self.match, self.frame = state.match, state.frame - 1
        return skipped


def run_session(connection, controller, matches, timeout, record):
    if type(matches) is not int or matches < 0 or timeout < 0:
        raise ValueError("matches and timeout must be nonnegative; zero means wait for player exit")
    started = time.monotonic()
    completed, decisions, frames, scores = 0, 0, 0, (0, 0)
    submitted, busy = 0, 0
    recoveries, skipped_frames, policy_timeouts = 0, 0, 0

    def result(termination):
        return dict(termination=termination, matches=completed, decisions=decisions, frames=frames,
                    submitted=submitted, busy=busy, scores=scores, seconds=time.monotonic()-started,
                    recoveries=recoveries, skipped_frames=skipped_frames, policy_timeouts=policy_timeouts)

    try:
        while not timeout or time.monotonic()-started < timeout:
            batch = connection.request("poll", {})
            if batch["closed"]:
                return result(batch["termination"])
            for event in batch["events"]:
                record(event)
            for item in batch["records"]:
                frame = item["frame"]
                skipped = controller.catch_up(frame)
                if skipped:
                    recoveries += 1
                    skipped_frames += skipped
                    record({"kind": "policy_resync", "match": frame.match.match,
                            "frame": frame.match.frame, "skipped": skipped})
                before = time.perf_counter_ns()
                try:
                    step = controller.advance(frame)
                except TimeoutError as error:
                    controller.stop()
                    policy_timeouts += 1
                    record({"kind": "policy_timeout", "match": frame.match.match,
                            "frame": frame.match.frame, "reason": str(error)})
                    continue
                inference_ms = (time.perf_counter_ns()-before)/1e6
                state = frame.match
                scores = state.scores
                record({"kind": "frame", **asdict(state), "phase": step.phase,
                    "events": [asdict(event) for event in step.events],
                    "engine_inputs": item["engine_inputs"], "characters": item["characters"]})
                frames += state.phase == "battle"
                if any(event.kind == "match_finished" for event in step.events):
                    completed += 1
                    if matches and completed == matches:
                        return result("matches_completed")
                if step.inputs:
                    reply = connection.request("submit", {"state": state, "keys": step.inputs[controller.seat]})
                    record({"kind": "command", "match": state.match, "round": state.round,
                            "observed": state.frame, "keys": step.inputs[controller.seat],
                            "inference_ms": inference_ms, **reply})
                    decisions += 1
                    submitted += reply["submitted"]
                    busy += not reply["submitted"]
            if not batch["records"]:
                time.sleep(.001)
        raise TimeoutError("play session reached its configured deadline")
    finally:
        controller.stop()
