"""Control selected seats while a local match advances one frame at a time."""
from collections import deque
from dataclasses import dataclass

from soku_rl.env.encoding import decode_action
from soku_rl.play.live_policy import LivePolicy
from soku_rl.play.match import MatchEvent, MatchLifecycle


@dataclass(frozen=True)
class PlayStep:
    phase: str
    events: tuple[MatchEvent, ...]
    inputs: dict[int, tuple[int, ...]]


class MatchPolicies:
    def __init__(self, policies, interface, seed, wins_required):
        if not policies or any(type(seat) is not int or seat not in (0, 1) for seat in policies):
            raise ValueError("play requires a policy for one or both seats")
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("play seed must be a supported uint32")
        self.seed, self.interface = seed, interface
        self.live = {seat: LivePolicy(policy, interface, seat) for seat, policy in policies.items()}
        self.pending = {seat: deque() for seat in policies}
        self.held = {seat: 256 for seat in policies}
        self.instances = {seat: 0 for seat in policies}
        self.lifecycle = MatchLifecycle(wins_required)
        self.match, self.frame = 0, -1

    def _release(self, seat):
        self.live[seat].stop()
        self.pending[seat].clear()
        self.held[seat] = 256

    def advance(self, state, observations):
        if state.phase != "battle":
            raise ValueError("local policy decisions require a paused battle state")
        expected = self.frame + 1 if state.match == self.match else 0
        if state.frame != expected:
            raise ValueError("local play requires every frame from match frame zero")
        events = self.lifecycle.update(state)
        kinds = {event.kind for event in events}
        if "match_started" in kinds:
            self.stop()
        for seat, live in self.live.items():
            continuous = live.active and not live.reset_each_round
            if self.lifecycle.can_act:
                if "round_started" in kinds and not continuous:
                    self._release(seat)
                    seed = (self.seed + 2 * self.instances[seat] + seat) % 0xFFFFFFFF
                    live.start_round(state.frame, observations, seed)
                    self.instances[seat] += 1
                else:
                    live.observe(state.frame, observations)
            elif continuous:
                # The original script keeps running through KO and the next
                # round's introduction. Do not reset or omit those frames.
                live.observe(state.frame, observations)
            else:
                self._release(seat)
            if live.decision_due:
                command = live.act()
                self.pending[seat].append((state.frame + self.interface.episode.latency_frames, command))
            if self.pending[seat] and self.pending[seat][0][0] == state.frame:
                _, self.held[seat] = self.pending[seat].popleft()
        self.match, self.frame = state.match, state.frame
        return PlayStep(self.lifecycle.phase, events,
                        {seat: decode_action(command).inputs for seat, command in self.held.items()})

    def stop(self):
        for seat in self.live:
            self._release(seat)
