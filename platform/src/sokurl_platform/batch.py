"""Owned process, reset and frame scheduling with caller-supplied observers."""
import ctypes
from sokurl_platform import sokurl
from sokurl_platform.bridge_shared import BridgeClient, FRAME_RING_CAPACITY, wait_for_steps
from sokurl_platform.frames import FRAME_SIZE, drain_frames_into, wait_for_frame_zero
from sokurl_platform.match import MatchConfig

RESET_METHODS = {"image": "process_restart", "state": "native_scene_reload",
    "diagnostic_state": "native_scene_reload", "privileged_state": "native_scene_reload",
    "animation": "native_scene_reload"}

class EngineBatch:
    def __init__(self, launch_timeout):
        if launch_timeout <= 0:
            raise ValueError("launch timeout must be positive")
        self.launch_timeout = launch_timeout
        self.processes = {}
        self.clients = {}
        self.buffers = {}
        self.frames = {}
        self.active = set()
        self.readers = {}
        self.observation_mode = "diagnostic_state"
        self.recording_enabled = False
        self.recordings = {}
        self.completed_replays = []
        self.slot_matches = {}
        self.fresh_reset = False
        self.cards_disabled = False
        self.fixed_weather = None

    def enable_recording(self):
        if self.processes:
            raise RuntimeError("enable replay recording before launching a game")
        self.recording_enabled = True

    def _finish_recording(self, slot, reason):
        if slot in self.recordings:
            recording = self.recordings.pop(slot)
            self.completed_replays.append(recording.finish(self.clients[slot], reason))

    def take_replays(self):
        completed, self.completed_replays = self.completed_replays, []
        return completed


    def _observe(self, slot, raw, dropped, *, snapshot_validated=False):
        if dropped:
            raise RuntimeError("frame recording was incomplete")
        return self.readers[slot].read(raw, self.clients[slot],
                                      known_dropped=dropped if snapshot_validated else None)

    def reset(self, seeds):
        if self.processes or not seeds:
            raise ValueError("legacy batch reset requires a fresh batch")
        return self.reset_slots(dict(enumerate(seeds)))

    def reset_slots(self, seeds):
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("nonempty nonnegative slot IDs are required")
        if any(type(s) is not int or not 0 <= s < 0xFFFFFFFF for s in seeds.values()):
            raise ValueError("native seed 0xFFFFFFFF is reserved; use a smaller uint32")
        for slot in seeds:
            self._finish_recording(slot, "reset")
        if RESET_METHODS[self.observation_mode] == "process_restart":
            # Scene reload preserves renderer state that changes pixels across
            # episodes. Recreate only the selected image slots; state-only
            # observations have passed native scene-reload determinism checks.
            self._close_slots(set(seeds) & set(self.processes))
        existing = set(seeds) & set(self.processes)
        fresh = {slot: seed for slot, seed in seeds.items() if slot not in existing}
        self.active.difference_update(seeds)
        pending = {}
        for slot in sorted(existing):
            client = self.clients[slot]
            segment = (client.snapshot().latest.segmentId + 1) & 0xFFFFFFFF
            pending[slot] = (client.reset_episode(seeds[slot]), segment)
        states = self._launch_slots(fresh) if fresh else {}
        for slot, (sequence, segment) in pending.items():
            try:
                raw = self.clients[slot].wait_for_reset(sequence, segment, seeds[slot], self.launch_timeout)
            except TimeoutError as error:
                pid = self.processes[slot].pid
                try:
                    live = sokurl._read_process_values(pid)
                    detail = f"live_scene={live[0]} live_mode={live[1]} characters={live[2:4]}"
                except OSError as read_error:
                    detail = f"live process read failed: {read_error!r}"
                raise TimeoutError(f"{error}; {detail}") from error
            self.frames[slot] = 0
            states[slot] = self._observe(slot, raw, 0)
        self.active.update(seeds)
        if self.recording_enabled:
            for slot, seed in seeds.items():
                self.recordings[slot] = self._create_recording(self.processes[slot].pid, slot, seed)
        return states

    def reset_matchups(self, seeds, matches):
        if not seeds or set(seeds) != set(matches):
            raise ValueError("one match configuration is required per reset slot")
        if any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("nonnegative slot IDs are required")
        if any(type(seed) is not int or not 0 <= seed < 0xFFFFFFFF for seed in seeds.values()):
            raise ValueError("native seed 0xFFFFFFFF is reserved; use a smaller uint32")
        selected = {slot: MatchConfig(**match) for slot, match in matches.items()}
        for match in selected.values():
            self._check_match(match)
        changed = {slot for slot, match in selected.items() if slot in self.processes
                   and self.slot_matches.get(slot, self.match) != match}
        # Character selection belongs to process startup. Recreate only the
        # selected games whose matchup changed; unrelated slots remain paused.
        self._close_slots(changed)
        self.slot_matches.update(selected)
        return self.reset_slots(seeds)

    def _check_match(self, match):
        for player in (match.player_0, match.player_1):
            if player.palette or player.deck:
                raise NotImplementedError("this Platform adapter supports palette/deck zero only")
            if self.fresh_reset and player.character not in (0, 1):
                raise NotImplementedError("fresh reset is verified only for Reimu/Marisa")

    def _launch_slots(self, seeds):
        # Group identical startup settings; keep original slot order in replies.
        owned = {}
        try:
            groups = {}
            for slot in seeds:
                match = self.slot_matches.get(slot, self.match)
                self._check_match(match)
                groups.setdefault(match, []).append(slot)
            for match, slots in groups.items():
                processes = sokurl._launch_vs_group_from_title(
                    len(slots), self.launch_timeout, headless=True, unlimited=True,
                    seeds=tuple(seeds[slot] for slot in slots), pause_at_start=True,
                    capture_images=self.observation_mode == "image",
                    capture_state=self.observation_mode in {"state", "animation"},
                    characters=(match.player_0.character, match.player_1.character),
                    cards_disabled=self.cards_disabled,
                    original_replay_recording=self.recording_enabled)
                owned.update(zip(slots, processes, strict=True))
                self.processes.update(zip(slots, processes, strict=True))
        except BaseException:
            self._close_slots(set(owned))
            raise
        processes = [owned[slot] for slot in seeds]
        self.processes.update(zip(seeds, processes, strict=True))
        states = {}
        try:
            for slot, process in zip(seeds, processes, strict=True):
                client = BridgeClient(process.pid)
                self.clients[slot] = client
                if self.recording_enabled and not client.original_replay_recording_enabled:
                    raise RuntimeError("installed Bridge lacks native REP input-boundary protection")
                if self.fresh_reset and not client.fresh_reset_enabled:
                    raise RuntimeError("installed Bridge lacks requested fresh-reset capability")
                if self.fixed_weather:
                    client.wait_for_ack(client.set_fixed_weather(self.fixed_weather), self.launch_timeout)
                    # A weather change invalidates the old checkpoint. Start a
                    # genuinely new episode through the existing reset command.
                    segment = (client.snapshot().latest.segmentId + 1) & 0xFFFFFFFF
                    sequence = client.reset_episode(seeds[slot])
                    raw = client.wait_for_reset(sequence, segment, seeds[slot], self.launch_timeout)
                else:
                    raw = wait_for_frame_zero(client, process.pid, self.launch_timeout)
                self.buffers[slot] = (ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))()
                self.frames[slot] = 0
                self.readers[slot] = self._create_reader(process.pid)
                states[slot] = self._observe(slot, raw, 0)
                self.active.add(slot)
        except BaseException:
            self._close_slots(set(seeds))
            raise
        return states

    def step(self, actions):
        if not actions or not set(actions) <= self.active:
            raise ValueError("joint actions must refer to active games")
        slots = list(actions)
        for joint in actions.values():
            if len(joint) != 2:
                raise ValueError("two actions are required")
            for action in joint:
                values = action.inputs
                if (len(values) != 8 or any(type(v) is not int for v in values)
                        or any(v not in (-1, 0, 1) for v in values[:2])
                        or any(v not in (0, 1) for v in values[2:])):
                    raise ValueError("invalid logical input")
        sequences = [self.clients[s].step_with_inputs(actions[s][0].inputs, actions[s][1].inputs)
                     for s in slots]
        for slot in slots:
            self.frames[slot] += 1
        snapshots = wait_for_steps([self.clients[s] for s in slots], sequences,
                                   [self.frames[s] for s in slots], 10.0)
        states = {}
        for slot, snapshot in zip(slots, snapshots, strict=True):
            states[slot] = self._observe(slot, snapshot.latest, snapshot.dropped_frames,
                                         snapshot_validated=True)
            drain_frames_into(self.clients[slot], self.buffers[slot])
            if states[slot].ended:
                self._finish_recording(slot, states[slot].outcome.value)
            elif self.recording_enabled and states[slot].frame >= self.max_frames:
                self._finish_recording(slot, "time_limit")
        self.active.difference_update(s for s, state in states.items() if state.ended)
        return states

    def status(self, slots):
        if not slots or not set(slots) <= set(self.clients):
            raise ValueError("status requires owned game slots")
        result = {}
        for slot in slots:
            snapshot = self.clients[slot].snapshot()
            result[slot] = {"pid": self.processes[slot].pid, "frame": snapshot.game_frame,
                            "segment": snapshot.latest.segmentId, "seed": snapshot.latest.randomSeed,
                            "weather": snapshot.latest.activeWeather,
                            "hash": f"{snapshot.latest.stateHash:016X}",
                            "fresh_reset": self.clients[slot].fresh_reset_enabled}
        return result

    def close(self):
        self._close_slots(set(self.processes))

    def _close_slots(self, slots):
        errors = []
        for slot in slots:
            try:
                self._finish_recording(slot, "close")
            except Exception as error:
                errors.append(repr(error))
            for resources in (self.readers, self.clients):
                try:
                    if slot in resources:
                        resources.pop(slot).close()
                except Exception as error:
                    errors.append(repr(error))
            try:
                process = self.processes[slot]
                if process.is_running():
                    sokurl.shutdown(5.0, process.pid)
                del self.processes[slot]
            except Exception as error:
                errors.append(repr(error))
            self.buffers.pop(slot, 0)
            self.frames.pop(slot, 0)
            self.active.discard(slot)
        if errors:
            raise RuntimeError(f"failed to close owned game resources: {errors}")
