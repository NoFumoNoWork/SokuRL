"""Adapt owned th123 processes to the simultaneous two-player game contract."""
import ctypes

from soku_rl.observations import observe
from soku_rl.visible_state import observe_visible_states
from soku_rl.visibility import VisibilityConfig
from soku_rl.pomg import Outcome, TimeStep
from bridge_shared import BridgeClient, FRAME_RING_CAPACITY, wait_for_steps
from headless_validation import wait_for_frame_zero
from unlimited_benchmark import FRAME_SIZE, _drain_fast
import sokurl


def _time_step(raw, dropped_frames, observations, pid):
    if raw.sceneId != sokurl.SCENE_BATTLE or raw.battleMode != sokurl.BATTLE_MODE_VSPLAYER:
        raise RuntimeError("game left VS battle")
    if dropped_frames:
        raise RuntimeError("frame recording was incomplete")
    hp = raw.p1.hp, raw.p2.hp
    if hp[0] <= 0 and hp[1] <= 0:
        outcome, rewards = Outcome.DRAW, (0.0, 0.0)
    elif hp[0] <= 0:
        outcome, rewards = Outcome.P2_WIN, (-1.0, 1.0)
    elif hp[1] <= 0:
        outcome, rewards = Outcome.P1_WIN, (1.0, -1.0)
    else:
        outcome, rewards = Outcome.ONGOING, (0.0, 0.0)
    return TimeStep(raw.frameId, observations, rewards, outcome, {
        "pid": pid, "segment": raw.segmentId,
        "hp": hp, "characters": (raw.p1.characterId, raw.p2.characterId),
        "stage": raw.stageId, "weather": raw.activeWeather,
        "hash": f"{raw.stateHash:016X}", "dropped_frames": dropped_frames,
        "objects": (raw.p1ObjectCount, raw.p2ObjectCount),
    })


class SokuGameBatch:
    def __init__(self, launch_timeout):
        if launch_timeout <= 0:
            raise ValueError("launch timeout must be positive")
        self.launch_timeout = launch_timeout
        self.processes = {}
        self.clients = {}
        self.buffers = {}
        self.frames = {}
        self.active = set()
        self.image_clients = {}
        self.observation_mode = "diagnostic_state"

    def configure_observation(self, configuration):
        mode = configuration["mode"]
        if self.processes or mode not in {"image", "state", "diagnostic_state"}:
            raise ValueError("set a supported observation mode before launching games")
        self.observation_mode = mode
        self.visibility = VisibilityConfig(**configuration["visibility"])

    def _observe(self, slot, raw, dropped):
        if self.observation_mode == "image":
            scene = self.image_clients[slot].read(int(raw.frameId), 10.0)
            observations = (scene.image, scene.image)
        elif self.observation_mode == "state":
            render = self.image_clients[slot].read_state(int(raw.frameId), 10.0)
            observations = observe_visible_states(raw, render, self.visibility)
        else:
            observations = tuple(observe(raw, p) for p in (0, 1))
        return _time_step(raw, dropped, observations, self.processes[slot].pid)

    def reset(self, seeds):
        if self.processes or not seeds:
            raise ValueError("legacy batch reset requires a fresh batch")
        return self.reset_slots(dict(enumerate(seeds)))

    def reset_slots(self, seeds):
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("nonempty nonnegative slot IDs are required")
        if any(type(s) is not int or not 0 <= s < 0xFFFFFFFF for s in seeds.values()):
            raise ValueError("native seed 0xFFFFFFFF is reserved; use a smaller uint32")
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
        return states

    def _launch_slots(self, seeds):
        processes = sokurl._launch_vs_group_from_title(
            len(seeds), self.launch_timeout, headless=True, unlimited=True,
            seeds=tuple(seeds.values()), pause_at_start=True,
            capture_images=self.observation_mode == "image",
            capture_state=self.observation_mode == "state",
        )
        self.processes.update(zip(seeds, processes, strict=True))
        states = {}
        try:
            for slot, process in zip(seeds, processes, strict=True):
                client = BridgeClient(process.pid)
                self.clients[slot] = client
                raw = wait_for_frame_zero(client, process.pid, self.launch_timeout)
                self.buffers[slot] = (ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))()
                self.frames[slot] = 0
                if self.observation_mode in {"image", "state"}:
                    from image_shared import ImageClient
                    self.image_clients[slot] = ImageClient(process.pid)
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
            states[slot] = self._observe(slot, snapshot.latest, snapshot.dropped_frames)
            _drain_fast(self.clients[slot], self.buffers[slot])
        self.active.difference_update(s for s, state in states.items() if state.ended)
        return states

    def close(self):
        self._close_slots(set(self.processes))

    def _close_slots(self, slots):
        errors = []
        for slot in slots:
            if slot in self.image_clients:
                self.image_clients.pop(slot).close()
            try:
                if slot in self.clients:
                    self.clients.pop(slot).close()
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
