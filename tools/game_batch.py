"""Adapt owned th123 processes to the simultaneous two-player game contract."""
import ctypes
from dataclasses import replace

from soku_rl.observations import observe
from soku_rl.pomg import Outcome, TimeStep
from bridge_shared import BridgeClient, FRAME_RING_CAPACITY, wait_for_steps
from headless_validation import wait_for_frame_zero
from unlimited_benchmark import FRAME_SIZE, _drain_fast
import sokurl


def _time_step(raw, dropped_frames):
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
    return TimeStep(raw.frameId, (observe(raw, 0), observe(raw, 1)), rewards, outcome, {
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

    def configure_observation(self, mode):
        if self.processes or mode not in {"image", "diagnostic_state"}:
            raise ValueError("set a supported observation mode before launching games")
        self.observation_mode = mode

    def _observe(self, slot, raw, dropped):
        step = _time_step(raw, dropped)
        if self.observation_mode == "image":
            image = self.image_clients[slot].read(int(raw.frameId), 10.0)
            return replace(step, observations=(image, image))
        return step

    def reset(self, seeds):
        if self.processes or not seeds:
            raise ValueError("legacy batch reset requires a fresh batch")
        return self.reset_slots(dict(enumerate(seeds)))

    def reset_slots(self, seeds):
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("nonempty nonnegative slot IDs are required")
        if any(type(s) is not int or not 0 <= s < 0xFFFFFFFF for s in seeds.values()):
            raise ValueError("native seed 0xFFFFFFFF is reserved; use a smaller uint32")
        self._close_slots(set(seeds) & set(self.processes))
        processes = sokurl._launch_vs_group_from_title(
            len(seeds), self.launch_timeout, headless=True, unlimited=True,
            seeds=tuple(seeds.values()), pause_at_start=True,
            capture_images=self.observation_mode == "image",
        )
        self.processes.update(zip(seeds, processes, strict=True))
        states = {}
        try:
            for slot, process in zip(seeds, processes, strict=True):
                client = BridgeClient(process.pid)
                self.clients[slot] = client
                raw = wait_for_frame_zero(client, process.pid)
                self.buffers[slot] = (ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))()
                self.frames[slot] = 0
                if self.observation_mode == "image":
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
