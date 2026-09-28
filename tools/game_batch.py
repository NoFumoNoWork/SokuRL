"""Adapt owned th123 processes to the simultaneous two-player game contract."""
import ctypes

from soku_rl.observations import observe
from soku_rl.pomg import Outcome, TimeStep
from bridge_shared import BridgeClient, FRAME_RING_CAPACITY
from headless_validation import wait_for_frame_zero
from interaction_benchmark import wait_group
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
        self.processes = []
        self.clients = []
        self.buffers = []
        self.frame = 0
        self.active = set()

    def reset(self, seeds):
        if self.processes or not seeds or len(set(seeds)) != 1:
            raise ValueError("fresh batch and one common world seed are required")
        self.frame = 0
        self.processes = sokurl._launch_vs_group_from_title(
            len(seeds), self.launch_timeout, headless=True, unlimited=True,
            seed=seeds[0], pause_at_start=True,
        )
        states = {}
        for slot, process in enumerate(self.processes):
            client = BridgeClient(process.pid)
            self.clients.append(client)
            raw = wait_for_frame_zero(client, process.pid)
            self.buffers.append((ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))())
            states[slot] = _time_step(raw, 0)
        self.active = set(states)
        return states

    def step(self, actions):
        if not actions or set(actions) != self.active:
            raise ValueError("joint actions must cover every active game")
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
        self.frame += 1
        snapshots = wait_group([self.clients[s] for s in slots], sequences, self.frame, "ready")
        states = {}
        for slot, snapshot in zip(slots, snapshots, strict=True):
            states[slot] = _time_step(snapshot.latest, snapshot.dropped_frames)
            _drain_fast(self.clients[slot], self.buffers[slot])
        self.active = {s for s, state in states.items() if not state.ended}
        return states

    def close(self):
        errors = []
        for client in self.clients:
            try:
                client.close()
            except Exception as error:
                errors.append(repr(error))
        for process in self.processes:
            try:
                if process.is_running():
                    sokurl.shutdown(5.0, process.pid)
            except Exception as error:
                errors.append(repr(error))
        self.clients.clear()
        self.processes.clear()
        self.buffers.clear()
        self.active.clear()
        if errors:
            raise RuntimeError(f"failed to close owned game resources: {errors}")
