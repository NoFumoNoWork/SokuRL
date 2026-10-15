"""Use the same paused-frame observations for live environments and replays."""
import os
from soku_rl.env.observation.visible_state import observe_visible_states
from soku_rl.env.observation.animation import observe_animation
from soku_rl.env.combat_events import CombatEventTracker
from soku_rl.env.input_lineage import InputLineage
from soku_rl.env.passive import PassiveQualifier
from soku_rl.pomg import Outcome, TimeStep
from game_runtime.offline_snapshot import OfflineSnapshotReader, offline_snapshot_requested


def time_step(raw, dropped_frames, observations, pid):
    if raw.sceneId != 5 or raw.battleMode != 3:
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


class ObservationReader:
    def __init__(self, pid, mode, visibility, animation_slots=16, horizon=7200, passive_policy=None):
        if mode not in {"privileged_state", "image", "state", "diagnostic_state", "animation"}:
            raise ValueError("unsupported observation mode")
        self.pid, self.mode, self.visibility = pid, mode, visibility
        self.animation_slots, self.horizon = animation_slots, horizon
        self.combat_events = CombatEventTracker() if mode == 'animation' else None
        self.native_combat = None
        self.previous_hp = None
        self.input_lineage = InputLineage(accepted_action_roots=True)
        self.passive_qualifier = PassiveQualifier(**(passive_policy or {}))
        self.resources = []
        if offline_snapshot_requested() and mode != "privileged_state":
            raise ValueError("offline snapshots require complete privileged observations")
        if mode == "privileged_state":
            from game_runtime.privileged import PrivilegedReader, ProcessMemory
            if offline_snapshot_requested():
                from play_runtime.channels import SnapshotHistory
                self.privileged = OfflineSnapshotReader(SnapshotHistory(pid))
            else:
                self.privileged = PrivilegedReader(ProcessMemory(pid))
            self.resources.append(self.privileged)
        if mode == "diagnostic_state":
            from game_runtime.diagnostic import DiagnosticReader
            from game_runtime.privileged import ProcessMemory
            self.diagnostic = DiagnosticReader(ProcessMemory(pid))
            self.resources.append(self.diagnostic)
        if mode in {"image", "state", "animation"}:
            from game_runtime.images import ImageClient
            self.image = ImageClient(pid)
            self.resources.append(self.image)
        if mode == 'animation' and os.environ.get('SOKURL_COMBAT_EVENTS') == '1':
            from game_runtime.combat_capture import CombatCaptureClient
            self.native_combat = CombatCaptureClient(pid)
            self.resources.append(self.native_combat)
            if os.environ.get('SOKURL_NATIVE_INPUT_COUNTERS')=='1' and not self.native_combat.native_input_counters:
                self.close()
                raise RuntimeError('installed Bridge does not advertise enabled native input counters')

    def read(self, raw, bridge, *, known_dropped=None):
        if self.mode == "privileged_state":
            observations = self.privileged.observe(raw, bridge)
        elif self.mode == "image":
            scene = self.image.read(int(raw.frameId), 10.0)
            observations = (scene.image, scene.image)
        elif self.mode in {"state", "animation"}:
            render = self.image.read_state(int(raw.frameId), 10.0)
            observations = (observe_animation(raw, render, self.visibility, self.animation_slots, self.horizon)
                            if self.mode == 'animation' else observe_visible_states(raw, render, self.visibility))
        else:
            observations = self.diagnostic.observe(raw, bridge)
        dropped = bridge.snapshot().dropped_frames if known_dropped is None else known_dropped
        result = time_step(raw, dropped, observations, self.pid)
        if self.mode == 'animation':
            # Events stay outside encoded policy observations and the old reward.
            result.diagnostics['combat_events'] = self.combat_events.observe(raw)
            if self.native_combat is not None:
                if int(raw.frameId)==0:
                    self.input_lineage.reset(int(raw.segmentId))
                events = self.input_lineage.observe(self.native_combat.drain(),
                    (int(raw.p1.characterId),int(raw.p2.characterId)))
                if any(event['frame']!=int(raw.frameId) or event['segment']!=int(raw.segmentId) for event in events):
                    raise RuntimeError('native contacts do not match current frame/epoch')
                hp = (int(raw.p1.hp),int(raw.p2.hp))
                if int(raw.frameId) and self.previous_hp is not None:
                    contacts = [event for event in events if event['kind'] in (
                        'HIT_RESOLUTION','BLOCK_HIGH_RESOLUTION','BLOCK_LOW_RESOLUTION')]
                    for seat in range(2):
                        measured = sum(max(0,event['hp_before']-event['hp_after']) for event in contacts
                                       if event['defender_seat']==seat)
                        if measured != max(0,self.previous_hp[seat]-hp[seat]):
                            raise RuntimeError('HP change outside covered native contact paths')
                self.previous_hp = hp
                result.diagnostics['native_combat_events'] = events
                result.diagnostics.update(self.passive_qualifier.observe(raw,events))
        return result

    def close(self):
        errors = []
        while self.resources:
            try:
                self.resources.pop().close()
            except Exception as error:
                errors.append(repr(error))
        if errors:
            raise RuntimeError(f"cannot close observation readers: {errors}")
