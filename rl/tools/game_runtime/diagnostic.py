"""Select learner-visible projectile features from Platform objects."""
from sokurl_platform.projectile_reader import ProjectileReader
from soku_rl.env.encoding import MAX_PROJECTILES
from soku_rl.env.observation.diagnostic import Projectile, observe, observe_projectiles

class DiagnosticReader(ProjectileReader):
    def __init__(self, memory):
        super().__init__(memory, MAX_PROJECTILES, Projectile)

    def observe(self, raw, bridge):
        if not (raw.p1ObjectOverflow or raw.p2ObjectOverflow):
            return tuple(observe(raw, seat) for seat in (0, 1))
        before = bridge.snapshot()
        if (before.run_state_name != "PAUSED" or before.game_frame != raw.frameId
                or before.latest.segmentId != raw.segmentId):
            raise RuntimeError("diagnostic reads require the exact paused simulation frame")
        self.memory.begin_frame()
        projectiles = tuple(self.projectiles(seat) for seat in (0, 1))
        after = bridge.snapshot()
        if (after.run_state_name != "PAUSED" or after.game_frame != raw.frameId
                or after.latest.segmentId != raw.segmentId):
            raise RuntimeError("game advanced during diagnostic observation")
        return tuple(observe_projectiles(raw, seat, projectiles[1 - seat]) for seat in (0, 1))

    def observe_snapshot(self, raw):
        """Decode immutable realtime capture memory, never the advancing process."""
        if not (raw.p1ObjectOverflow or raw.p2ObjectOverflow):
            return tuple(observe(raw, seat) for seat in (0, 1))
        projectiles = tuple(self.projectiles(seat) for seat in (0, 1))
        return tuple(observe_projectiles(raw, seat, projectiles[1 - seat]) for seat in (0, 1))

    def close(self):
        self.memory.close()
