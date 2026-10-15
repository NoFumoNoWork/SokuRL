"""Add scaled combat context without dropping any privileged observation fields."""
import torch

from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, PLAYER_WIDTH, WORLD_NAMES
from soku_rl.rl.features import PrivilegedFeatures, NumericPrivilegedFeatures


# Engineering scales, not bounds or claims about every character's valid range.
# Values are not clipped; the complete original two-part encoding remains present.
FIGHTER_SCALES = {
    "hp": 10000., "rei": 5000., "rmax": 5000., "x": 1280., "y": 1280.,
    "xspeed": 20., "yspeed": 20., "dir": 1., "air": 1., "hitstop": 60.,
    "act": 1000., "act_block": 10., "frame": 60., "combo": 20.,
    "dam": 10000., "limit": 100., "timestop": 60., "rei_stop": 60.,
}


class CombatPrivilegedFeatures(PrivilegedFeatures):
    context_width = 2 * len(FIGHTER_SCALES) + 6

    def __init__(self, observation_space, history_frames, object_features, player_features, features_dim):
        super().__init__(observation_space, history_frames, object_features, player_features, features_dim)
        indices = [[(len(WORLD_NAMES) + seat * PLAYER_WIDTH + FIGHTER_NAMES.index(name)) * 2
                    for name in FIGHTER_SCALES] for seat in (0, 1)]
        self.register_buffer("combat_indices", torch.tensor(indices), persistent=False)
        self.register_buffer("combat_scales", torch.tensor(list(FIGHTER_SCALES.values())), persistent=False)

    def frame_context(self, frames):
        # Encoded players are already in own/opponent order for the observer.
        values = frames[:, self.combat_indices] * 4294967296. + frames[:, self.combat_indices + 1] * 65536.
        scaled = values / self.combat_scales
        fields = {name: index for index, name in enumerate(FIGHTER_SCALES)}
        own, opponent = scaled[:, 0], scaled[:, 1]
        difference = opponent - own
        facing = own[:, fields["dir"]]
        relative = torch.stack((difference[:, fields["x"]] * facing, difference[:, fields["y"]],
            difference[:, fields["xspeed"]] * facing, difference[:, fields["yspeed"]],
            -difference[:, fields["hp"]], -difference[:, fields["rei"]]), dim=1)
        return torch.cat((scaled.flatten(1), relative), dim=1)


class NumericCombatPrivilegedFeatures(CombatPrivilegedFeatures):
    """Combine full-record numeric channels with the existing combat context."""
    numeric_width = 3

    def numeric_features(self, records):
        return NumericPrivilegedFeatures.numeric_features(self, records)
