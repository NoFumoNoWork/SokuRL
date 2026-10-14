"""Exclude process-local entity pointers from neural combat representations."""
from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, FIGHTER_WIDTH, OBJECT_WIDTH
from soku_rl.rl.combat_features import NumericCombatPrivilegedFeatures


class AddressInvariantCombatFeatures(NumericCombatPrivilegedFeatures):
    """Mask address parts and their numeric channel; retain all game fields.

    This ablation deliberately omits pointer identity as well as its magnitude.
    Object list order, every present object, and the shared raw observation are
    preserved. Original rule policies still receive their complete addresses.
    """

    def numeric_features(self, records):
        features = super().numeric_features(records)
        if records.shape[-1] in (2 * FIGHTER_WIDTH, 2 * OBJECT_WIDTH):
            address = FIGHTER_NAMES.index("address")
            features[..., 2 * address:2 * address + 2] = 0.
            features[..., records.shape[-1] + address] = 0.
        return features
