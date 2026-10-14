"""Append categorical action-ID embeddings to the complete numeric combat records."""
import torch
from torch import nn

from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH
from soku_rl.rl.address_invariant_features import AddressInvariantCombatFeatures


class ActionIdCombatFeatures(AddressInvariantCombatFeatures):
    def __init__(self, observation_space, history_frames, object_features, player_features,
                 features_dim, action_embedding_dim):
        if type(action_embedding_dim) is not int or action_embedding_dim < 1:
            raise ValueError("action_embedding_dim must be a positive integer")
        super().__init__(observation_space, history_frames, object_features, player_features, features_dim)
        # The memory schema declares act as uint16. Cover every possible ID,
        # without clipping, hashing, or imposing numeric proximity on categories.
        self.action_embedding = nn.Embedding(65536, action_embedding_dim)
        self.object_encoder = nn.Sequential(
            nn.Linear(OBJECT_WIDTH * self.numeric_width + action_embedding_dim, object_features), nn.Tanh())
        self.player_encoder = nn.Sequential(nn.Linear(FIGHTER_WIDTH * self.numeric_width
            + action_embedding_dim + MAX_OBJECTS * object_features, player_features), nn.Tanh())

    def numeric_features(self, records):
        numeric = super().numeric_features(records)
        if records.shape[-1] in (2 * FIGHTER_WIDTH, 2 * OBJECT_WIDTH):
            index = FIGHTER_NAMES.index("act") * 2
            actions = records[..., index] * 4294967296. + records[..., index + 1] * 65536.
            return torch.cat((numeric, self.action_embedding(actions.to(torch.long))), dim=-1)
        return numeric

    def encode_objects(self, objects, present):
        # Padding is not a game record and may contain arbitrary values. Look
        # up IDs only for present objects, preserving every valid slot and order.
        return self.encode_present_objects(objects, present)
