"""Condition ordered-object attention on both fighters and observer-relative geometry."""
import math

import torch
from torch import nn

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH, WORLD_NAMES)
from soku_rl.rl.address_invariant_features import AddressInvariantCombatFeatures


class RelationalCombatFeatures(AddressInvariantCombatFeatures):
    def __init__(self, observation_space, history_frames, object_features, player_features,
                 features_dim, attention_heads, attention_queries):
        if (type(attention_heads) is not int or attention_heads < 1
                or type(attention_queries) is not int or attention_queries < 1
                or object_features % attention_heads):
            raise ValueError("object width must be divisible by positive integer attention_heads; queries must be positive")
        super().__init__(observation_space, history_frames, object_features, player_features, features_dim)
        self.attention_queries = attention_queries
        self.object_features = object_features
        self.fighter_encoder = nn.Sequential(nn.Linear(FIGHTER_WIDTH * self.numeric_width, player_features), nn.Tanh())
        self.query_encoder = nn.Sequential(nn.Linear(2 * player_features, attention_queries * object_features), nn.Tanh())
        self.object_context = nn.Sequential(nn.Linear(object_features + 7, object_features), nn.Tanh())
        self.attention = nn.MultiheadAttention(object_features, attention_heads, dropout=0., batch_first=True)
        # Replace the inherited flat object-slot projection; no second PPO or
        # alternative observation contract is introduced.
        self.player_encoder = nn.Sequential(
            nn.Linear(player_features + attention_queries * object_features, player_features), nn.Tanh())
        position = torch.arange(MAX_OBJECTS, dtype=torch.float32) / (MAX_OBJECTS - 1)
        self.register_buffer("object_positions", torch.stack(
            (position, torch.sin(2 * math.pi * position), torch.cos(2 * math.pi * position)), dim=-1), persistent=False)
        self.register_buffer("geometry_indices", torch.tensor(
            [FIGHTER_NAMES.index(name) * 2 for name in ("x", "y", "xspeed", "yspeed")]), persistent=False)
        self.register_buffer("geometry_scales", torch.tensor([1280., 1280., 20., 20.]), persistent=False)

    def relative_object_geometry(self, objects, fighters):
        """Use observer (first fighter) coordinates for both owners' object lists."""
        own = fighters.reshape(-1, 2, FIGHTER_WIDTH * 2)[:, 0]
        origin = own[:, self.geometry_indices] * 4294967296. + own[:, self.geometry_indices + 1] * 65536.
        position = objects[..., self.geometry_indices] * 4294967296. + objects[..., self.geometry_indices + 1] * 65536.
        relative = (position - origin.repeat_interleave(2, dim=0).unsqueeze(1)) / self.geometry_scales
        direction = FIGHTER_NAMES.index("dir") * 2
        facing = own[:, direction] * 4294967296. + own[:, direction + 1] * 65536.
        signs = torch.stack((facing, torch.ones_like(facing), facing, torch.ones_like(facing)), dim=-1)
        return relative * signs.repeat_interleave(2, dim=0).unsqueeze(1)

    def attend_objects(self, objects, present, fighters, fighter_features):
        # All lists occupy a valid prefix. Remove only the batch-wide suffix
        # absent from every list, never a present object or an interior slot.
        width = int(present.any(dim=0).sum().item())
        objects, present = objects[:, :width], present[:, :width]
        encoded = self.encode_objects(objects, present)
        geometry = self.relative_object_geometry(objects, fighters)
        positions = self.object_positions[:width].unsqueeze(0).expand(objects.shape[0], -1, -1)
        entities = self.object_context(torch.cat((encoded, geometry, positions), dim=-1))
        pairs = fighter_features.reshape(-1, 2, fighter_features.shape[-1])
        queries = self.query_encoder(torch.cat((pairs, pairs.flip(1)), dim=-1).flatten(0, 1))
        queries = queries.reshape(-1, self.attention_queries, self.object_features)
        # An always-unmasked zero token prevents all-masked softmax NaNs. Empty
        # lists are explicitly zeroed afterwards, including projection biases.
        entities = torch.cat((entities.new_zeros((entities.shape[0], 1, self.object_features)), entities), dim=1)
        padding = torch.cat((present.new_zeros((present.shape[0], 1)), ~present), dim=1)
        pooled, _ = self.attention(queries, entities, entities, key_padding_mask=padding, need_weights=False)
        return pooled.flatten(1) * present.any(dim=1, keepdim=True)

    def forward(self, observations):
        batch = observations.shape[0]
        frames = observations[:, :self.base_width].reshape(batch * self.history_frames, -1)
        world = frames[:, :len(WORLD_NAMES) * 2]
        players = frames[:, len(WORLD_NAMES) * 2:].reshape(-1, PLAYER_WIDTH * 2)
        fighters = players[:, :FIGHTER_WIDTH * 2]
        objects = players[:, FIGHTER_WIDTH * 2:].reshape(-1, MAX_OBJECTS, OBJECT_WIDTH * 2)
        count_index = FIGHTER_NAMES.index("obj_n") * 2
        counts = fighters[:, count_index] * 4294967296. + fighters[:, count_index + 1] * 65536.
        present = self.object_indices.unsqueeze(0) < counts.unsqueeze(1)
        fighter_features = self.fighter_encoder(self.numeric_features(fighters))
        pooled = self.attend_objects(objects, present, fighters, fighter_features)
        players = self.player_encoder(torch.cat((fighter_features, pooled), dim=1))
        context = torch.cat((self.numeric_features(world), players.reshape(batch * self.history_frames, -1),
                             self.frame_context(frames)), dim=1)
        return self.output(torch.cat((context.reshape(batch, -1), observations[:, self.base_width:]), dim=1))
