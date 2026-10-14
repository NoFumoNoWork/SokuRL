"""Express spatial fields and input history in the observer's current facing."""
import math

import torch

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_BOXES, MAX_OBJECTS, OBJECT_NAMES,
    OBJECT_WIDTH, PLAYER_WIDTH, PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.rl.address_invariant_features import AddressInvariantCombatFeatures


def geometry_parts(values, dtype):
    """Two float32 parts retain subtraction residuals before the learned projection.

    These parts use the existing decoder weights, but need not be the raw
    reader's truncated-base representation. Normalize both facing branches so
    the representation does not itself reveal the original facing.
    """
    high = values.to(dtype) / 4294967296.
    low = ((values - high.double() * 4294967296.) / 65536.).to(dtype)
    return torch.stack((high, low), dim=-1)


class CanonicalCombatFeatures(AddressInvariantCombatFeatures):
    def __init__(self, observation_space, history_frames, object_features, player_features,
                 features_dim, arena_width):
        if type(arena_width) not in (int, float) or not math.isfinite(arena_width) or arena_width <= 0:
            raise ValueError("arena_width must be finite and positive")
        super().__init__(observation_space, history_frames, object_features, player_features, features_dim)
        if (observation_space.shape[0] - self.base_width) % 8:
            raise ValueError("canonical features require only complete command-history suffixes")
        self.arena_width = float(arena_width)
        self.register_buffer("geometry_fields", torch.tensor(
            [OBJECT_NAMES.index(name) for name in ("x", "xspeed", "dir")]), persistent=False)
        self.register_buffer("box_indices", torch.arange(MAX_BOXES), persistent=False)

    def canonical_entities(self, records, names, mirrored, present):
        pairs = records.reshape(*records.shape[:-1], records.shape[-1] // 2, 2)
        result = pairs.clone()
        parts = pairs[..., self.geometry_fields, :].double()
        values = parts[..., 0] * 4294967296. + parts[..., 1] * 65536.
        reflected = torch.stack((self.arena_width - values[..., 0],
                                 -values[..., 1], -values[..., 2]), dim=-1)
        values = torch.where(mirrored.unsqueeze(-1), reflected, values)
        result[..., self.geometry_fields, :] = geometry_parts(values, records.dtype)
        for section, count_name in enumerate(("hitarea_n", "attackarea_n")):
            offset = len(names) + section * MAX_BOXES * 4
            boxes = pairs[..., offset:offset + MAX_BOXES * 4, :].reshape(
                *records.shape[:-1], MAX_BOXES, 4, 2)
            changed = boxes.clone()
            horizontal = boxes[..., [0, 2], :].double()
            horizontal = horizontal[..., 0] * 4294967296. + horizontal[..., 1] * 65536.
            horizontal = torch.where(mirrored[..., None, None],
                                     self.arena_width - horizontal.flip(-1), horizontal)
            changed[..., [0, 2], :] = geometry_parts(horizontal, records.dtype)
            count = pairs[..., names.index(count_name), :]
            count = count[..., 0] * 4294967296. + count[..., 1] * 65536.
            active = self.box_indices < count.unsqueeze(-1)
            changed = torch.where(active[..., None, None], changed, boxes)
            result[..., offset:offset + MAX_BOXES * 4, :] = changed.flatten(-3, -2)
        # Never transform absent object slots or zero-filled history/padding.
        return torch.where(present[..., None, None], result, pairs).flatten(-2)

    def canonical_observations(self, observations):
        batch = observations.shape[0]
        frames = observations[:, :self.base_width].reshape(batch, self.history_frames, -1)
        direction = 2 * (len(WORLD_NAMES) + FIGHTER_NAMES.index("dir"))
        own = frames[..., direction] * 4294967296. + frames[..., direction + 1] * 65536.
        # All history frames and previous commands share the latest reference
        # direction; old frames can have a different physical facing.
        mirror = (own[:, -1] < 0)[:, None].expand(-1, self.history_frames).reshape(-1)
        active_frames = (own != 0).reshape(-1)
        flat = frames.reshape(batch * self.history_frames, -1)
        world = flat[:, :len(WORLD_NAMES) * 2]
        players = flat[:, len(WORLD_NAMES) * 2:].reshape(-1, PLAYER_WIDTH * 2)
        fighters = players[:, :FIGHTER_WIDTH * 2]
        objects = players[:, FIGHTER_WIDTH * 2:].reshape(-1, MAX_OBJECTS, OBJECT_WIDTH * 2)
        mirror = mirror.repeat_interleave(2)
        present = active_frames.repeat_interleave(2)
        transformed = self.canonical_entities(fighters, FIGHTER_NAMES, mirror, present)
        key_start = 2 * (len(FIGHTER_NAMES) + MAX_BOXES * 8 + 10 + 16 + 28)
        horizontal_keys = fighters[:, key_start + 4:key_start + 8].reshape(-1, 2, 2)
        transformed[:, key_start + 4:key_start + 8] = torch.where(
            (mirror & present)[:, None, None], horizontal_keys.flip(1), horizontal_keys).flatten(1)
        count_index = FIGHTER_NAMES.index("obj_n") * 2
        counts = fighters[:, count_index] * 4294967296. + fighters[:, count_index + 1] * 65536.
        object_present = (self.object_indices < counts[:, None]) & present[:, None]
        objects = self.canonical_entities(objects, OBJECT_NAMES, mirror[:, None], object_present)
        players = torch.cat((transformed, objects.flatten(1)), dim=1)
        frames = torch.cat((world, players.reshape(batch * self.history_frames, -1)), dim=1)
        commands = observations[:, self.base_width:].reshape(batch, -1, 8).clone()
        commands[..., 0] *= torch.where(own[:, -1] < 0, -1., 1.)[:, None]
        return torch.cat((frames.reshape(batch, -1), commands.flatten(1)), dim=1)

    def forward(self, observations):
        return super().forward(self.canonical_observations(observations))
