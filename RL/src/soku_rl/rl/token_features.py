"""Public entity-set attention followed by causal, fixed-window history encoding."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from soku_rl.env.observation.visible_state import OBJECT_SLOTS, STATE_FEATURES


class PublicTokenFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, history_frames=4, width=64, heads=4,
                 spatial_layers=2, temporal_layers=1, features_dim=128):
        if min(history_frames, width, heads, spatial_layers, temporal_layers, features_dim) < 1:
            raise ValueError('positive token architecture dimensions required')
        if width % heads:
            raise ValueError('width must be divisible by heads')
        base = history_frames * STATE_FEATURES
        if len(observation_space.shape) != 1 or observation_space.shape[0] < base:
            raise ValueError('flat public state history is required')
        super().__init__(observation_space, features_dim)
        self.history_frames = history_frames
        self.base_width = base
        self.fighter = nn.Sequential(nn.Linear(8, width), nn.GELU())
        # Presence, absolute position, and relative position: no inferred velocity.
        self.object = nn.Sequential(nn.Linear(5, width), nn.GELU())
        self.role = nn.Embedding(4, width)
        layer = lambda: nn.TransformerEncoderLayer(width, heads, width * 2,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(layer(), spatial_layers, enable_nested_tensor=False)
        self.temporal = nn.TransformerEncoder(layer(), temporal_layers, enable_nested_tensor=False)
        self.time = nn.Parameter(torch.empty(history_frames, width))
        nn.init.normal_(self.time, std=.02)
        self.output = nn.Sequential(nn.Linear(width + observation_space.shape[0] - base,
                                             features_dim), nn.Tanh())
        self.register_buffer('roles', torch.tensor([0, 1] + [2] * OBJECT_SLOTS +
                              [3] * OBJECT_SLOTS), persistent=False)
        self.register_buffer('causal', torch.ones(history_frames, history_frames,
                              dtype=torch.bool).triu(1), persistent=False)

    def encode_frames(self, frames):
        fighters = frames[:, :16].reshape(-1, 2, 8)
        objects = frames[:, 16:].reshape(-1, 2 * OBJECT_SLOTS, 3)
        relative = objects[..., 1:3] - fighters[:, :1, 1:3]
        tokens = torch.cat((self.fighter(fighters),
                            self.object(torch.cat((objects, relative), -1))), 1)
        tokens = tokens + self.role(self.roles)[None]
        # Fighters remain context even if their pose is invisible: gauges are public.
        pad = torch.cat((torch.zeros_like(fighters[..., 0], dtype=torch.bool),
                         objects[..., 0] <= 0), 1)
        return self.spatial(tokens, src_key_padding_mask=pad)[:, 0]

    def forward(self, observations):
        batch = observations.shape[0]
        frames = observations[:, :self.base_width].reshape(-1, STATE_FEATURES)
        encoded = self.encode_frames(frames).reshape(batch, self.history_frames, -1)
        # ObservationHistory supplies oldest->newest, reset frames are repeated.
        history = self.temporal(encoded + self.time[None], mask=self.causal)[:, -1]
        return self.output(torch.cat((history, observations[:, self.base_width:]), -1))
