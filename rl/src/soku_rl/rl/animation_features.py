"""Categorical animation proxies with sparse past-only temporal attention."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from soku_rl.env.observation.animation import frame_width, HISTORY_OFFSETS


class AnimationTokenFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, object_slots=16, width=64, heads=4, features_dim=128,
                 timing_vocab_size=1024):
        base = len(HISTORY_OFFSETS) * frame_width(object_slots)
        if observation_space.shape[0] < base or width % heads:
            raise ValueError('invalid animation-token observation contract')
        super().__init__(observation_space, features_dim)
        if type(timing_vocab_size) is not int or timing_vocab_size < 1024:
            raise ValueError('timing vocabulary must have at least 1024 rows')
        self.slots, self.base, self.frame_size = object_slots, base, frame_width(object_slots)
        self.position = nn.Linear(4, width)
        # Exact bounded IDs. Out-of-vocabulary states trap, never hash or alias.
        self.categories = nn.ModuleList(nn.Embedding(size, width) for size in
                                       (20, 2048, 256, timing_vocab_size, timing_vocab_size, timing_vocab_size))
        self.roles = nn.Embedding(4, width)
        layer = lambda: nn.TransformerEncoderLayer(width, heads, width*2, dropout=0.,
                                                   activation='gelu', batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(layer(), 1, enable_nested_tensor=False)
        self.temporal = nn.TransformerEncoder(layer(), 1, enable_nested_tensor=False)
        self.time = nn.Parameter(torch.zeros(8, width))
        self.output = nn.Sequential(nn.Linear(width+5+observation_space.shape[0]-base, features_dim), nn.Tanh())
        self.register_buffer('causal', torch.ones(8, 8, dtype=torch.bool).triu(1), persistent=False)

    def forward(self, observations):
        frames = observations[:, :self.base].reshape(-1, self.frame_size)
        tokens = frames[:, :-5].reshape(-1, 2+self.slots, 10)
        encoded = self.position(tokens[..., :4])
        for index, embedding in enumerate(self.categories):
            values = tokens[..., 4+index]
            # Object last column is ownership, not fighter elapsed counter.
            ids = values.long()
            if torch.any(values != ids) or torch.any(ids < 0) or torch.any(ids >= embedding.num_embeddings):
                raise ValueError(f'unsupported animation category {index}')
            encoded = encoded + embedding(ids)
        roles = torch.zeros_like(tokens[..., 0], dtype=torch.long)
        roles[:, 1] = 1
        roles[:, 2:] = 2 + tokens[:, 2:, 9].long()
        encoded = encoded + self.roles(roles)
        mask = tokens[..., 0] <= 0
        # Keep self summary query valid even when the fighter is invisible.
        mask[:, 0] = False
        summaries = self.spatial(encoded, src_key_padding_mask=mask)[:, 0]
        history = summaries.reshape(observations.shape[0], 8, -1)
        latest = self.temporal(history+self.time, mask=self.causal)[:, -1]
        globals_ = frames.reshape(observations.shape[0], 8, -1)[:, -1, -5:]
        return self.output(torch.cat((latest, globals_, observations[:, self.base:]), -1))
