"""Keep the existing spatial/temporal Transformer; add a zero-output context path."""
import torch
from torch import nn
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from soku_rl.env.observation.shared_native import PUBLIC_FEATURES, SHARED_FEATURES
from soku_rl.env.observation.memory_schema import PRIVILEGED_FEATURES
from .animation_features import AnimationTokenFeatures
from .address_invariant_features import AddressInvariantCombatFeatures


class SharedNativeFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,public_kwargs=None,features_dim=128,
                 object_features=8,player_features=64):
        if observation_space.shape!=(SHARED_FEATURES,):
            raise ValueError('explicit shared_current_native_v1 observation required')
        public_kwargs=dict(public_kwargs or {})
        if (public_kwargs.get('object_slots',16)!=16 or
                public_kwargs.get('world_coordinates',True) is not True or
                public_kwargs.get('features_dim',features_dim)!=features_dim):
            raise ValueError('public backbone must keep the1832 world contract')
        public_kwargs.update(object_slots=16,world_coordinates=True,features_dim=features_dim)
        super().__init__(observation_space,features_dim)
        box=lambda width:spaces.Box(-np.inf,np.inf,(width,),np.float32)
        self.public=AnimationTokenFeatures(box(PUBLIC_FEATURES),**public_kwargs)
        # Reuse the existing present-object encoder and preserve native list
        # order/geometry. Do NOT make a600k-column dense Transformer projection.
        self.native=AddressInvariantCombatFeatures(box(PRIVILEGED_FEATURES),
            history_frames=1,object_features=object_features,
            player_features=player_features,features_dim=features_dim)
        nn.init.zeros_(self.native.output[0].weight)
        nn.init.zeros_(self.native.output[0].bias)

    def forward(self,observations):
        return self.public(observations[:,:PUBLIC_FEATURES])+self.native(observations[:,PUBLIC_FEATURES:])


def initialize_shared_policy(target,source):
    """Transfer existing Actor/Critic/backbone exactly; optimizer must be fresh."""
    if (not isinstance(source.features_extractor,AnimationTokenFeatures) or
            not isinstance(target.features_extractor,SharedNativeFeatures) or
            source.observation_space.shape!=(PUBLIC_FEATURES,) or
            type(source) is not type(target) or
            getattr(source,'masked_actions',())!=getattr(target,'masked_actions',()) or
            source.action_space.n!=90 or target.action_space.n!=90 or
            source.action_space!=target.action_space):
        raise ValueError('shared expansion requires an existing1832/90 animation policy')
    old=source.state_dict()
    new=target.state_dict()
    mapped={}
    for name,value in old.items():
        key=name
        for prefix in ('features_extractor.','pi_features_extractor.','vf_features_extractor.'):
            if name.startswith(prefix):
                key=prefix+'public.'+name[len(prefix):]
                break
        if key not in new or new[key].shape!=value.shape:
            raise ValueError('incompatible policy parameter: '+name)
        mapped[key]=value.detach().clone()
    extras=set(new)-set(mapped)
    if any('.native.' not in name for name in extras):
        raise ValueError('shared expansion changed a non-context policy parameter')
    # A reused target cannot smuggle a previously trained context into an exact
    # initialization claim. Enforce the zero residual again before transfer.
    for name in extras:
        if '.native.output.0.' in name: new[name]=torch.zeros_like(new[name])
    target.load_state_dict(new|mapped,strict=True)
    target.optimizer=target.optimizer_class(target.parameters(),
        lr=source.optimizer.param_groups[0]['lr'],**target.optimizer_kwargs)
    return dict(schema='shared_current_native_v1',copied_parameters=len(mapped),
        added_parameters=sorted(extras),optimizer='fresh',
        scope='offline weight transfer; NOT a live environment or training authorization')
