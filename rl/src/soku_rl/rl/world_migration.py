"""Explicit observation expansion; preserve old parameters, zero new columns."""
from pathlib import Path
import hashlib
import json

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3 import PPO
import torch
from torch import nn

from soku_rl.env.observation.animation import frame_width, HISTORY_OFFSETS
from soku_rl.rl.animation_features import AnimationTokenFeatures


def expand_observation(values, slots):
    """Insert new per-frame columns without moving/reinterpreting old channels."""
    width, frames = frame_width(slots), len(HISTORY_OFFSETS)
    old_base = width * frames
    world = np.zeros((*values.shape[:-1], frames, 2*(2+slots)), np.float32)
    core = values[..., :old_base].reshape(*values.shape[:-1], frames, width)
    return np.concatenate((np.concatenate((core,world),axis=-1).reshape(*values.shape[:-1],-1),
                           values[...,old_base:]), axis=-1)


def expand_world(model):
    extractor = model.policy.features_extractor
    if (not isinstance(extractor, AnimationTokenFeatures) or extractor.world_coordinates
            or model.policy.pi_features_extractor is not extractor
            or model.policy.vf_features_extractor is not extractor):
        raise ValueError('world migration requires the unchanged shared animation extractor')
    extra = len(HISTORY_OFFSETS)*2*(2+extractor.slots)
    shape = (model.observation_space.shape[0]+extra,)
    observation_space = spaces.Box(-np.inf,np.inf,shape,np.float32)
    layer = extractor.output[0]
    weight = torch.zeros((layer.out_features,layer.in_features+extra), device=layer.weight.device,
                          dtype=layer.weight.dtype)
    weight[:,:layer.in_features].copy_(layer.weight.detach())
    layer.weight, layer.in_features = nn.Parameter(weight), layer.in_features+extra
    extractor.world_coordinates, extractor.world_features = True, extra
    extractor.base += extra
    extractor.frame_size = frame_width(extractor.slots,True)
    extractor._observation_space = observation_space
    model.observation_space = model.policy.observation_space = observation_space
    model.policy_kwargs['features_extractor_kwargs']['world_coordinates'] = True
    model.policy.features_extractor_kwargs['world_coordinates'] = True
    # New experiment objective: do not misrepresent old Adam moments as resumed.
    model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
        lr=model.lr_schedule(1), **model.policy.optimizer_kwargs)
    return model


def migrate_world(checkpoint, config, destination):
    checkpoint, config, destination = Path(checkpoint), Path(config), Path(destination)
    if destination.exists():
        raise ValueError('preserve existing migration; use a new destination')
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = PPO.load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    slots = model.policy.features_extractor.slots
    old_sample = np.zeros((4,*model.observation_space.shape),np.float32)
    width = frame_width(slots)
    for row in range(4):
        for t in range(8):
            core = old_sample[row,t*width:(t+1)*width]
            core[0], core[10], core[3], core[13] = 1,1,1,-1
            core[1], core[11], core[4] = .2+row*.01,.8,1
    with torch.no_grad():
        before = model.policy.get_distribution(torch.as_tensor(old_sample)).distribution.probs.clone()
        value_before = model.policy.predict_values(torch.as_tensor(old_sample)).clone()
    old_state = {k:v.clone() for k,v in model.policy.state_dict().items()}
    sample = expand_observation(old_sample,slots)
    expand_world(model)
    for name,value in model.policy.state_dict().items():
        original = old_state[name]
        if original.shape != value.shape:
            assert name.endswith('output.0.weight')
            assert torch.equal(original, value[:,:original.shape[1]])
            assert torch.count_nonzero(value[:,original.shape[1]:]) == 0
        else:
            assert torch.equal(original,value)
    with torch.no_grad():
        after = model.policy.get_distribution(torch.as_tensor(sample)).distribution.probs
        value_after = model.policy.predict_values(torch.as_tensor(sample))
        torch.testing.assert_close(before,after,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(value_before,value_after,rtol=1e-6,atol=1e-7)
    destination.mkdir(parents=True)
    model.save(destination/'model.zip')
    loaded = PPO.load(destination/'model.zip',device='cpu')
    loaded.policy.set_training_mode(False)
    with torch.no_grad():
        torch.testing.assert_close(after,loaded.policy.get_distribution(torch.as_tensor(sample)).distribution.probs,rtol=0,atol=0)
    cfg = OmegaConf.load(config)
    cfg.episode.animation_world_coordinates = True
    cfg.rl.ppo.policy_kwargs.features_extractor_kwargs.world_coordinates = True
    if 'algorithm' in cfg:
        cfg.algorithm.ppo = cfg.rl.ppo
    OmegaConf.save(cfg,destination/'config.yaml')
    report = dict(status='PASS_EXPLICIT_WORLD_MIGRATION',source=str(checkpoint.resolve()),
        source_sha256=source_hash,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        max_probability_error=float((before-after).abs().max()),
        max_value_error=float((value_before-value_after).abs().max()),
        preservation='old columns and all other parameters bit-exact; new columns zero; CPU outputs within stated tolerance',
        optimizer='fresh; not optimizer-continuation evidence',native_steps=0,gradient_updates=0,
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()})
    (destination/'migration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
