"""Explicit, non-destructive expansion of animation timing embeddings and Adam."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from stable_baselines3 import PPO
from omegaconf import OmegaConf

from soku_rl.rl.animation_features import AnimationTokenFeatures


def expand_timing(model, size=8192, seed=1024):
    extractor = model.policy.features_extractor
    if not isinstance(extractor, AnimationTokenFeatures):
        raise ValueError('migration requires AnimationTokenFeatures')
    if (model.policy.pi_features_extractor is not extractor or
            model.policy.vf_features_extractor is not extractor):
        raise ValueError('migration requires shared actor/critic extractor')
    if any(e.num_embeddings > size for e in extractor.categories[3:]):
        raise ValueError('migration cannot shrink vocabulary')
    optimizer = model.policy.optimizer
    # Shared actor/critic extractors alias these same modules and parameters.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for embedding in extractor.categories[3:]:
            old = embedding.weight
            if old.shape[0] == size:
                continue
            replacement = nn.Embedding(size, old.shape[1], device='cpu', dtype=old.dtype)
            replacement.weight.data[:old.shape[0]].copy_(old.detach().cpu())
            parameter = nn.Parameter(replacement.weight.detach().to(old.device))
            state = optimizer.state.pop(old, {})
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.shape == old.shape:
                    expanded = torch.zeros_like(parameter)
                    expanded[:old.shape[0]].copy_(value)
                    state[key] = expanded
                elif torch.is_tensor(value) and value.ndim > 0:
                    raise ValueError(f'unsupported optimizer state {key}')
            for group in optimizer.param_groups:
                group['params'] = [parameter if p is old else p for p in group['params']]
            if state:
                optimizer.state[parameter] = state
            embedding.weight = parameter
            embedding.num_embeddings = size
    model.policy_kwargs.setdefault('features_extractor_kwargs', {})['timing_vocab_size'] = size
    model.policy.features_extractor_kwargs['timing_vocab_size'] = size
    return model


def migrate_checkpoint(checkpoint, config, destination, size=8192):
    checkpoint, config, destination = Path(checkpoint), Path(config), Path(destination)
    identity = dict(source=str(checkpoint.resolve()),
                    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                    timing_vocab_size=size)
    if destination.exists():
        report = json.loads((destination/'migration.json').read_text())
        if report['input'] != identity or report['status'] != 'PASS':
            raise ValueError('migration destination belongs to different input')
        for name, digest in report['outputs'].items():
            if hashlib.sha256((destination/name).read_bytes()).hexdigest() != digest:
                raise ValueError('migrated artifact hash mismatch')
        return destination/'model.zip', destination/'config.yaml'
    destination.mkdir(parents=True)
    model = PPO.load(checkpoint, device='cpu')
    sample = torch.zeros((1, *model.observation_space.shape))
    model.policy.set_training_mode(False)
    with torch.no_grad():
        before = model.policy.get_distribution(sample).distribution.probs.clone()
    expand_timing(model, size)
    with torch.no_grad():
        torch.testing.assert_close(before, model.policy.get_distribution(sample).distribution.probs, rtol=0, atol=0)
    model.save(destination/'model.zip')
    loaded = PPO.load(destination/'model.zip', device='cpu')
    loaded.policy.set_training_mode(False)
    with torch.no_grad():
        torch.testing.assert_close(before, loaded.policy.get_distribution(sample).distribution.probs, rtol=0, atol=0)
    cfg = OmegaConf.load(config)
    cfg.rl.ppo.policy_kwargs.features_extractor_kwargs.timing_vocab_size = size
    OmegaConf.save(cfg, destination/'config.yaml')
    outputs = {name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
               for name in ('model.zip','config.yaml')}
    (destination/'migration.json').write_text(json.dumps(dict(status='PASS', input=identity,
        outputs=outputs, steps=model.num_timesteps, seed=1024,
        meaning='capacity expansion; old rows/Adam preserved, new rows seeded, moments zero; not training evidence'), indent=2))
    return destination/'model.zip', destination/'config.yaml'
