"""Explicit weight-only conversion from shared to independent PPO features."""
import copy

import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from soku_rl.rl import ppo_settings
from soku_rl.rl.learner import learner_kind


def split_feature_config(training):
    converted = copy.deepcopy(training)
    shared = ppo_settings(converted)
    if learner_kind(shared) != 'ppo' or any(key in shared for key in ('rehearsal', 'online_anchor')):
        raise ValueError('feature splitting requires standard PPO without auxiliary objectives')
    architecture = shared['ppo']['policy_kwargs']
    if (architecture.get('share_features_extractor', True) is not True
            or architecture.get('shared_lstm', False) is not False):
        raise ValueError('feature splitting requires shared features and independent actor/critic recurrence')
    learner = converted['algorithm']['response'] if converted['algorithm']['name'] == 'psro' else converted['algorithm']
    for config in (shared, learner):
        config['ppo']['policy_kwargs']['share_features_extractor'] = False
    ppo_settings(converted)
    return converted


def copy_split_feature_weights(source, destination):
    """Copy every tensor; reset optimization explicitly by requiring a fresh target."""
    if type(source) not in (PPO, RecurrentPPO) or type(destination) is not type(source):
        raise ValueError('feature splitting requires matching standard PPO implementations')
    if (destination.num_timesteps or destination._n_updates or destination.policy.optimizer.state):
        raise ValueError('feature splitting requires a fresh destination optimizer and counters')
    if (source.observation_space != destination.observation_space
            or source.action_space != destination.action_space
            or type(source.policy) is not type(destination.policy)):
        raise ValueError('feature splitting must preserve policy and observation/action contracts')
    old, new = source.policy, destination.policy
    if (not old.share_features_extractor or new.share_features_extractor
            or old.pi_features_extractor is not old.vf_features_extractor
            or new.pi_features_extractor is new.vf_features_extractor
            or old.features_extractor is not old.pi_features_extractor
            or new.features_extractor is not new.pi_features_extractor):
        raise ValueError('feature splitting requires shared source and independent destination features')
    architectures = [dict(model.policy_kwargs) for model in (source, destination)]
    for architecture in architectures:
        architecture.pop('share_features_extractor', True)
    if architectures[0] != architectures[1]:
        raise ValueError('feature splitting cannot change another network setting')
    actor = list(new.pi_features_extractor.parameters()) + list(new.pi_features_extractor.buffers())
    critic = list(new.vf_features_extractor.parameters()) + list(new.vf_features_extractor.buffers())
    if not list(old.pi_features_extractor.parameters()):
        raise ValueError('feature splitting requires trainable features')
    actor_storage = {(tensor.device, tensor.untyped_storage().data_ptr()) for tensor in actor if tensor.numel()}
    critic_storage = {(tensor.device, tensor.untyped_storage().data_ptr()) for tensor in critic if tensor.numel()}
    if actor_storage & critic_storage:
        raise ValueError('destination actor and critic feature storage must be independent')
    original, target = old.state_dict(), new.state_dict()
    if original.keys() != target.keys() or any(
            original[key].shape != target[key].shape or original[key].dtype != target[key].dtype
            for key in original):
        raise ValueError('feature splitting must retain every policy tensor and buffer')
    original_buffers = dict(old.named_buffers(remove_duplicate=False))
    target_buffers = dict(new.named_buffers(remove_duplicate=False))
    if original_buffers.keys() != target_buffers.keys() or any(
            value.shape != target_buffers[key].shape or value.dtype != target_buffers[key].dtype
            for key, value in original_buffers.items()):
        raise ValueError('feature splitting must preserve all buffer layouts')
    # Nonpersistent buffers are reconstructed by the policy constructor on load.
    # Copying changed runtime values would falsely promise a faithful export.
    if any(not torch.equal(value.cpu(), target_buffers[key].cpu())
            for key, value in original_buffers.items() if key not in original):
        raise ValueError('feature splitting cannot export changed nonpersistent buffers')
    new.load_state_dict(original, strict=True)
    if any(not torch.equal(tensor.cpu(), original[key].cpu()) for key, tensor in new.state_dict().items()):
        raise RuntimeError('feature splitting failed exact tensor preservation')
    if any(not torch.equal(value.cpu(), original_buffers[key].cpu())
            for key, value in new.named_buffers(remove_duplicate=False)):
        raise RuntimeError('feature splitting failed exact buffer preservation')
    return {'method': 'split_ppo_features_weights_only', 'source_steps': source.num_timesteps,
        'source_ppo_n_updates': source._n_updates, 'destination_steps': 0, 'destination_ppo_n_updates': 0,
        'source_parameters': sum(p.numel() for p in old.parameters()),
        'destination_parameters': sum(p.numel() for p in new.parameters()),
        'copied_state_tensors': len(original), 'optimizer_reset': True,
        'actor_critic_feature_storage_independent': True, 'all_policy_tensors_and_buffers_equal': True}
