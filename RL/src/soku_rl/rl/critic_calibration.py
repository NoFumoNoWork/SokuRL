"""Fit a recurrent PPO's private critic to returns from its frozen behavior policy."""
import math

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.policies import BaseModel

from soku_rl.rl.recurrent_cloning import episode_chunks, zero_states
from soku_rl.rl.sparse_transfer import restore_batch


def private_critic_parameters(model):
    if not isinstance(model, RecurrentPPO) or model.policy.lstm_critic is None:
        raise ValueError("critic calibration requires RecurrentPPO with an independent critic LSTM")
    policy = model.policy
    modules = (policy.lstm_critic, policy.mlp_extractor.value_net, policy.value_net)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("private critic parameters must not alias")
    actor_ids = {id(parameter) for module in (policy.pi_features_extractor, policy.lstm_actor,
        policy.mlp_extractor.policy_net, policy.action_net) for parameter in module.parameters()}
    if actor_ids & {id(parameter) for parameter in parameters}:
        raise ValueError("critic calibration cannot update actor parameters")
    return parameters


def critic_epoch(model, episodes, order, batch_size, sequence_length, optimizer, training):
    if (type(batch_size) is not int or type(sequence_length) is not int or sequence_length < 1
            or batch_size < sequence_length or batch_size % sequence_length or type(training) is not bool):
        raise ValueError("critic calibration requires explicit training mode and whole sequence batches")
    parameters = private_critic_parameters(model)
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in parameters}:
        raise ValueError("critic optimizer must contain exactly the private critic parameters")
    # Frozen feature buffers and dropout behavior must also remain fixed.
    policy = model.policy
    policy.set_training_mode(False)
    # cuDNN LSTM backward requires training mode on this branch, while frozen
    # feature buffers/dropout and the actor must stay in evaluation mode.
    for module in (policy.lstm_critic, policy.mlp_extractor.value_net, policy.value_net):
        module.train(training)
    squared_error, frames, updates = 0., 0, 0
    with torch.set_grad_enabled(training):
        for chunk in episode_chunks(episodes, order, sequence_length, batch_size // sequence_length):
            if chunk['new_group']:
                states = zero_states(policy, len(chunk['keep'])).vf
            else:
                states = tuple(value[:, chunk['keep'], :] for value in states)
            rows = chunk['samples']
            observations = restore_batch([row[0] for row in rows], model.device)
            with torch.no_grad():
                features = BaseModel.extract_features(policy, observations, policy.vf_features_extractor)
            latent, states = policy._process_sequence(features, states,
                torch.zeros(len(rows), device=model.device), policy.lstm_critic)
            values = policy.value_net(policy.mlp_extractor.forward_critic(latent)).flatten()
            targets = torch.as_tensor(np.asarray([row[2] for row in rows], dtype=np.float32), device=model.device)
            valid = torch.as_tensor(chunk['valid'], device=model.device)
            loss = ((values - targets) ** 2)[valid].mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite critic calibration loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, model.max_grad_norm)
                optimizer.step()
                updates += 1
            states = tuple(value.detach() for value in states)
            count = int(valid.sum())
            squared_error += float(loss.detach()) * count
            frames += count
    return {'mse': squared_error / frames, 'frames': frames, 'updates': updates}


def validate_calibration(config, manifest, checkpoint_sha):
    if (manifest['schema'] != 2 or manifest['control'] != 'learner'
            or manifest['behavior_fingerprint'] != checkpoint_sha):
        raise ValueError("critic calibration requires trajectories from this exact frozen checkpoint")
    for key in ('epochs', 'batch_size', 'sequence_length'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"critic calibration {key} must be a positive integer")
    if config['batch_size'] % config['sequence_length']:
        raise ValueError("critic batch_size must be a multiple of sequence_length")
    rate = config['learning_rate']
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("critic learning_rate must be finite and positive")
