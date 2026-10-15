"""Keep the actor's pretrained representation fixed during shared PPO updates."""
from types import MethodType


def keep_evaluation_mode(module, mode):
    # Parent .train(), as well as direct branch .train(), must preserve frozen
    # batch-normalization buffers and disable dropout in this representation.
    if type(mode) is not bool:
        raise ValueError("training mode must be a boolean")
    return type(module).train(module, False)


def freeze_actor_representation(model):
    policy = model.policy
    modules = [policy.pi_features_extractor]
    if hasattr(policy, "lstm_actor"):
        if policy.shared_lstm or policy.lstm_critic is None:
            raise ValueError("frozen actor representation requires an independent critic LSTM")
        modules.append(policy.lstm_actor)
    frozen = {id(parameter) for module in modules for parameter in module.parameters()}
    if not frozen or len(frozen) == len(list(policy.parameters())):
        raise ValueError("frozen actor representation requires both frozen and trainable parameters")
    if any("train" in module.__dict__ for module in modules):
        raise ValueError("cannot replace an existing representation training-mode hook")
    for module in modules:
        module.requires_grad_(False)
        for parameter in module.parameters():
            parameter.grad = None
        module.train(False)
        module.train = MethodType(keep_evaluation_mode, module)
    model.frozen_actor_representation = {
        "schema": 1,
        "parameters": [name for name, parameter in policy.named_parameters() if id(parameter) in frozen],
        "frozen_parameter_count": sum(parameter.numel() for parameter in policy.parameters() if id(parameter) in frozen),
        "trainable_parameter_count": sum(parameter.numel() for parameter in policy.parameters() if id(parameter) not in frozen),
        "shared_features": policy.pi_features_extractor is policy.vf_features_extractor,
    }
    return model
