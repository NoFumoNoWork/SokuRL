"""Shared reinforcement-learning settings, independent of MARL scheduling."""


def configure_runtime(config):
    import torch
    count = config["cpu_threads"]
    if type(count) is not int or count < 1:
        raise ValueError("rl.cpu_threads must be a positive integer")
    torch.set_num_threads(count)


def ppo_settings(config):
    shared = config["rl"]
    algorithm = config["algorithm"]
    learner = algorithm["response"] if algorithm["name"] == "psro" else algorithm
    for name in ("policy_type", "timeout_payoff", "ppo", "learner", "dqn"):
        if name not in shared and name not in learner:
            continue
        if learner[name] != shared[name]:
            raise ValueError(f"algorithm-specific {name} differs from the shared rl configuration; configure rl.{name}")
    for option in ("rehearsal", "online_anchor", "online_teacher"):
        if option in learner and (option not in shared or learner[option] != shared[option]):
            raise ValueError(f"algorithm-specific {option} differs from shared rl.{option}")
        if option in shared:
            learner[option] = shared[option]
    if shared["policy_type"] not in {"mlp", "lstm"}:
        raise ValueError("shared PPO policy_type must be mlp or lstm")
    if shared["timeout_payoff"] != "zero_at_horizon":
        raise ValueError("shared PPO requires zero_at_horizon")
    from soku_rl.rl.learner import learner_kind, validate_dqn
    if learner_kind(shared) == "dqn":
        if shared["policy_type"] != "mlp":
            raise ValueError("DQN requires a feedforward policy")
        validate_dqn(shared["dqn"])
    return shared


def validate_payoff(interface, config):
    """The implemented potential telescopes only for undiscounted episode returns."""
    if config["timeout_payoff"] != "zero_at_horizon":
        raise ValueError("PPO requires the declared finite-horizon payoff")
    from soku_rl.rl.learner import learner_kind
    if interface.config.health_potential_scale and config[learner_kind(config)]["gamma"] != 1.:
        raise ValueError("the finite-horizon health potential requires gamma=1")
