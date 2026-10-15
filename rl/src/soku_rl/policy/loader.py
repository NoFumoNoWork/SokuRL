"""Load every policy type through the same observation and action contract."""
def load_policy(name, spec, interface, device):
    kind = spec["kind"]
    if kind == "teacher_takeover":
        from soku_rl.policy.takeover import TeacherTakeoverPolicy, validate_takeover_frame
        if (set(spec) != {"kind", "learner", "teacher", "after_frames"}
                or any(not isinstance(spec[key], dict) for key in ("learner", "teacher"))):
            raise ValueError("teacher_takeover requires learner, teacher and after_frames")
        validate_takeover_frame(spec["after_frames"])
        if interface.episode.decision_frames != 1 or interface.episode.latency_frames != 0:
            raise ValueError("teacher takeover requires every-frame decisions and zero latency")
        return TeacherTakeoverPolicy(name,
            load_policy(name + ":learner", spec["learner"], interface, device),
            load_policy(name + ":teacher", spec["teacher"], interface, device), spec["after_frames"])
    if kind == "onnx":
        from soku_rl.policy.portable import load_portable
        if str(device) != "cpu":
            raise ValueError("the deployment model requires device=cpu")
        return load_portable(name, spec["path"], interface)
    if kind == "greedy":
        from soku_rl.policy.greedy import GreedyPPOPolicy
        if (set(spec) != {"kind", "policy"} or not isinstance(spec["policy"], dict)
                or spec["policy"].get("kind") not in {"sb3", "sb3_recurrent", "sb3_dqn"}):
            raise ValueError("greedy requires an sb3, sb3_recurrent or sb3_dqn policy")
        if spec["policy"]["kind"] == "sb3_dqn":
            return load_policy(name, spec["policy"], interface, device)
        return GreedyPPOPolicy(name, load_policy(name, spec["policy"], interface, device))
    if kind == "action_noise":
        from soku_rl.policy.action_noise import ActionNoisePolicy
        if set(spec) != {"kind", "policy", "random_probability"}:
            raise ValueError("action_noise requires policy and random_probability")
        return ActionNoisePolicy(name, load_policy(name, spec["policy"], interface, device),
            int(interface.action_space.n), spec["random_probability"])
    if kind == "block_action_noise":
        from soku_rl.policy.block_noise import BlockActionNoisePolicy
        if set(spec) != {"kind", "policy", "random_probability", "block_decisions"}:
            raise ValueError("block_action_noise requires policy, random_probability and block_decisions")
        return BlockActionNoisePolicy(name, load_policy(name, spec["policy"], interface, device),
            int(interface.action_space.n), spec["random_probability"], spec["block_decisions"])
    if kind == "rule":
        from soku_rl.policy.rules.observed_rules import RulePolicy, LearningRulePolicy
        from soku_rl.policy.rules.strategies import rule_implementation
        rules = spec["rules"]
        if spec["name"] not in rules["roster"]:
            raise ValueError("rule policy must belong to the configured roster")
        policy = RulePolicy(spec["name"], rules, interface.episode, rule_implementation())
        return LearningRulePolicy(policy, interface)
    if kind == "uniform":
        from soku_rl.policy.population import UniformPolicy
        return UniformPolicy(name, interface.action_space.n)
    if kind == "onnx_recurrent":
        from soku_rl.policy.onnx import OnnxPolicy
        if str(device) != "cpu":
            raise ValueError("the deployment model requires device=cpu")
        return OnnxPolicy(name, spec["path"], interface)
    if kind == "onnx_dqn":
        from soku_rl.policy.onnx_dqn import OnnxDQNPolicy
        if str(device) != "cpu":
            raise ValueError("the deployment model requires device=cpu")
        return OnnxDQNPolicy(name, spec["path"], interface)
    if kind in {"sb3", "sb3_dqn", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        from soku_rl.policy.checkpoint import load_checkpoint
        return load_checkpoint(name, spec, interface, device)
    raise ValueError(f"unsupported policy kind: {kind}")
