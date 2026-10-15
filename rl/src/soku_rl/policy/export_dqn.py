"""Compatibility entry for the shared, verified ONNX exporter."""


def export_dqn(config):
    from soku_rl.policy.export_actor import export_actor
    if config["candidate"]["policy"]["kind"] != "sb3_dqn":
        raise ValueError("DQN export requires an sb3_dqn policy")
    return export_actor(config)
