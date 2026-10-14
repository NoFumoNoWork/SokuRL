"""Export and verify BC, PPO, DQN and frozen populations for CPU-only play."""
import json
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="export_policy")
def main(cfg):
    from soku_rl.policy.export_actor import export_actor
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(json.dumps(export_actor(config), indent=2), flush=True)


if __name__ == "__main__":
    main()
