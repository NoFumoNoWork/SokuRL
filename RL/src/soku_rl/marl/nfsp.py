"""NFSP with on-policy PPO responses and a reservoir-trained average policy."""
from pathlib import Path
import json

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from soku_rl.marl.average import AveragePolicy, BestResponseSamples
from soku_rl.policy.population import UniformPolicy
from soku_rl.policy.contract import read_training_contract
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.ppo import create_ppo
from soku_rl.rl.learner import create_learner, snapshot, save_checkpoint, learner_kind


def train_nfsp(env, config, device, seed, directory):
    if config["policy_type"] != "mlp" or len(env.single_observation_space.shape) != 1:
        raise ValueError("NFSP supervised reservoirs require feedforward numeric observations")
    if (min(config["iterations"], config["timesteps_per_iteration"], config["checkpoint_every"]) < 1
            or not 0 < config["anticipatory_param"] < 1):
        raise ValueError("invalid NFSP budgets or anticipatory probability")
    resume = config["resume"]
    saved = None
    if resume != {"kind": "fresh"}:
        if set(resume) != {"kind", "path", "training_config"} or resume["kind"] != "checkpoint":
            raise ValueError("NFSP resume requires a complete training checkpoint")
        previous = read_training_contract(resume["training_config"], env.interface)["algorithm"]
        previous = {"learner": "ppo", "dqn": {}} | previous
        current = {"learner": "ppo", "dqn": {}} | config
        for key in ("policy_type", "ppo", "average", "anticipatory_param", "learner", "dqn"):
            if previous[key] != current[key]:
                raise ValueError(f"NFSP continuation changed {key}")
        saved = torch.load(resume["path"], map_location="cpu", weights_only=False)
        if saved["format"] != "sokurl-ppo-nfsp-v1":
            raise ValueError("NFSP checkpoint format differs")
    models, averages, views = [], [], []
    try:
        for player in (0, 1):
            view = OpponentMixtureVecEnv(env, player,
                [UniformPolicy("initial", env.single_action_space.n)], [1.], seed + player)
            views.append(view)
            initialized = []
            for name, model_seed in (("response", seed + player), ("average", seed + 2 + player)):
                source = {"kind": "fresh"} if saved is None else {
                    "kind": "checkpoint",
                    "path": str(Path(resume["path"]).parent / saved[f"{name}_models"][player]),
                    "training_config": resume["training_config"],
                }
                factory = create_learner if name == "response" and learner_kind(config) == "dqn" else create_ppo
                model, _ = factory(view, env.interface, config, source, device, model_seed)
                initialized.append(model)
            model, average_model = initialized
            if saved is not None:
                view.rng.bit_generator.state = saved["opponent_rngs"][player]
            model.set_logger(configure(str(directory / f"player_{player}" / "scalars"), ["csv"]))
            models.append(model)
            average = AveragePolicy(average_model, config["average"], seed + 4 + player)
            if saved is not None:
                average.restore(saved["reservoirs"][player])
            averages.append(average)
        first = 0 if saved is None else saved["iteration"]
        reports = []
        for iteration in range(first + 1, first + config["iterations"] + 1):
            frozen = directory / f"iteration-{iteration}"
            frozen.mkdir()
            opponents = []
            for player in (0, 1):
                pair = []
                for name, model in (("response", models[player]), ("average", averages[player].model)):
                    path = frozen / f"{name}-p{player}.zip"
                    model.save(path)
                    copy = type(model).load(path, device=device)
                    pair.append(snapshot(f"{name}-p{player}", copy, path))
                opponents.append(pair)
            learning = []
            for player, model in enumerate(models):
                views[player].opponents = opponents[1 - player]
                views[player].probabilities = np.array([config["anticipatory_param"], 1 - config["anticipatory_param"]])
                # Another learner used the shared game. Reset before collecting.
                if hasattr(model, "replay_buffer") and hasattr(model.replay_buffer, "cut_trajectories"):
                    model.replay_buffer.cut_trajectories()
                model._last_obs = None
                model.learn(config["timesteps_per_iteration"],
                    callback=BestResponseSamples(averages[player]), reset_num_timesteps=False)
                learning.append(averages[player].fit())
            reports.append({"iteration": iteration, "learning": learning})
            if iteration % config["checkpoint_every"] == 0 or iteration == first + config["iterations"]:
                checkpoint = directory / f"checkpoint-{iteration}"
                checkpoint.mkdir()
                response_paths, average_paths = [], []
                for player, model in enumerate(models):
                    response_paths.append(f"response-p{player}.zip")
                    average_paths.append(f"average-p{player}.zip")
                    save_checkpoint(model, checkpoint / response_paths[-1])
                    averages[player].model.save(checkpoint / average_paths[-1])
                torch.save({"format": "sokurl-ppo-nfsp-v1", "iteration": iteration,
                    "response_models": response_paths, "average_models": average_paths,
                    "reservoirs": [average.state() for average in averages],
                    "opponent_rngs": [view.rng.bit_generator.state for view in views]}, checkpoint / "training.pt")
        for player, average in enumerate(averages):
            destination = directory / f"player_{player}"
            destination.mkdir(exist_ok=True)
            snapshot(f"average-p{player}", average.model, destination / "final.zip")
        report = {"format": "sokurl-shared-ppo-v1", "iterations": reports,
                  "resume_checkpoint": str(checkpoint / "training.pt")}
        (directory / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        for view in views:
            view.close()
