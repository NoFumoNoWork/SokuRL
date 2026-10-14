"""Validate the observation and action contract shared by saved policies."""
from dataclasses import asdict, replace

from omegaconf import OmegaConf

from soku_rl.env.wrappers.learning import LearningConfig
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import PlayerSetup


def read_training_contract(path, interface):
    training = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    episode = EpisodeConfig.from_dict(training["episode"])
    if ("algorithm" in training and training["algorithm"]["name"] == "br"
            and "matchups" in training["algorithm"]
            and training["algorithm"]["matchups"]["mode"] == "sampled"):
        algorithm = training["algorithm"]
        learner = PlayerSetup(**algorithm["matchups"]["learner"])
        if learner not in (interface.episode.match.player_0, interface.episode.match.player_1):
            raise ValueError("BR evaluation must include the trained learner character setup")
        # Seat/other-character changes are explicit in this training contract;
        # frame timing, horizon, observation layout and wrappers stay strict.
        episode = replace(episode, match=interface.episode.match)
    if episode != interface.episode:
        raise ValueError("checkpoint and evaluation episode configurations differ")
    # Artifacts from before learning wrappers use the original identity contract.
    expected = (LearningConfig(**training["wrappers"]) if "wrappers" in training else
                LearningConfig("full", False, 0, 0.))
    if expected != interface.config:
        raise ValueError("checkpoint and evaluation learning wrappers differ")
    return training
