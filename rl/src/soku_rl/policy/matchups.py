"""Bind a frozen opponent specification to its declared character setup."""
from dataclasses import replace
from pathlib import PurePosixPath

from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.wrappers.learning import LearningInterface


def opponent_interface(interface, learner, entry):
    match = MatchConfig(PlayerSetup(**learner), PlayerSetup(**entry["setup"]))
    spec = entry["policy"]
    while spec["kind"] in {"action_noise", "block_action_noise"}:
        spec = spec["policy"]
    if spec["kind"] == "rule" and spec["name"] == "god":
        script = spec["rules"]["god"]["script"]
        if script != "character":
            stem = PurePosixPath(script).name
            if not stem[:2].isdigit() or int(stem[:2]) != match.player_1.character:
                raise ValueError("BR god script and opponent character do not match")
    return LearningInterface(replace(interface.episode, match=match), interface.config)


def select_opponents(population, names):
    available = {entry["name"]: entry for entry in population}
    if not available or len(available) != len(population):
        raise ValueError("opponent population must be nonempty with distinct names")
    if names == "all":
        return list(population)
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))):
        raise ValueError("opponent_names must be all or a nonempty list of distinct names")
    missing = set(names) - set(available)
    if missing:
        raise ValueError(f"unknown evaluation opponents: {sorted(missing)}")
    return [available[name] for name in names]
