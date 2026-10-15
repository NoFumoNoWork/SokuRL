"""Enumerate playable opponents and bind original scripts to their characters."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath

from soku_rl.policy.god.package import CHARACTERS


@dataclass(frozen=True)
class Opponent:
    name: str
    label: str
    characters: tuple[int, ...]
    tracks: tuple[str, ...]
    policy: dict
    script: str

    def character(self, selection):
        if selection == "opponent":
            if len(self.characters) != 1:
                raise ValueError(f"{self.name} needs an explicit play.ai.character from {self.characters}")
            return self.characters[0]
        if type(selection) is not int or selection not in self.characters:
            raise ValueError(f"{self.name} supports these AI characters: {self.characters}")
        return selection

    def configuration(self, track, character, rules):
        if track not in self.tracks:
            raise ValueError(f"{self.name} requires one of these tracks: {self.tracks}")
        if type(character) is not int or character not in self.characters:
            raise ValueError(f"{self.name} supports these AI characters: {self.characters}")
        settings = deepcopy(rules)
        if self.policy["kind"] == "rule":
            settings["roster"] = [self.policy["name"]]
        if self.script:
            settings["god"]["script"] = self.script
        return {"name": self.name, "policy": deepcopy(self.policy)}, settings


def opponent_catalog(rules, package, checkpoints):
    result = {}
    for name in rules["roster"]:
        if name == "god":
            continue
        result[name] = Opponent(name, name, (0, 1), ("human", "superhuman"),
                                {"kind": "rule", "name": name}, "")
    for script in sorted(package.files):
        if not script.endswith(".ai") or not script[:2].isdigit() or "_main" not in script:
            continue
        character = int(script[:2])
        if not 0 <= character < len(CHARACTERS):
            raise ValueError(f"script refers to an unsupported character: {script}")
        standard = package.character_script(character)
        name = "god:" + (CHARACTERS[character] if script == standard else PurePosixPath(script).stem)
        package.source(script)
        result[name] = Opponent(name, PurePosixPath(script).stem, (character,), ("superhuman",),
                                {"kind": "rule", "name": "god"}, script)
    for name, entry in checkpoints.items():
        if name in result:
            raise ValueError(f"duplicate opponent name: {name}")
        characters, tracks = tuple(entry["characters"]), tuple(entry["tracks"])
        if (not characters or any(type(c) is not int or not 0 <= c < 20 for c in characters)
                or not tracks or any(t not in ("human", "superhuman") for t in tracks)):
            raise ValueError(f"invalid opponent capabilities: {name}")
        if entry["policy"]["kind"] not in {
                "checkpoint", "sb3", "sb3_dqn", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo", "onnx", "onnx_recurrent", "onnx_dqn"}:
            raise ValueError(f"unsupported checkpoint opponent: {name}")
        result[name] = Opponent(name, entry["label"], characters, tracks, deepcopy(entry["policy"]), "")
    return result
