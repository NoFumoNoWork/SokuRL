"""Construct a fresh policy from a serializable, fingerprinted strategy."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from soku_rl.policy.rules.baselines import TreeConfig, TreePolicy
from soku_rl.policy.rules.community_rules import CommunityConfig, CommunityPolicy
from soku_rl.policy.rules.tactical_rules import TacticalConfig, TacticalPolicy


def rule_implementation():
    """Hash every rule implementation used by training and policy benchmarks."""
    names = ("strategies.py", "baselines.py", "community_rules.py", "observed_rules.py",
             "tactical_rules.py", "tactical_observation.py")
    return hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
                                   for name in names)).hexdigest()


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    kind: str
    config_json: str
    implementation: str

    def __post_init__(self):
        if not self.name or not self.implementation:
            raise ValueError("strategy name and implementation identity are required")
        self.spawn(0)

    @property
    def fingerprint(self):
        value = [self.kind, json.loads(self.config_json), self.implementation]
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def spawn(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("policy seed must be a uint32")
        values = json.loads(self.config_json)
        if self.kind == "tree":
            return TreePolicy(TreeConfig(**values))
        if self.kind == "community":
            return CommunityPolicy(CommunityConfig(**values["rules"]),
                                   TreeConfig(**values["movement"]))
        if self.kind == "tactical":
            return TacticalPolicy(TacticalConfig(**values["rules"]),
                                  TreeConfig(**values["movement"]))
        raise ValueError(f"unsupported strategy kind: {self.kind}")


def strategy_from_config(name, config, implementation):
    if "tactical" in config and name in config["tactical"]:
        rules = config["tactical_defaults"] | config["tactical"][name]
        movement_style = rules.pop("movement")
        movement = config["tree"] | config["overrides"][movement_style] | {"style": movement_style}
        values = {"rules": rules | {"style": name}, "movement": movement}
        kind = "tactical"
    elif name in config["community"]:
        rules = dict(config["community"][name])
        movement_style = rules.pop("movement")
        movement = dict(config["tree"], **config["overrides"][movement_style])
        movement["style"] = movement_style
        values = {"rules": dict(rules, style=name), "movement": movement}
        kind = "community"
    elif name in config["overrides"]:
        values = config["tree"] | config["overrides"][name] | {"style": name}
        kind = "tree"
    else:
        raise ValueError(f"unknown strategy: {name}")
    return Strategy(name, kind, json.dumps(values, sort_keys=True), implementation)
