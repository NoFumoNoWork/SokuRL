"""Construct a fresh policy from a serializable, fingerprinted strategy."""
from dataclasses import dataclass
import hashlib
import json

from .baselines import TreeConfig, TreePolicy
from .community_rules import CommunityConfig, CommunityPolicy


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
        raise ValueError(f"unsupported strategy kind: {self.kind}")


def strategy_from_config(name, config, implementation):
    if name in config["community"]:
        rules = dict(config["community"][name])
        movement_style = rules.pop("movement")
        movement = dict(config["tree"], **config["overrides"][movement_style])
        movement["style"] = movement_style
        values = {"rules": dict(rules, style=name), "movement": movement}
        kind = "community"
    else:
        values = config["tree"] | config["overrides"][name] | {"style": name}
        kind = "tree"
    return Strategy(name, kind, json.dumps(values, sort_keys=True), implementation)
