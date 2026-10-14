"""Episode-stable opponent noise controlled by long-term learner performance."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from soku_rl.policy.action_noise import ActionNoisePolicy
from soku_rl.policy.population import UniformPolicy


class FixedOpponentSchedule:
    def spawn(self, opponent, seed):
        return opponent.spawn(seed), {}

    def observe(self, context, info):
        return {}

    def snapshot(self):
        return {"kind": "fixed"}

    def scalar_metrics(self):
        return {}

    def save(self, checkpoint, steps):
        pass

    def restore(self, source):
        if source["kind"] == "checkpoint":
            path = Path(source["path"]).with_suffix(".curriculum.json")
            if path.exists():
                raise ValueError("adaptive checkpoint requires its original curriculum configuration")


class AdaptiveActionNoise:
    kind = "adaptive_action_noise"

    def __init__(self, config, opponents, probabilities, num_actions):
        fields = {"kind", "initial_random_probability", "min_random_probability",
            "max_random_probability", "target_win_rate", "deadband", "ema_half_life",
            "warmup_episodes", "update_every", "gain", "max_change"}
        if set(config) != fields or config["kind"] != self.kind:
            raise ValueError("invalid adaptive action noise configuration fields")
        for key in fields - {"kind", "warmup_episodes", "update_every"}:
            if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
                raise ValueError(f"curriculum {key} must be finite")
        for key in ("warmup_episodes", "update_every"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"curriculum {key} must be a positive integer")
        if not (0 <= config["min_random_probability"] <= config["initial_random_probability"]
                <= config["max_random_probability"] <= 1):
            raise ValueError("invalid curriculum probability bounds")
        if not (0 < config["target_win_rate"] < 1 and 0 <= config["deadband"]
                < min(config["target_win_rate"], 1 - config["target_win_rate"])
                and config["ema_half_life"] > 0 and config["gain"] > 0
                and 0 < config["max_change"] <= 1):
            raise ValueError("invalid curriculum controller parameters")
        if (type(num_actions) is not int or num_actions < 1 or not opponents
                or len({p.name for p in opponents}) != len(opponents)
                or len(probabilities) != len(opponents)
                or any(not math.isfinite(p) or p < 0 for p in probabilities)
                or not math.isclose(sum(probabilities), 1., abs_tol=1e-8)):
            raise ValueError("invalid curriculum action space or opponent distribution")
        self.config = dict(config)
        self.num_actions = num_actions
        self.identities = [{"name": p.name, "fingerprint": p.fingerprint,
                            "probability": float(w)} for p, w in zip(opponents, probabilities, strict=True)]
        self.decay = 2 ** (-1 / config["ema_half_life"])
        if not 0 < self.decay < 1:
            raise ValueError("curriculum half-life is outside numerical precision")
        self.states = {p.name: {"episodes": 0, "win_numerator": 0., "ema_weight": 0.,
            "random_probability": float(config["initial_random_probability"])} for p in opponents}

    def spawn(self, opponent, seed):
        state = self.states[opponent.name]
        noisy = ActionNoisePolicy(opponent.name, opponent, self.num_actions, state["random_probability"])
        context = {"random_probability": state["random_probability"],
            "controller_episodes_at_start": state["episodes"],
            "training_opponent_fingerprint": noisy.fingerprint}
        return noisy.spawn(seed), {"curriculum": context}

    def observe(self, context, info):
        outcome = info["outcome"]
        player = context["player"]
        if type(player) is not int or player not in (0, 1) or outcome not in (
                "p1_win", "p2_win", "double_ko", "time_limit"):
            raise ValueError("curriculum requires an actual learner seat and terminal outcome")
        state = self.states[context["opponent"]]
        won = outcome == f"p{player + 1}_win"
        state["episodes"] += 1
        state["win_numerator"] = self.decay * state["win_numerator"] + (1 - self.decay) * won
        state["ema_weight"] = self.decay * state["ema_weight"] + (1 - self.decay)
        average = state["win_numerator"] / state["ema_weight"]
        before = state["random_probability"]
        c = self.config
        reason = "interval"
        if state["episodes"] < c["warmup_episodes"]:
            reason = "warmup"
        elif (state["episodes"] - c["warmup_episodes"]) % c["update_every"] == 0:
            error = c["target_win_rate"] - average
            if abs(error) <= c["deadband"]:
                reason = "deadband"
            else:
                # Subtract the deadband for a continuous response at its boundary.
                change = math.copysign(min(c["max_change"],
                    c["gain"] * (abs(error) - c["deadband"])), error)
                after = min(c["max_random_probability"], max(c["min_random_probability"], before + change))
                state["random_probability"] = after
                reason = "clamped" if after == before else "increase_uniform" if after > before else "decrease_uniform"
        return {"opponent": context["opponent"], "episodes": state["episodes"],
            "won": won, "ema_win_rate": average,
            "episode_random_probability": context["curriculum"]["random_probability"],
            "previous_random_probability": before, "next_random_probability": state["random_probability"],
            "reason": reason}

    def snapshot(self):
        states = {}
        for name, state in self.states.items():
            states[name] = dict(state)
            if state["episodes"]:
                states[name]["ema_win_rate"] = state["win_numerator"] / state["ema_weight"]
        return {"kind": self.kind, "schema": 1, "config": dict(self.config),
            "num_actions": self.num_actions, "opponents": self.identities, "states": states}

    def scalar_metrics(self):
        result = {"curriculum/uniform_probability": sum(
            row["probability"] * self.states[row["name"]]["random_probability"] for row in self.identities)}
        for index, row in enumerate(self.identities):
            state = self.states[row["name"]]
            prefix = f"curriculum/opponent_{index}"
            result[f"{prefix}/uniform_probability"] = state["random_probability"]
            result[f"{prefix}/episodes"] = state["episodes"]
            if state["episodes"]:
                result[f"{prefix}/ema_win_rate"] = state["win_numerator"] / state["ema_weight"]
        return result

    def save(self, checkpoint, steps):
        path = Path(checkpoint)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        payload = self.snapshot() | {"checkpoint_sha256": digest, "steps": int(steps)}
        path.with_suffix(".curriculum.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def restore(self, source):
        if source["kind"] != "checkpoint":
            return
        path = Path(source["path"])
        payload = json.loads(path.with_suffix(".curriculum.json").read_text(encoding="utf-8"))
        expected = self.snapshot()
        for key in ("kind", "schema", "config", "num_actions", "opponents"):
            if payload[key] != expected[key]:
                raise ValueError(f"curriculum checkpoint mismatch: {key}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != payload["checkpoint_sha256"]:
                raise ValueError("curriculum checkpoint hash mismatch")
        states = payload["states"]
        if set(states) != set(self.states) or type(payload["steps"]) is not int or payload["steps"] < 0:
            raise ValueError("invalid curriculum checkpoint counters")
        restored = {}
        for name, state in states.items():
            n = state["episodes"]
            if type(n) is not int or n < 0:
                raise ValueError("invalid curriculum episode count")
            keys = {"episodes", "win_numerator", "ema_weight", "random_probability"}
            if set(state) != keys | ({"ema_win_rate"} if n else set()):
                raise ValueError("invalid curriculum state fields")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in state.values()):
                raise ValueError("invalid curriculum checkpoint numbers")
            weight, numerator, p = state["ema_weight"], state["win_numerator"], state["random_probability"]
            if (not 0 <= numerator <= weight <= 1 or not math.isclose(weight, 1 - self.decay ** n, abs_tol=1e-10)
                    or not self.config["min_random_probability"] <= p <= self.config["max_random_probability"]
                    or (n and (weight <= 0 or not math.isclose(state["ema_win_rate"], numerator / weight)))):
                raise ValueError("invalid curriculum checkpoint statistics")
            restored[name] = {key: state[key] for key in keys}
        self.states = restored


class AdaptiveEpisodeMixture(AdaptiveActionNoise):
    """Use the same feedback rule, selecting an intact policy for each game."""
    kind = "adaptive_episode_mixture"

    def spawn(self, opponent, seed):
        state = self.states[opponent.name]
        probability = state["random_probability"]
        # Separate the selection stream from both possible actor streams. Keep
        # the actor's original seed, including at the pure-policy endpoints.
        selection_seed = np.random.SeedSequence(seed).spawn(1)[0]
        uniform = np.random.default_rng(selection_seed).random() < probability
        selected = UniformPolicy(opponent.name, self.num_actions) if uniform else opponent
        identity = ["adaptive-episode-mixture-v1", opponent.fingerprint, self.num_actions, probability]
        context = {"random_probability": probability,
            "controller_episodes_at_start": state["episodes"],
            "training_opponent_fingerprint": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
            "selected_policy": "uniform" if uniform else "original",
            "selected_policy_fingerprint": selected.fingerprint}
        return selected.spawn(seed), {"curriculum": context}

    def observe(self, context, info):
        return super().observe(context, info) | {
            "selected_policy": context["curriculum"]["selected_policy"]}


def create_curriculum(config, opponents, probabilities, num_actions):
    if config == {"kind": "fixed"}:
        return FixedOpponentSchedule()
    if config.get("kind") == "adaptive_block_action_noise":
        from soku_rl.rl.block_curriculum import AdaptiveBlockActionNoise
        return AdaptiveBlockActionNoise(config, opponents, probabilities, num_actions)
    if "kind" in config and config["kind"] == AdaptiveEpisodeMixture.kind:
        return AdaptiveEpisodeMixture(config, opponents, probabilities, num_actions)
    return AdaptiveActionNoise(config, opponents, probabilities, num_actions)
