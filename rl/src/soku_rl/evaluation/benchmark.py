"""Evaluate tensor policies in paired games and retain exact action replays."""
from dataclasses import asdict
import json
import time

import numpy as np

from soku_rl.env.encoding import AGENTS
from soku_rl.evaluation.tournament import make_plan, summarize
from soku_rl.policy.batch import evaluation_inference


def benchmark(env, strategies, candidate, config, game_identity, directory):
    if candidate not in strategies or len(strategies) < 2:
        raise ValueError("a candidate and opponents are required")
    plan = [t for t in make_plan(strategies, config["world_seeds"], config["policy_seed"], game_identity)
            if candidate in t.players and t.players[0] != t.players[1]]
    progress = run_plan(env, strategies, plan, config, directory, reset_trials)
    opponents = [entry for row in progress["summary"]["matrix"] for entry in row
                 if entry["row"] == candidate and entry["column"] != candidate]
    beaten = [entry["column"] for entry in opponents if entry["win_by_horizon"] > .5]
    progress["gate"] = {"criterion": "strictly more than half of games won against each beaten opponent",
                        "beaten": beaten, "opponents": len(opponents),
                        "fraction_beaten": len(beaten) / len(opponents),
                        "passed": len(beaten) / len(opponents) >= .5}
    return progress


def reset_trials(env, trials):
    return env.reset({s: t.world_seed for s, t in trials.items()})


def run_plan(env, strategies, plan, config, directory, reset_batch):
    """Execute explicit paired trials; the reset callback owns matchup selection."""
    if not plan:
        raise ValueError("evaluation requires a nonempty trial plan")
    episode_actions, inference_name = evaluation_inference(config)
    (directory / "plan.json").write_text(json.dumps([asdict(t) for t in plan], indent=2), encoding="utf-8")
    records = []
    started = time.perf_counter()
    for start in range(0, len(plan), env.num_envs):
        trials = dict(enumerate(plan[start:start + env.num_envs]))
        obs, _ = reset_batch(env, trials)
        actors = {s: tuple(strategies[name].roles[seat].spawn(t.policy_seeds[seat])
                           for seat, name in enumerate(t.players)) for s, t in trials.items()}
        traces = {s: [] for s in trials}
        returns = {s: np.zeros(2) for s in trials}
        reward_parts = {s: np.zeros((2, 2)) for s in trials}
        reward_parts_available = dict.fromkeys(trials, True)
        public_hp = {}
        public_loss = {s: np.zeros(2) for s in trials}
        interface = getattr(env, "interface", None)
        if interface is not None and interface.episode.observation_mode == "state":
            from soku_rl.env.observation.visible_state import STATE_FEATURES
            for slot, observation in obs.items():
                last = interface.base_observation(observation[AGENTS[0]])[-STATE_FEATURES:]
                public_hp[slot] = np.asarray([last[5], last[13]], dtype=np.float64)
        while obs:
            predicted = episode_actions({(s, a): (actors[s][i], value[a])
                                         for s, value in obs.items() for i, a in enumerate(AGENTS)})
            actions = {s: {a: predicted[s, a] for a in AGENTS} for s in obs}
            for slot, joint in actions.items():
                traces[slot].append([joint[a] for a in AGENTS])
            try:
                obs, rewards, terms, truncs, infos = env.step(actions)
            except Exception as error:
                failed = []
                for slot in actions:
                    trial = trials[slot]
                    name = "failed-" + trial.trial_id + ".npz"
                    np.savez_compressed(directory / name, seed=trial.world_seed,
                                        actions=np.asarray(traces[slot], dtype=np.int16))
                    failed.append(asdict(trial) | {"slot": slot, "replay": name})
                (directory / "failure.json").write_text(json.dumps({
                    "error": repr(error), "active_trials": failed}, indent=2), encoding="utf-8")
                raise
            for slot in list(obs):
                if slot in public_hp:
                    last = interface.base_observation(obs[slot][AGENTS[0]])[-STATE_FEATURES:]
                    current = np.asarray([last[5], last[13]], dtype=np.float64)
                    public_loss[slot] += np.maximum(public_hp[slot] - current, 0.)
                    public_hp[slot] = current
                returns[slot] += [rewards[slot][a] for a in AGENTS]
                for seat, agent in enumerate(AGENTS):
                    info = infos[slot][agent]
                    if "base_reward" in info and "shaping_reward" in info:
                        reward_parts[slot][seat] += [info["base_reward"], info["shaping_reward"]]
                    else:
                        reward_parts_available[slot] = False
                if not (terms[slot][AGENTS[0]] or truncs[slot][AGENTS[0]]):
                    continue
                trial = trials[slot]
                name = trial.trial_id + ".npz"
                np.savez_compressed(directory / name, seed=trial.world_seed,
                                    actions=np.asarray(traces[slot], dtype=np.int16))
                info = infos[slot][AGENTS[0]]
                records.append(asdict(trial) | {"status": "complete", "outcome": info["outcome"],
                    "frames": info["frame"], "returns": returns[slot].tolist(), "replay": name})
                if reward_parts_available[slot]:
                    if not np.allclose(reward_parts[slot].sum(axis=1), returns[slot], rtol=0, atol=1e-7):
                        raise RuntimeError("evaluation reward components do not sum to returns")
                    records[-1]["reward_components_by_seat"] = [
                        {"base": float(row[0]), "shaping": float(row[1])}
                        for row in reward_parts[slot]]
                if slot in public_hp:
                    records[-1]["public_hp_loss_fraction_by_seat"] = public_loss[slot].tolist()
                    records[-1]["public_hp_final_fraction_by_seat"] = public_hp[slot].tolist()
                if "combat_metrics" in info:
                    records[-1]["combat_metrics_by_seat"] = [
                        infos[slot][agent]["combat_metrics"] for agent in AGENTS]
                if 'combat_reward_summary' in info:
                    records[-1]['combat_reward_summary_by_seat']=[
                        infos[slot][agent]['combat_reward_summary'] for agent in AGENTS]
                if "input_metrics" in info:
                    records[-1]["input_metrics_by_seat"] = [
                        infos[slot][agent]["input_metrics"] for agent in AGENTS]
                del obs[slot]
                progress = {"games": records, "seconds": time.perf_counter() - started,
                            "summary": summarize(plan, records, config["alpha"])}
                if config.get("recurrent_batch", False):
                    progress["policy_inference"] = inference_name
                (directory / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    return progress
