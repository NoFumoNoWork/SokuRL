"""Evaluate tensor policies in paired games and retain exact action replays."""
from dataclasses import asdict
import json
import time

import numpy as np

from .env.encoding import AGENTS
from .evaluation import make_plan, summarize


def benchmark(env, strategies, candidate, config, game_identity, directory):
    if candidate not in strategies or len(strategies) < 2:
        raise ValueError("a candidate and opponents are required")
    plan = [t for t in make_plan(strategies, config["world_seeds"], config["policy_seed"], game_identity)
            if candidate in t.players and t.players[0] != t.players[1]]
    (directory / "plan.json").write_text(json.dumps([asdict(t) for t in plan], indent=2), encoding="utf-8")
    records = []
    started = time.perf_counter()
    for start in range(0, len(plan), env.num_envs):
        trials = dict(enumerate(plan[start:start + env.num_envs]))
        seeds = {s: t.world_seed for s, t in trials.items()}
        obs, _ = env.reset(seeds)
        actors = {s: tuple(strategies[name].roles[seat].spawn(t.policy_seeds[seat])
                           for seat, name in enumerate(t.players)) for s, t in trials.items()}
        traces = {s: [] for s in trials}
        returns = {s: np.zeros(2) for s in trials}
        while obs:
            actions = {s: {a: actors[s][i].act(value[a]) for i, a in enumerate(AGENTS)}
                       for s, value in obs.items()}
            for slot, joint in actions.items():
                traces[slot].append([joint[a] for a in AGENTS])
            obs, rewards, terms, truncs, infos = env.step(actions)
            for slot in list(obs):
                returns[slot] += [rewards[slot][a] for a in AGENTS]
                if not (terms[slot][AGENTS[0]] or truncs[slot][AGENTS[0]]):
                    continue
                trial = trials[slot]
                name = trial.trial_id + ".npz"
                np.savez_compressed(directory / name, seed=trial.world_seed,
                                    actions=np.asarray(traces[slot], dtype=np.int16))
                info = infos[slot][AGENTS[0]]
                records.append(asdict(trial) | {"status": "complete", "outcome": info["outcome"],
                    "frames": info["frame"], "returns": returns[slot].tolist(), "replay": name})
                del obs[slot]
        progress = {"games": records, "seconds": time.perf_counter() - started,
                    "summary": summarize(plan, records, config["alpha"])}
        (directory / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    opponents = [entry for row in progress["summary"]["matrix"] for entry in row
                 if entry["row"] == candidate and entry["column"] != candidate]
    beaten = [entry["column"] for entry in opponents if entry["win_by_horizon"] > .5]
    progress["gate"] = {"criterion": "strictly more than half of games won against each beaten opponent",
                        "beaten": beaten, "opponents": len(opponents),
                        "fraction_beaten": len(beaten) / len(opponents),
                        "passed": len(beaten) / len(opponents) >= .5}
    return progress
