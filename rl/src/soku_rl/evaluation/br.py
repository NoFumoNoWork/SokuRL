"""Evaluate one BR on paired seats with the correct character in each seat."""
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json

from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.combat_metrics import summarize_combat
from soku_rl.env.input_metrics import summarize_inputs
from soku_rl.evaluation.benchmark import run_plan
from soku_rl.evaluation.tournament import Trial, make_plan
from soku_rl.pomg import Outcome


@dataclass(frozen=True, slots=True)
class MatchupTrial(Trial):
    match: dict
    learner_seat: int
    opponent: str


def reset_matchup_trials(env, trials):
    return env.reset_matchups({s: t.world_seed for s, t in trials.items()},
                             {s: MatchConfig(**t.match) for s, t in trials.items()})


def matchup_plan(strategies, candidate, learner, setups, config, game_identity):
    if candidate not in strategies or set(setups) != set(strategies) - {candidate}:
        raise ValueError("each evaluated opponent requires a character setup")
    if config["policy_seed_mode"] not in {"strategy", "common_roles"}:
        raise ValueError("policy_seed_mode must be strategy or common_roles")
    learner = PlayerSetup(**learner)
    plan = []
    for name, setup in setups.items():
        opponent = PlayerSetup(**setup)
        pair = {candidate: strategies[candidate], name: strategies[name]}
        identity = hashlib.sha256(json.dumps([game_identity, asdict(learner), asdict(opponent)],
                                             sort_keys=True).encode()).hexdigest()
        for trial in make_plan(pair, config["world_seeds"], config["policy_seed"], identity):
            if trial.players[0] == trial.players[1]:
                continue
            seat = trial.players.index(candidate)
            if config["policy_seed_mode"] == "common_roles":
                # Different candidate models share logical actor randomness;
                # character setup, opponent and world seed still define a block.
                key = ["br-common-roles-v1", identity, name, trial.world_seed, config["policy_seed"]]
                seeds = tuple(int(hashlib.sha256(json.dumps([key, role], sort_keys=True).encode())
                                  .hexdigest()[:8], 16) for role in ("learner", "opponent"))
                block = hashlib.sha256(json.dumps([trial.block_id, key, seeds], sort_keys=True).encode()).hexdigest()
                trial_id = hashlib.sha256(json.dumps([block, trial.swapped]).encode()).hexdigest()
                trial = replace(trial, block_id=block, trial_id=trial_id,
                                policy_seeds=seeds if seat == 0 else seeds[::-1])
            match = MatchConfig(learner, opponent) if seat == 0 else MatchConfig(opponent, learner)
            plan.append(MatchupTrial(**asdict(trial), match=asdict(match), learner_seat=seat, opponent=name))
    # Keep identical matchups together to reuse native resets; paired seed
    # identities remain unchanged even when the two seats run in separate batches.
    return sorted(plan, key=lambda t: (t.opponent, t.learner_seat, t.world_seed))


def benchmark_br(env, strategies, candidate, learner, setups, config, game_identity, directory):
    plan = matchup_plan(strategies, candidate, learner, setups, config, game_identity)
    report = run_plan(env, strategies, plan, config, directory, reset_matchup_trials)
    groups = {}
    combat_groups = {}
    input_groups = {}
    for game in report["games"]:
        key = (game["opponent"], game["learner_seat"])
        counts = groups.setdefault(key, Counter(win=0, loss=0, double_ko=0, time_limit=0))
        combat = (game["combat_metrics_by_seat"][game["learner_seat"]]
                  if "combat_metrics_by_seat" in game else {"available": False})
        combat_groups.setdefault(key, []).append(combat)
        inputs = (game["input_metrics_by_seat"][game["learner_seat"]]
                  if "input_metrics_by_seat" in game else {"available": False})
        input_groups.setdefault(key, []).append(inputs)
        outcome = game["outcome"]
        if outcome in (Outcome.DRAW.value, Outcome.TRUNCATED.value):
            counts[outcome] += 1
        elif outcome == (Outcome.P1_WIN.value if game["learner_seat"] == 0 else Outcome.P2_WIN.value):
            counts["win"] += 1
        else:
            counts["loss"] += 1
    report["by_opponent_and_seat"] = [{"opponent": name, "learner_seat": seat,
        "opponent_character": setups[name]["character"], "counts": dict(counts),
        "games": sum(counts.values()), "win_rate": counts["win"] / sum(counts.values()),
        "combat_summary": summarize_combat(combat_groups[name, seat]),
        "input_summary": summarize_inputs(input_groups[name, seat])}
        for (name, seat), counts in sorted(groups.items())]
    report["combat_summary"] = summarize_combat([record for records in combat_groups.values() for record in records])
    report["input_summary"] = summarize_inputs([record for records in input_groups.values() for record in records])
    return report
