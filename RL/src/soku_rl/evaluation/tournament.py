"""Plan paired games, run isolated policies, and report censored win estimates."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from math import log, sqrt
import time

from soku_rl.pomg import Outcome


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Trial:
    trial_id: str
    block_id: str
    world_seed: int
    players: tuple[str, str]
    strategy_ids: tuple[str, str]
    policy_seeds: tuple[int, int]
    swapped: bool


def make_plan(strategies, world_seeds, policy_seed, game_identity):
    names = list(strategies)
    if not names or any(name != strategies[name].name for name in names):
        raise ValueError("nonempty strategies must be indexed by their names")
    if not world_seeds or len(set(world_seeds)) != len(world_seeds):
        raise ValueError("world seeds must be nonempty and distinct")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in [policy_seed, *world_seeds]):
        raise ValueError("world and policy seeds must be uint32")
    if not game_identity:
        raise ValueError("game identity is required")
    trials = []
    for seed in world_seeds:
        for a, b in itertools.combinations_with_replacement(names, 2):
            ids = (strategies[a].fingerprint, strategies[b].fingerprint)
            block = _digest([game_identity, seed, a, b, ids, policy_seed])
            # Logical agents retain their RNG seeds when their seats are swapped.
            seeds = tuple(int(_digest([block, policy_seed, side])[:8], 16) for side in (0, 1))
            for swapped in (False, True):
                players = (b, a) if swapped else (a, b)
                trials.append(Trial(_digest([block, swapped]), block, seed, players,
                                    ids[::-1] if swapped else ids,
                                    seeds[::-1] if swapped else seeds, swapped))
    return trials


def run_batch(backend, trials, strategies, max_frames):
    """Fail the batch on protocol errors; callers must retain error records."""
    if max_frames < 1 or not trials:
        raise ValueError("positive frame limit and nonempty trials are required")
    started = time.perf_counter()
    try:
        states = dict(backend.reset(tuple(t.world_seed for t in trials)))
        if set(states) != set(range(len(trials))):
            raise RuntimeError("reset returned incorrect slots")
        policies, returns, counts, initial = {}, {}, {}, {}
        for slot, trial in enumerate(trials):
            state = states[slot]
            if state.frame != 0 or state.ended or state.rewards != (0, 0):
                raise RuntimeError("reset must return ongoing frame zero")
            if tuple(strategies[n].fingerprint for n in trial.players) != trial.strategy_ids:
                raise ValueError("strategy implementation differs from the trial plan")
            policies[slot] = tuple(strategies[name].spawn(seed)
                                   for name, seed in zip(trial.players, trial.policy_seeds, strict=True))
            returns[slot] = [0.0, 0.0]
            counts[slot] = [Counter(), Counter()]
            initial[slot] = dict(state.diagnostics)
        reset_seconds = time.perf_counter() - started
        sampling_started = time.perf_counter()
        records, frames, policy_seconds = [], 0, 0.0
        while states:
            actions = {}
            policy_started = time.perf_counter()
            for slot, state in states.items():
                actions[slot] = tuple(p.act(obs) for p, obs in
                                      zip(policies[slot], state.observations, strict=True))
                for side, action in enumerate(actions[slot]):
                    # Optional diagnostics only; the game contract has no rule labels.
                    if hasattr(action, "rule"):
                        counts[slot][side][action.rule] += 1
            policy_seconds += time.perf_counter() - policy_started
            next_states = dict(backend.step(actions))
            if set(next_states) != set(states):
                raise RuntimeError("step returned incorrect slots")
            frames += len(states)
            active = {}
            for slot, state in next_states.items():
                if state.frame != states[slot].frame + 1:
                    raise RuntimeError("step must advance exactly one simulation frame")
                for side in (0, 1):
                    returns[slot][side] += state.rewards[side]
                outcome = state.outcome
                if not state.ended and state.frame == max_frames:
                    outcome = Outcome.TRUNCATED
                if outcome == Outcome.ONGOING:
                    active[slot] = state
                    continue
                records.append(asdict(trials[slot]) | {
                    "status": "complete", "outcome": outcome.value, "frames": state.frame,
                    "returns": returns[slot], "initial": initial[slot],
                    "final": dict(state.diagnostics),
                    "rules": [dict(c) for c in counts[slot]],
                })
            states = active
        return {"games": records, "simulation_steps": frames,
                "reset_seconds": reset_seconds,
                "sampling_seconds": time.perf_counter() - sampling_started,
                "policy_seconds": policy_seconds}
    finally:
        backend.close()


def summarize(plan, records, alpha):
    """Timeouts give bounds, not draws. Intervals treat paired blocks as units."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    indexed = {}
    planned = {t.trial_id: t for t in plan}
    for record in records:
        key = record["trial_id"]
        if key in indexed or key not in planned:
            raise ValueError("duplicate or unplanned trial result")
        expected = asdict(planned[key])
        for field in ("block_id", "world_seed", "swapped"):
            if record[field] != expected[field]:
                raise ValueError(f"result does not match plan: {field}")
        for field in ("players", "strategy_ids", "policy_seeds"):
            if tuple(record[field]) != tuple(expected[field]):
                raise ValueError(f"result does not match plan: {field}")
        if record["status"] not in ("complete", "error"):
            raise ValueError("invalid record status")
        if record["status"] == "complete" and record["outcome"] not in {
                o.value for o in Outcome if o != Outcome.ONGOING}:
            raise ValueError("a complete record requires an ended outcome")
        indexed[key] = record
    names = list(dict.fromkeys(name for t in plan for name in t.players))
    matrix = []
    for a in names:
        row = []
        for b in names:
            relevant = [t for t in plan if sorted(t.players) == sorted((a, b))]
            blocks = {}
            for trial in relevant:
                blocks.setdefault(trial.block_id, []).append(trial)
            counts = Counter(win=0, loss=0, draw=0, timeout=0)
            for trials in blocks.values():
                if len(trials) != 2:
                    raise ValueError("each block must contain two seat-swapped games")
                if not all(t.trial_id in indexed and indexed[t.trial_id]["status"] == "complete"
                           for t in trials):
                    continue
                for trial in trials:
                    outcome = indexed[trial.trial_id]["outcome"]
                    side = int(trial.swapped) if a == b else trial.players.index(a)
                    if outcome == Outcome.TRUNCATED.value:
                        counts["timeout"] += 1
                    elif outcome == Outcome.DRAW.value:
                        counts["draw"] += 1
                    elif outcome == (Outcome.P1_WIN.value if side == 0 else Outcome.P2_WIN.value):
                        counts["win"] += 1
                    else:
                        counts["loss"] += 1
            games = sum(counts.values())
            complete_blocks = games // 2
            entry = {"row": a, "column": b, "planned_blocks": len(blocks),
                     "complete_blocks": complete_blocks, "counts": dict(counts),
                     "all_planned_blocks_complete": complete_blocks == len(blocks)}
            if games:
                w, l, d, t = (counts[k] / games for k in ("win", "loss", "draw", "timeout"))
                radius = sqrt(log(2 / alpha) / (2 * complete_blocks))
                entry.update({
                    "win_by_horizon": w, "timeout_rate": t,
                    "eventual_win_bounds": [w, w + t],
                    "eventual_win_confidence_bounds": [max(0.0, w - radius), min(1.0, w + t + radius)],
                    "payoff_bounds": [w - l - t, w - l + t],
                    "score_bounds": [w + 0.5 * d, w + 0.5 * d + t],
                })
            row.append(entry)
        matrix.append(row)
    return {
        "strategies": names, "matrix": matrix,
        "planned_games": len(plan), "recorded_games": len(records),
        "errors": sum(r["status"] == "error" for r in records),
        "missing_games": len(plan) - len(records),
        "confidence": {"alpha": alpha, "method": "Hoeffding bound over independent seed blocks",
                       "scope": "per matrix cell, not simultaneous across cells",
                       "assumption": "independent draws of world/policy seed blocks; fixed seed grids are descriptive"},
    }
