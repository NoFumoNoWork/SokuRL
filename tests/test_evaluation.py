"""Exercise the game contract with a game whose exact payoff is known."""
from dataclasses import asdict, replace
import random

import pytest

from soku_rl.evaluation import make_plan, run_batch, summarize
from soku_rl.pomg import Outcome, TimeStep


class CoinPolicy:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.calls = 0

    def act(self, observation):
        assert observation == self.calls
        self.calls += 1
        return self.rng.randrange(2)


class CoinStrategy:
    name = "coin"
    fingerprint = "coin-v1"

    def spawn(self, seed):
        return CoinPolicy(seed)


class MatchingCoins:
    """Player one wins on equal coins; player two wins otherwise."""
    def __init__(self, duration):
        self.duration = duration
        self.frame = 0
        self.closed = False

    def reset(self, seeds):
        self.slots = set(range(len(seeds)))
        return {s: TimeStep(0, (0, 0), (0, 0), Outcome.ONGOING, {}) for s in self.slots}

    def step(self, actions):
        assert set(actions) == self.slots
        self.frame += 1
        result = {}
        for slot, (a, b) in actions.items():
            outcome = Outcome.P1_WIN if a == b else Outcome.P2_WIN
            reward = (1, -1) if a == b else (-1, 1)
            if self.frame < self.duration:
                outcome, reward = Outcome.ONGOING, (0, 0)
            result[slot] = TimeStep(self.frame, (self.frame, self.frame), reward, outcome, {})
        return result

    def close(self):
        self.closed = True


def league():
    strategies = {"coin": CoinStrategy()}
    return strategies, make_plan(strategies, [1, 2, 3], 42, "matching-coins-v1")


def test_isolation_and_replay():
    strategies, plan = league()
    first, second = MatchingCoins(4), MatchingCoins(4)
    a = run_batch(first, plan, strategies, 4)
    b = run_batch(second, plan, strategies, 4)
    assert first.closed and second.closed
    assert a["games"] == b["games"]
    assert a["simulation_steps"] == 24
    summary = summarize(plan, a["games"], 0.05)
    entry = summary["matrix"][0][0]
    assert entry["counts"] == {"win": 3, "loss": 3, "draw": 0, "timeout": 0}
    assert entry["payoff_bounds"] == [0, 0]
    assert entry["eventual_win_bounds"] == [0.5, 0.5]
    assert summary["errors"] == summary["missing_games"] == 0


def test_truncation_is_not_a_draw():
    strategies, plan = league()
    report = run_batch(MatchingCoins(5), plan, strategies, 4)
    assert all(g["returns"] == [0, 0] for g in report["games"])
    entry = summarize(plan, report["games"], 0.05)["matrix"][0][0]
    assert entry["counts"]["timeout"] == 6 and entry["counts"]["draw"] == 0
    assert entry["eventual_win_bounds"] == [0, 1]
    assert entry["payoff_bounds"] == [-1, 1]


def test_terminal_at_limit_takes_precedence():
    strategies, plan = league()
    games = run_batch(MatchingCoins(4), plan, strategies, 4)["games"]
    assert all(g["outcome"] != "time_limit" for g in games)


def test_incomplete_pairs_and_errors_are_not_losses():
    strategies, plan = league()
    games = run_batch(MatchingCoins(1), plan, strategies, 1)["games"]
    games[0] = asdict(plan[0]) | {"status": "error", "error": "intentional"}
    result = summarize(plan, games[:-1], 0.05)
    entry = result["matrix"][0][0]
    assert result["errors"] == result["missing_games"] == 1
    assert entry["planned_blocks"] == 3 and entry["complete_blocks"] == 1
    assert sum(entry["counts"].values()) == 2


def test_duplicate_results_and_tampered_plan_are_rejected():
    strategies, plan = league()
    games = run_batch(MatchingCoins(1), plan, strategies, 1)["games"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize(plan, [games[0], games[0]], 0.05)
    games[0]["policy_seeds"] = (1, 1)
    with pytest.raises(ValueError, match="policy_seeds"):
        summarize(plan, games, 0.05)


def test_plan_identity_and_seat_exchange():
    strategies, plan = league()
    a, b = plan[:2]
    assert a.block_id == b.block_id and a.trial_id != b.trial_id
    assert a.policy_seeds == b.policy_seeds[::-1]
    assert plan == make_plan(strategies, [1, 2, 3], 42, "matching-coins-v1")
    assert plan != make_plan(strategies, [1, 2, 3], 43, "matching-coins-v1")
    with pytest.raises(ValueError):
        make_plan(strategies, [1, 1], 42, "game")


def test_cleanup_on_policy_error():
    class BrokenPolicy:
        def act(self, observation):
            raise RuntimeError("intentional")

    class BrokenStrategy(CoinStrategy):
        def spawn(self, seed):
            return BrokenPolicy()

    _, plan = league()
    game = MatchingCoins(1)
    with pytest.raises(RuntimeError, match="intentional"):
        run_batch(game, plan, {"coin": BrokenStrategy()}, 1)
    assert game.closed


def test_time_step_contract():
    step = TimeStep(1, (0, 0), (0, 0), Outcome.TRUNCATED, {})
    assert step.ended and step.truncated and not step.terminated
    with pytest.raises(ValueError):
        replace(step, rewards=(float("nan"), 0))
