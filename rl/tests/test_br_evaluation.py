"""A BR benchmark must swap characters with the model and count all outcomes."""
import json
from pathlib import Path
import runpy

import pytest

from soku_rl.evaluation.br import benchmark_br, matchup_plan
from soku_rl.policy.population import SeatPolicies, UniformPolicy
from test_matchup_response import SeatGame
from soku_rl.policy.matchups import select_opponents


@pytest.mark.parametrize("recurrent_batch", [False, True])
@pytest.mark.parametrize("failure_phase", ["loading_policies", "running_games"])
def test_failure_retains_config_and_effective_loaded_numerics(tmp_path, monkeypatch, recurrent_batch, failure_phase):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import soku_rl.policy.loader
    import soku_rl.env.worker_pipe
    import torch
    root = Path(__file__).parents[1]
    source = tmp_path / "training"
    source.mkdir()
    output = tmp_path / "evaluation"
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
        training = compose(config_name="train", overrides=["runtime.command=[python,tools/rollout_worker.py]", "algorithm=br", "track=superhuman",
                           "wrappers=superhuman_learning", "rules=god"])
        cfg = compose(config_name="benchmark_br", overrides=[f'training_directory="{source.as_posix()}"',
                      f'output="{output.as_posix()}"', "runtime.command=[python,tools/rollout_worker.py]",
                      "device=cpu", "require_complete=false",
                      f"+benchmark.recurrent_batch={str(recurrent_batch).lower()}"])
    (source / "config.yaml").write_text(OmegaConf.to_yaml(training, resolve=True))
    (source / "final.zip").write_bytes(b"load failure fixture")

    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)

    def fail_load(name, specification, interface, device):
        if failure_phase == "loading_policies":
            raise RuntimeError("model allocation failed")
        monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
        return UniformPolicy(name, 576)

    def fail_worker(**kwargs):
        raise RuntimeError("worker allocation failed")

    monkeypatch.setattr(soku_rl.policy.loader, "load_policy", fail_load)
    monkeypatch.setattr(soku_rl.env.worker_pipe, "WorkerBackend", fail_worker)
    main = runpy.run_path(str(root / "tools/benchmark_br.py"))["main"]
    error = "model allocation failed" if failure_phase == "loading_policies" else "worker allocation failed"
    with pytest.raises(RuntimeError, match=error):
        main.__wrapped__(cfg)
    report = json.loads((output / "result.json").read_text())
    assert report["success"] is False
    assert report["phase"] == failure_phase
    assert error in report["error"]
    if failure_phase == "running_games":
        assert report["inference_numerics"]["cudnn_deterministic"] is True
        assert report["inference_numerics"]["torch_version"] == torch.__version__
    else:
        assert "inference_numerics" not in report
    assert (output / "config.yaml").is_file()
    saved = OmegaConf.to_container(OmegaConf.load(output / "config.yaml"), resolve=True)
    assert ("grouped_recurrent_ppo" in saved["policy_inference"]) == recurrent_batch
    assert not (output / "plan.json").exists()


class EvaluationGame(SeatGame):
    def __init__(self, outcome):
        self.outcome = outcome
        self.resets = []

    def reset_matchups(self, seeds, matches):
        self.resets.append(dict(matches))
        return super().reset_matchups(seeds, matches)

    def step(self, actions):
        observations, rewards, terms, truncs, infos = super().step(actions)
        for pair in infos.values():
            for info in pair.values():
                info.update(outcome=self.outcome, frame=1)
        return observations, rewards, terms, truncs, infos


def benchmark_inputs():
    strategies = {name: SeatPolicies(name, (UniformPolicy(name, 4), UniformPolicy(name, 4)))
                  for name in ("learned", "reimu", "remilia")}
    learner = {"character": 1, "palette": 0, "deck": 0}
    setups = {name: {"character": c, "palette": 0, "deck": 0}
              for name, c in (("reimu", 0), ("remilia", 6))}
    config = {"world_seeds": [101, 103], "policy_seed": 19, "policy_seed_mode": "common_roles", "alpha": .05}
    return strategies, learner, setups, config


def test_pairing_keeps_logical_policy_seeds_and_hashes_character_setup():
    strategies, learner, setups, config = benchmark_inputs()
    plan = matchup_plan(strategies, "learned", learner, setups, config, "game")
    assert len(plan) == 8
    for trial in plan:
        mate, = [t for t in plan if t.block_id == trial.block_id and t.trial_id != trial.trial_id]
        assert mate.policy_seeds[::-1] == trial.policy_seeds
        assert mate.world_seed == trial.world_seed
        assert trial.match[f"player_{trial.learner_seat}"] == learner
        assert trial.match[f"player_{1-trial.learner_seat}"] == setups[trial.opponent]
    changed = matchup_plan(strategies, "learned", learner | {"character": 2}, setups, config, "game")
    assert not {t.trial_id for t in plan} & {t.trial_id for t in changed}


def test_review_seat_filter_preserves_trial_identity_and_default_pairing():
    strategies, learner, setups, config = benchmark_inputs()
    complete = matchup_plan(strategies, "learned", learner, setups, config, "game")
    for seat in (0, 1):
        selected = matchup_plan(strategies, "learned", learner, setups,
                                config | {"learner_seats": [seat]}, "game")
        assert selected == [t for t in complete if t.learner_seat == seat]
    for invalid in ([], [0, 0], [2], [True], "left"):
        with pytest.raises(ValueError, match="learner_seats"):
            matchup_plan(strategies, "learned", learner, setups,
                         config | {"learner_seats": invalid}, "game")


def test_single_seat_review_finishes_without_fabricated_pair_statistics(tmp_path):
    strategies, learner, setups, config = benchmark_inputs()
    report = benchmark_br(EvaluationGame("p1_win"), strategies, "learned", learner,
                          setups, config | {"learner_seats": [0]}, "game", tmp_path)
    assert len(report["games"]) == 4
    assert report["summary"]["kind"] == "unpaired_review"
    assert report["summary"]["paired_inference"] == "NOT APPLICABLE"


def test_common_role_seeds_are_model_independent_but_trial_identity_is_not():
    strategies, learner, setups, config = benchmark_inputs()
    changed = strategies | {"learned": SeatPolicies("learned", (
        UniformPolicy("learned", 5), UniformPolicy("learned", 5)))}
    first = matchup_plan(strategies, "learned", learner, setups, config, "game")
    second = matchup_plan(changed, "learned", learner, setups, config, "game")
    for before, after in zip(first, second, strict=True):
        assert before.policy_seeds == after.policy_seeds
        assert before.strategy_ids != after.strategy_ids
        assert before.trial_id != after.trial_id
    legacy = config | {"policy_seed_mode": "strategy"}
    first = matchup_plan(strategies, "learned", learner, setups, legacy, "game")
    second = matchup_plan(changed, "learned", learner, setups, legacy, "game")
    assert all(a.policy_seeds != b.policy_seeds for a, b in zip(first, second, strict=True))
    with pytest.raises(ValueError, match="policy_seed_mode"):
        matchup_plan(strategies, "learned", learner, setups, config | {"policy_seed_mode": "invalid"}, "game")


def test_explicit_screening_panel_keeps_declared_names_and_order():
    population = [{"name": "reimu"}, {"name": "remilia"}, {"name": "suwako"}]
    assert select_opponents(population, ["suwako", "reimu"]) == [population[2], population[0]]
    assert select_opponents(population, "all") == population
    for names in ([], ["reimu", "reimu"], "reimu", ["unknown"]):
        with pytest.raises(ValueError):
            select_opponents(population, names)


def test_completed_game_survives_failure_of_another_slot(tmp_path):
    class FailingGame(EvaluationGame):
        def step(self, actions):
            if 0 not in actions:
                saved = json.loads((tmp_path / "progress.json").read_text())
                assert len(saved["games"]) == 1
                assert saved["summary"]["missing_games"] == 7
                raise RuntimeError("remaining slot failed")
            observations, rewards, terms, truncs, infos = super().step(actions)
            for slot, pair in terms.items():
                if slot != 0:
                    for agent in pair:
                        pair[agent] = False
            return observations, rewards, terms, truncs, infos

    strategies, learner, setups, config = benchmark_inputs()
    with pytest.raises(RuntimeError, match="remaining slot failed"):
        benchmark_br(FailingGame("p1_win"), strategies, "learned", learner,
                     setups, config, "game", tmp_path)
    record, = json.loads((tmp_path / "progress.json").read_text())["games"]
    assert (tmp_path / record["replay"]).is_file()
    failures = json.loads((tmp_path / "failure.json").read_text())["active_trials"]
    assert len(failures) == 7
    assert record["trial_id"] not in {trial["trial_id"] for trial in failures}


@pytest.mark.parametrize("outcome", ["p1_win", "p2_win", "double_ko", "time_limit"])
@pytest.mark.parametrize("recurrent_batch", [False, True])
def test_benchmark_counts_seats_and_censoring_without_confusing_timeouts(tmp_path, outcome, recurrent_batch):
    strategies, learner, setups, config = benchmark_inputs()
    config["recurrent_batch"] = recurrent_batch
    game = EvaluationGame(outcome)
    report = benchmark_br(game, strategies, "learned", learner, setups, config, "game", tmp_path)
    assert report["summary"]["missing_games"] == 0
    assert len(report["games"]) == 8
    assert len(report["by_opponent_and_seat"]) == 4
    assert report["combat_summary"]["episodes"] == 8
    assert report["combat_summary"]["measured_episodes"] == 0
    assert ("policy_inference" in report) == recurrent_batch
    for row in report["by_opponent_and_seat"]:
        assert row["games"] == 2
        if outcome in ("double_ko", "time_limit"):
            assert row["counts"][outcome] == 2
            assert row["win_rate"] == 0
        else:
            won = outcome == ("p1_win" if row["learner_seat"] == 0 else "p2_win")
            assert row["win_rate"] == float(won)
            assert row["counts"]["win" if won else "loss"] == 2
    saved = json.loads((tmp_path / "plan.json").read_text())
    for slot, trial in enumerate(saved):
        assert game.resets[0][slot].player_0.character == trial["match"]["player_0"]["character"]
        assert (tmp_path / f'{trial["trial_id"]}.npz').is_file()


def test_combat_summary_follows_learner_when_swapping_seats(tmp_path):
    from soku_rl.env.input_metrics import InputMetrics
    from soku_rl.env.encoding import decode_action

    class CombatGame(EvaluationGame):
        def step(self, actions):
            result = super().step(actions)
            inputs = InputMetrics()
            inputs.step((decode_action(480), decode_action(256)))
            for pair in result[-1].values():
                for seat in (0, 1):
                    pair[f"player_{seat}"]["input_metrics"] = inputs.snapshot(seat)
                    pair[f"player_{seat}"]["combat_metrics"] = {
                        "available": True, "own_hp_loss": 100 + seat * 600,
                        "opponent_hp_loss": 700 - seat * 600, "own_hp_loss_frames": 1,
                        "opponent_hp_loss_frames": 2, "own_final_hp": 9900 - seat * 600,
                        "opponent_final_hp": 9300 + seat * 600,
                        "own_spell_action_entries": seat, "opponent_spell_action_entries": 1 - seat}
            return result

    strategies, learner, setups, config = benchmark_inputs()
    report = benchmark_br(CombatGame("p1_win"), strategies, "learned", learner,
                          setups, config, "game", tmp_path)
    assert report["combat_summary"]["means"]["own_hp_loss"] == 400
    assert report["input_summary"]["pooled"]["own_spell_key_rate"] == .5
    for row in report["by_opponent_and_seat"]:
        seat = row["learner_seat"]
        assert row["combat_summary"]["means"]["own_hp_loss"] == 100 + seat * 600
        assert row["combat_summary"]["action_means"]["own_spell_action_entries"] == seat
        assert row["input_summary"]["pooled"]["own_spell_key_rate"] == 1 - seat
