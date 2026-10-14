from collections import Counter

import pytest

from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS, decode_action, encode_action
from soku_rl.env.input_metrics import InputMetrics, summarize_inputs
from soku_rl.env.match import LEGACY_MATCH
from test_env_timing import RecordingBackend, VISIBILITY


def test_full_command_space_seats_and_snapshot_survive_reset():
    metrics = InputMetrics()
    for action in range(576):
        metrics.step((decode_action(action), decode_action(575 - action)))
    left, right = metrics.snapshot(0), metrics.snapshot(1)
    assert left["own"] == right["opponent"]
    assert left["opponent"] == right["own"]
    assert left["own"]["command_counts"] == {str(action): 1 for action in range(576)}
    summary = summarize_inputs([left])
    assert summary["pooled"]["own_attack_key_rate"] == 7 / 8
    assert summary["pooled"]["own_spell_key_rate"] == .5
    assert summary["pooled"]["own_change_card_key_rate"] == .5
    assert summary["pooled"]["own_right_rate"] == 1 / 3
    assert summary["pooled"]["own_neutral_rate"] == 1 / 576
    assert summary["pooled"]["own_command_change_rate"] == 1
    metrics.reset()
    assert left["frames"] == 576 and metrics.snapshot(0)["frames"] == 0


def test_pooling_excludes_cross_episode_transitions_and_missing_records():
    metrics = InputMetrics()
    metrics.step((decode_action(480), decode_action(256)))
    first = metrics.snapshot(0)
    single = summarize_inputs([first])
    assert "own_command_change_rate" not in single["pooled"]
    metrics.reset()
    for action in [480, 452, 452]:
        metrics.step((decode_action(action), decode_action(256)))
    summary = summarize_inputs([first, metrics.snapshot(0), {"available": False}])
    assert summary["episodes"] == 3 and summary["measured_episodes"] == 2
    assert summary["frames"] == 4 and summary["transitions"] == 2
    assert summary["pooled"]["own_command_change_rate"] == .5
    assert summary["pooled"]["own_mean_run_frames"] == 4 / 3
    assert summary["pooled"]["own_spell_key_rate"] == .5
    assert summary["pooled"]["own_modal_command_fraction"] == .5
    assert summary["pooled"]["opponent_mean_run_frames"] == 2
    assert summarize_inputs([]) == {"episodes": 0, "measured_episodes": 0}
    assert "pooled" not in summarize_inputs([{"available": False}])


def test_applied_frames_include_latency_holds_partial_reset_and_final_partial_decision():
    backend = RecordingBackend()
    env = TwoPlayerVectorEnv(backend, 2,
        EpisodeConfig(17, 4, 3, 5, "diagnostic_state", VISIBILITY, LEGACY_MATCH))
    _, infos = env.reset({0: 1, 1: 2})
    assert "input_metrics" not in infos[0][AGENTS[0]]
    for _ in range(2):
        env.step({slot: dict(zip(AGENTS, (480, 452), strict=True)) for slot in (0, 1)})
    env.reset({0: 3})
    for index in range(6):
        slots = (0, 1) if index < 4 else (0,)
        result = env.step({slot: dict(zip(AGENTS, (480, 452), strict=True)) for slot in slots})
        for slot in slots:
            terminal = result[3][slot][AGENTS[0]]
            info = result[4][slot][AGENTS[0]]
            assert ("input_metrics" in info) == terminal
            if terminal:
                actual = info["input_metrics"]
                assert actual["frames"] == len(backend.inputs[slot]) == 17
                for seat, role in ((0, "own"), (1, "opponent")):
                    commands = [encode_action(joint[seat].inputs) for joint in backend.inputs[slot]]
                    assert actual[role]["command_counts"] == {str(k): v for k, v in Counter(commands).items()}
                    assert actual[role]["changed_commands"] == sum(a != b for a, b in zip(commands, commands[1:]))
                assert actual["own"]["command_counts"] == {"256": 5, "480": 12}
    env.close()


@pytest.mark.parametrize("damage", ["zero_frames", "count", "changes", "command"])
def test_corrupt_completed_metrics_fail(damage):
    metrics = InputMetrics()
    metrics.step((decode_action(256), decode_action(256)))
    record = metrics.snapshot(0)
    if damage == "zero_frames":
        record["frames"] = 0
    elif damage == "count":
        record["own"]["command_counts"]["256"] = 2
    elif damage == "changes":
        record["own"]["changed_commands"] = 1
    else:
        record["own"]["command_counts"] = {"576": 1}
    with pytest.raises(ValueError):
        summarize_inputs([record])
