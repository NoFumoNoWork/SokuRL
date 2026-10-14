"""Read-only teacher-label diagnostics on fixed whole-game validation populations."""
import numpy as np
from sb3_contrib import RecurrentPPO

from soku_rl.rl.behavior_cloning import score_samples
from soku_rl.rl.recurrent_cloning import demonstration_episodes, sequence_epoch
from soku_rl.rl.dqn import DoubleDQN
from soku_rl.rl.demonstration_supervision import supervised_samples


def score_validation(model, samples, manifest, batch_size, sequence_length):
    if (type(batch_size) is not int or type(sequence_length) is not int
            or sequence_length < 1 or batch_size < sequence_length or batch_size % sequence_length):
        raise ValueError("validation batch_size must be a positive multiple of sequence_length")
    if set(samples) != {"train", "validation"} or not samples["validation"]:
        raise ValueError("validation requires explicit, nonempty held-out samples")
    by_seat, cursor = {0: [], 1: []}, 0
    for episode in manifest["episodes"]:
        if episode["split"] != "validation":
            continue
        size = episode["steps"]
        rows = samples["validation"][cursor:cursor + size]
        if len(rows) != size or not rows or rows[0][3] != -1 or any(row[3] == -1 for row in rows[1:]):
            raise ValueError("validation manifest and whole-episode sample boundaries disagree")
        by_seat[episode["learner_seat"]].extend(rows)
        cursor += size
    if cursor != len(samples["validation"]) or any(not rows for rows in by_seat.values()):
        raise ValueError("validation must account for all samples and both seats")
    groups = {"overall": samples["validation"], "player_0": by_seat[0], "player_1": by_seat[1]}
    learner_controlled = manifest["schema"] >= 2 and manifest["control"] in {"learner", "teacher_takeover", "teacher_noise"}
    scores = {}
    for name, rows in groups.items():
        if isinstance(model, RecurrentPPO):
            episodes = demonstration_episodes(rows)
            metrics, _, updates = sequence_epoch(model, episodes, np.arange(len(episodes)),
                batch_size, sequence_length, 0., False, 1.)
            assert updates == 0
        else:
            metrics = score_samples(model, rows, batch_size)
        if learner_controlled:
            # These returns came from a different behavior policy. Do not present
            # their MSE as an error against the teacher's value function.
            del metrics["value_mse"]
        labelled = supervised_samples(rows)
        transitions = int(sum(row[3] >= 0 for row in labelled))
        scores[name] = {"frames": len(rows), "episodes": int(sum(row[3] == -1 for row in rows)),
            "metrics": metrics, "copy_previous_action": {"transitions": transitions}}
        if manifest["schema"] == 3:
            scores[name]["supervised_frames"] = len(labelled)
        if transitions:
            scores[name]["copy_previous_action"]["accuracy"] = float(sum(row[3] == 0 for row in labelled) / transitions)
    return {"value_target": "omitted_for_learner_controlled_data" if learner_controlled else
        "recorded_teacher_trajectory_return", "groups": scores,
        "prediction": "softmax_q_ranking_and_teacher_action_q" if isinstance(model, DoubleDQN) else "actor_and_state_value"}
