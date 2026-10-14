"""Batch stateless greedy DQN inference while retaining other actors' call order."""
import numpy as np

from soku_rl.policy.dqn import DQNEpisode


def evaluation_inference(config):
    enabled = config.get("recurrent_batch", False)
    if type(enabled) is not bool:
        raise ValueError("benchmark.recurrent_batch must be a boolean")
    if enabled:
        from soku_rl.policy.recurrent_batch import recurrent_episode_actions
        return recurrent_episode_actions, "grouped_recurrent_ppo_v1_grouped_greedy_dqn_v1"
    return episode_actions, "grouped_greedy_dqn_v1_other_actors_sequential"


def episode_actions(requests):
    actions, groups = {}, {}
    for key, (actor, observation) in requests.items():
        if isinstance(actor, DQNEpisode):
            groups.setdefault(id(actor.model), []).append((key, actor, observation))
        else:
            actions[key] = actor.act(observation)
    for entries in groups.values():
        observations = [entry[2] for entry in entries]
        if isinstance(observations[0], dict):
            batch = {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
        else:
            batch = np.stack(observations)
        predicted, _ = entries[0][1].model.predict(batch, deterministic=True)
        for (key, _, _), action in zip(entries, predicted, strict=True):
            actions[key] = int(action)
    return actions
