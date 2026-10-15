from types import SimpleNamespace
import importlib

import pytest

from soku_rl.env.encoding import AGENTS
from soku_rl.evaluation.tournament import Trial


@pytest.mark.parametrize('mismatch', [False, True])
def test_evaluation_reward_parts_are_observational_and_checked(tmp_path, monkeypatch, mismatch):
    module = importlib.import_module('soku_rl.evaluation.benchmark')
    monkeypatch.setattr(module, 'evaluation_inference', lambda config: (
        lambda entries: {(0, agent): 0 for agent in AGENTS}, 'test'))
    monkeypatch.setattr(module, 'summarize', lambda *args: {})
    observations = {0: dict.fromkeys(AGENTS, 0)}
    class Env:
        num_envs = 1
        def step(self, actions):
            assert actions == {0: dict.fromkeys(AGENTS, 0)}
            infos = {agent: dict(frame=3, outcome='p1_win', base_reward=sign,
                                shaping_reward=sign * .25)
                     for agent, sign in zip(AGENTS, (1, -1))}
            rewards = {agent: sign * (1.5 if mismatch else 1.25)
                       for agent, sign in zip(AGENTS, (1, -1))}
            return observations, {0: rewards}, {0: dict.fromkeys(AGENTS, True)}, {
                0: dict.fromkeys(AGENTS, False)}, {0: infos}
    actor = SimpleNamespace(spawn=lambda seed: object())
    strategies = {name: SimpleNamespace(roles=(actor, actor)) for name in ('a', 'b')}
    plan = [Trial('trial', 'block', 1, ('a', 'b'), ('a-id', 'b-id'), (2, 3), False)]
    call = lambda: module.run_plan(Env(), strategies, plan, {'alpha': .05}, tmp_path,
                                   lambda env, trials: (observations, {}))
    if mismatch:
        with pytest.raises(RuntimeError, match='components do not sum'):
            call()
    else:
        game = call()['games'][0]
        assert game['returns'] == [1.25, -1.25]
        assert game['reward_components_by_seat'] == [
            dict(base=1, shaping=.25), dict(base=-1, shaping=-.25)]
