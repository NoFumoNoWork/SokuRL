"""Offline contract tests; fixture Lua is not reported as original God-AI play."""
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from soku_rl.env.observation.animation import frame_width
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.policy.god.query import QueryTeacher, validate_spec
from soku_rl.rl.kickstarting import (KickstartingPPO, ExpertRolloutBuffer,
    attach_kickstarting, expert_loss, expert_weight)
from soku_rl.rl.masked_policy import CardMaskedActorCriticPolicy
from soku_rl.rl.ppo import create_ppo


ROOT = Path(__file__).resolve().parents[2]
API = ROOT/'third_party/th123_ai/source/th123_ai/api.ai'


def interface():
    width = frame_width(0)
    episode = SimpleNamespace(observation_mode='animation', history_frames=8,
        decision_frames=1, latency_frames=0, animation_object_slots=0,
        combat_reward=None, space=lambda: gym.spaces.Box(-np.inf, np.inf, (8*width,), np.float32))
    return LearningInterface(episode, LearningConfig('combat', False, 1, 0.))


def observation(contract, action=0):
    row = np.zeros(frame_width(0), np.float32)
    row[0] = row[10] = 1
    row[3], row[13] = 1, -1
    row[4], row[14] = 1, 0
    row[5] = action
    return np.r_[np.tile(row, 8), np.zeros(8, np.float32)]


def spec(directory):
    return dict(package=str(directory), api_source=str(API), script='fixture.ai',
                reaction_frames=2, instruction_limit=100000, seed=17)


def fixture_teacher(tmp_path, source):
    if not API.is_file():
        pytest.skip('external community API unavailable')
    directory = tmp_path/'script'
    directory.mkdir()
    (directory/'fixture.ai').write_text(source, encoding='ascii')
    contract = interface()
    return contract, QueryTeacher(contract, spec(directory))


def test_delayed_exact_inputs_no_game_control_and_own_history(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, '''
function main()
 while true do
  key_reset()
  if my_act == 400 then key_on(ACT_B) else key_on(ACT_A) end
  yield()
 end
end
''')
    actor = teacher.spawn(17)
    frames = [observation(contract, x) for x in (400, 0, 0, 0)]
    frames[-1][-6] = 0
    frames[-1][-5] = 1  # the student really submitted B after the previous query
    original = [x.copy() for x in frames]
    replies = [actor.query(x) for x in frames]
    assert [x.status for x in replies[:2]] == ['REACTION_WARMUP']*2
    assert replies[2].command == 258  # delayed frame 0, neutral+B
    assert replies[3].command == 257  # delayed frame 1, neutral+A
    assert replies[2].engine_acceptance == 'NOT_OBSERVED'
    assert all(np.array_equal(x, y) for x, y in zip(frames, original))
    assert all(not x.command & 48 for x in replies if x.command is not None)


@pytest.mark.parametrize('expression', [
    'enemy_keys', 'get_special_data(1,10)', 'get_obj_attackarea(1,0,0)',
    'x', 'get_key_stat2(1,4)', 'io.open("unsafe","w")', 'os.execute("unsafe")',
    'python.eval("1")', 'debug.getregistry()',
])
def test_privileged_and_external_dependencies_fail_closed(tmp_path, expression):
    contract, teacher = fixture_teacher(tmp_path, f'function main() local v={expression}; key_on(ACT_A); yield() end')
    actor = teacher.spawn(17)
    replies = [actor.query(observation(contract)) for _ in range(4)]
    assert replies[2].status == 'BLOCKED_SOURCE_DEPENDENCY'
    assert replies[2].action is None and not replies[2].input_legal
    assert replies[3] == replies[2]
    assert 'UNAVAILABLE_STUDENT_FIELD' in replies[2].detail
    assert not (tmp_path/'unsafe').exists()


@pytest.mark.parametrize('keys,status', [
    ('key_on(ACT_AB)', 'MASKED_CARD_INPUT'),
    ('key_on(ACT_A); key_on(ACT_B)', 'UNREPRESENTABLE_INPUT'),
])
def test_masked_and_unrepresentable_not_neutral_labels(tmp_path, keys, status):
    contract, teacher = fixture_teacher(tmp_path, f'function main() while true do {keys}; yield() end end')
    actor = teacher.spawn(17)
    answers = [actor.query(observation(contract)) for _ in range(3)]
    assert answers[-1].status == status
    assert answers[-1].action is None


def test_instruction_budget_and_no_zero_reaction(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, 'function main() while true do end end')
    actor = teacher.spawn(17)
    result = [actor.query(observation(contract)) for _ in range(3)][-1]
    assert result.status == 'BLOCKED_SOURCE_DEPENDENCY'
    assert 'QUERY_INSTRUCTION_LIMIT' in result.detail
    with pytest.raises(ValueError, match='reaction_frames'):
        validate_spec(teacher.spec | dict(reaction_frames=0))


def test_unexecuted_teacher_prefix_is_not_self_supervision(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, 'function main() key_on(ACT_A); while true do yield() end end')
    actor = teacher.spawn(17)
    for _ in range(3):
        answer = actor.query(observation(contract))
    assert answer.status == 'VALID_INPUT' and answer.command == 257
    obs = observation(contract)
    obs[-5] = 1  # student submitted B, not the expert A
    answer = actor.query(obs)
    assert answer.status == 'TEACHER_PREFIX_NOT_EXECUTED'
    assert answer.command == 257 and answer.action is None


def test_seeded_expert_rng_is_episode_local(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, '''
function main() while true do
key_reset(); if math.random(1,2)==1 then key_on(ACT_A) else key_on(ACT_B) end; yield()
end end
''')
    actors = [teacher.spawn(17), teacher.spawn(17)]
    for _ in range(10):
        answers = [actor.query(observation(contract)) for actor in actors]
        assert answers[0] == answers[1]


def test_original_marisa_is_not_silently_given_world_state():
    contract = interface()
    package = ROOT/'third_party/th123_ai/package/th123ai/script'
    if not package.is_dir():
        pytest.skip('external original script absent')
    teacher = QueryTeacher(contract, spec(package) | dict(script='01_marisa_main.ai', instruction_limit=1000000))
    actor = teacher.spawn(17)
    result = [actor.query(observation(contract)) for _ in range(3)][-1]
    assert result.status == 'BLOCKED_SOURCE_DEPENDENCY'
    assert 'UNAVAILABLE_STUDENT_FIELD' in result.detail


def test_ce_valid_denominator_mask_gradient_and_decay():
    class Policy:
        def __init__(self):
            self.logits = torch.tensor([[1., -1.], [-1., 1.], [0., 0.]], requires_grad=True)
        def get_distribution(self, obs):
            return torch.distributions.Categorical(logits=self.logits[obs[:,0].long()])
    policy = Policy()
    obs = torch.arange(3.).reshape(-1, 1)
    labels, mask = torch.tensor([0, 999, 1]), torch.tensor([True, False, True])
    loss = expert_loss(policy, obs, labels, mask, policy.logits.sum())
    expected = torch.nn.functional.cross_entropy(policy.logits[[0,2]], labels[[0,2]])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.equal(policy.logits.grad[1], torch.zeros(2))
    assert expert_loss(policy, obs, labels, ~torch.ones(3, dtype=torch.bool), policy.logits.sum()).item() == 0
    config = dict(lambda_initial=.5, decay_transitions=100)
    assert [expert_weight(config, x) for x in (0,50,100,200)] == [.5,.25,0.,0.]


class TinyEnv(gym.Env):
    def __init__(self, contract):
        self.contract = contract
        self.observation_space, self.action_space = contract.observation_space, contract.action_space
        self.actions = []
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frame = 0
        return observation(self.contract), {}
    def step(self, action):
        self.actions.append(int(action))
        self.frame += 1
        return observation(self.contract), float(action % 3), self.frame == 4, False, {}


def make_model(contract, kind=PPO, weight=0.):
    env = DummyVecEnv([lambda: TinyEnv(contract), lambda: TinyEnv(contract)])
    kwargs = dict(seed=39, n_steps=4, batch_size=4, n_epochs=2, device='cpu',
        policy_kwargs=dict(net_arch=[8], masked_actions=tuple(i for i,c in enumerate(contract.commands) if c&48)))
    if kind is KickstartingPPO:
        kwargs['rollout_buffer_class'] = ExpertRolloutBuffer
    return kind(CardMaskedActorCriticPolicy, env, **kwargs)


def test_zero_lambda_matches_upstream_ppo_exactly_and_save(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, 'function main() while true do key_on(ACT_A); yield() end end')
    cfg = dict(teacher=teacher.spec, lambda_initial=0., decay_transitions=100)
    baseline = make_model(contract)
    baseline.learn(16)
    expected = {k:v.clone() for k,v in baseline.policy.state_dict().items()}
    native_actions = [e.actions for e in baseline.env.envs]
    model = make_model(contract, KickstartingPPO)
    attach_kickstarting(model, contract, cfg, False)
    model.learn(16)
    assert all(torch.equal(expected[k], v) for k,v in model.policy.state_dict().items())
    assert [e.actions for e in model.env.envs] == native_actions
    assert model.kickstarting_state['elapsed'] == 16
    assert model.kickstarting_state['valid'] == 4
    path = tmp_path/'joint.zip'
    model.save(path)
    restored = KickstartingPPO.load(path, env=DummyVecEnv([lambda: TinyEnv(contract)]*2))
    attach_kickstarting(restored, contract, cfg, True)
    assert restored.kickstarting_state == model.kickstarting_state
    with pytest.raises(ValueError, match='identical'):
        changed = KickstartingPPO.load(path, env=DummyVecEnv([lambda: TinyEnv(contract)]*2))
        attach_kickstarting(changed, contract, cfg | dict(lambda_initial=.1), True)


def test_joint_loss_changes_actor_but_not_collected_actions_or_returns(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, 'function main() while true do key_on(ACT_A); yield() end end')
    cfg = dict(teacher=teacher.spec, lambda_initial=1., decay_transitions=100)
    baseline = make_model(contract)
    baseline.learn(8)
    model = make_model(contract, KickstartingPPO)
    attach_kickstarting(model, contract, cfg, False)
    model.learn(8)
    assert [e.actions for e in baseline.env.envs] == [e.actions for e in model.env.envs]
    assert np.array_equal(model.rollout_buffer.returns, baseline.rollout_buffer.returns)
    assert np.array_equal(model.rollout_buffer.log_probs, baseline.rollout_buffer.log_probs)
    assert not torch.equal(model.policy.action_net.weight, baseline.policy.action_net.weight)
    assert model.policy.optimizer.state_dict()['param_groups'][0]['lr'] == baseline.policy.optimizer.state_dict()['param_groups'][0]['lr']
    assert model.kickstarting_state['last_update']['expert_ce'] > 0
    assert model.kickstarting_state['valid'] == 2


def test_no_coverage_blocks_before_optimizer(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, 'function main() local hidden=x end')
    model = make_model(contract, KickstartingPPO)
    attach_kickstarting(model, contract,
        dict(teacher=teacher.spec, lambda_initial=.1, decay_transitions=100), False)
    before = {k:v.clone() for k,v in model.policy.state_dict().items()}
    with pytest.raises(RuntimeError, match='no reliable expert labels'):
        model.learn(8)
    assert all(torch.equal(before[k], v) for k,v in model.policy.state_dict().items())


def test_factory_integration(tmp_path):
    contract, teacher = fixture_teacher(tmp_path, 'function main() while true do key_on(ACT_A); yield() end end')
    config = dict(policy_type='mlp', timeout_payoff='zero_at_horizon',
        ppo=dict(n_steps=4,batch_size=4,n_epochs=1,mask_card_actions=True,policy_kwargs=dict(net_arch=[8])),
        kickstarting=dict(teacher=teacher.spec,lambda_initial=.1,decay_transitions=100))
    model, _ = create_ppo(DummyVecEnv([lambda:TinyEnv(contract)]*2), contract,
                         config, {'kind':'fresh'}, 'cpu', 17)
    assert isinstance(model, KickstartingPPO)
    model.learn(8)
