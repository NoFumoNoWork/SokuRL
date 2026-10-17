"""Command duration, native event reward and PPO/SMDP contracts."""
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from soku_rl.env.v3c import CommandExecution, V3Ledger, vocabulary
from soku_rl.rl.command_ppo import DurationRolloutBuffer


def states():
    return [dict(x=0., facing=1, action=0), dict(x=300., facing=-1, action=0)]


def action(action_id, before=0, **extra):
    return dict(kind='ACTION_TRANSITION', attacker_seat=0, action_before=before,
                action_after=action_id, accepted=True, lineage='DIRECT', native_input_scope=1,
                native_inputs_before=(0, -1, 0, 0, 0, 0, 0, 0), **extra)


def hit(n, combo=1):
    return dict(kind='HIT_RESOLUTION', attacker_seat=0, hp_before=10000,
                hp_after=9900, hit_or_block='HIT', native_combo_id=combo,
                combo_count_after=n, source_action=400)


def test_hit_index_not_attack_segment_and_no_defender_discount():
    ledger = V3Ledger()
    for n, factor in ((1,.5),(2,1),(3,1.2),(4,1.4),(5,1.6),(9,2.4)):
        parts = ledger.step([hit(n)], states(), states())
        assert 100 + parts[0]['combo_hit_scaling'] == pytest.approx(100 * factor)
        assert parts[1]['combo_hit_scaling'] == 0
    assert ledger.step([hit(1,2)], states(), states())[0]['combo_hit_scaling'] == -50
    unknown = hit(20,3)
    unknown['native_combo_id'] = None
    assert ledger.step([unknown], states(), states())[0]['combo_hit_scaling'] == -50
    assert ledger.totals[0]['unknown_hit_ordinals'] == 1


def test_repeated_jump_and_backdash_only_native_transitions():
    ledger = V3Ledger()
    assert ledger.step([action(8)], states(), states())[0]['back_jump'] == 0
    assert ledger.step([action(8)], states(), states())[0]['back_jump'] == -40
    assert ledger.step([], states(), states())[0]['back_jump'] == 0
    assert ledger.step([action(8,before=8)], states(), states())[0]['back_jump'] == 0
    assert ledger.step([action(7),action(6),action(200)], states(), states())[0]['back_d'] == 0
    assert ledger.step([action(201)], states(), states())[0]['back_d'] == -20
    assert ledger.step([action(203,before=70)], states(), states())[0]['back_d'] == 0
    # One forward frame must not erase the avoidance sequence.
    after = states(); after[0].update(x=50., action=4)
    ledger.step([], states(), after)
    assert ledger.step([action(8)], after, after)[0]['back_jump'] == -80
    ledger.step([hit(1)], states(), states())
    assert ledger.step([action(8)], states(), states())[0]['back_jump'] == 0


def test_shared_command_player_facing_hold_and_interrupt():
    program = dict(vocabulary())['236B']
    state = dict(action=0, facing=-1, segment=5, frame_flags=0, hitstop=0)
    execution = CommandExecution(program, state)
    from soku_rl.env.encoding import decode_action
    keys = []
    for frame in range(8):
        keys.append(decode_action(execution.action(frame, state)).inputs)
    assert keys[2][0] == -1 and keys[4][0] == -1
    assert keys[4][3] == 1
    assert execution.done
    execution = CommandExecution(dict(vocabulary())['2B{40}'], state)
    execution.action(0,state)
    execution.observe([],0,state | dict(action=70))
    assert execution.done and execution.interrupted


@pytest.mark.parametrize('durations', [[[1,1],[1,1],[1,1]], [[2,3],[4,1],[1,2]]])
def test_duration_gae_matches_formula_and_v2_sb3(durations):
    kwargs = dict(buffer_size=3, observation_space=spaces.Box(-1,1,(2,),np.float32),
                  action_space=spaces.Discrete(3), n_envs=2, gamma=.97, gae_lambda=.91)
    buffer = DurationRolloutBuffer(**kwargs)
    ordinary = RolloutBuffer(**kwargs)
    for b in (buffer,ordinary):
        b.rewards[:] = [[1,2],[3,4],[5,6]]
        b.values[:] = [[.1,.2],[.3,.4],[.5,.6]]
        b.episode_starts[:] = [[1,1],[0,0],[0,1]]
    buffer.durations[:] = durations
    dones = np.array([False, True])
    last = torch.tensor([.7,.8])
    buffer.compute_returns_and_advantage(last,dones)
    gae = np.zeros(2)
    for s in reversed(range(3)):
        live = 1 - (dones if s == 2 else buffer.episode_starts[s+1])
        nxt = last.numpy() if s == 2 else buffer.values[s+1]
        d = np.asarray(durations[s])
        gae = buffer.rewards[s] + .97 ** d * nxt * live - buffer.values[s] + (.97*.91) ** d * live * gae
        np.testing.assert_allclose(buffer.advantages[s],gae,rtol=1e-6)
    if np.all(np.asarray(durations) == 1):
        ordinary.compute_returns_and_advantage(last,dones)
        np.testing.assert_array_equal(buffer.advantages,ordinary.advantages)


def test_exhausted_slot_samples_do_not_enter_ppo():
    b = DurationRolloutBuffer(2,spaces.Box(-1,1,(2,),np.float32),spaces.Discrete(3),n_envs=2)
    b.durations[:] = [[1,0],[2,0]]
    b.actions[:] = [[[1],[2]],[[0],[2]]]
    b.full = True
    actions = torch.cat([batch.actions for batch in b.get(10)]).flatten().tolist()
    assert sorted(actions) == [0,1]


def test_native_movement_entry_and_hook_are_deduplicated():
    ledger = V3Ledger()
    before = states(); after = states(); after[0]['action'] = 8
    ledger.step([], before, after)
    assert ledger.totals[0]['back_jumps'] == 1
    ledger.step([], after, after)
    assert ledger.totals[0]['back_jumps'] == 1
    before = states(); after = states(); after[0]['action'] = 210
    parts = ledger.step([action(210)], before, after)
    assert ledger.totals[0]['back_jumps'] == 2
    assert parts[0]['back_jump'] == -40


def test_sustained_forward_progress_resets_not_one_frame():
    ledger = V3Ledger()
    ledger.step([action(8)],states(),states())
    ledger.step([action(8)],states(),states())
    before = states()
    for _ in range(16):
        after = [dict(s) for s in before]
        after[0].update(x=before[0]['x']+3,action=4)
        ledger.step([],before,after)
        before = after
    assert ledger.step([action(8)],before,before)[0]['back_jump'] == 0


def test_budget_cut_command_is_excluded_and_bootstrap_uses_its_start():
    b = DurationRolloutBuffer(2,spaces.Box(-1,1,(2,),np.float32),spaces.Discrete(3),
                              n_envs=1,gamma=1,gae_lambda=.99)
    b.values[:] = [[2],[7]]
    b.rewards[:] = [[3],[100]]
    b.durations[:] = [[2],[5]]
    b.censored[:] = [[False],[True]]
    b.compute_returns_and_advantage(torch.tensor([999.]),np.array([False]))
    assert b.advantages[0,0] == 8
    assert b.advantages[1,0] == 0
    b.full = True
    assert sum(len(batch.actions) for batch in b.get()) == 1
