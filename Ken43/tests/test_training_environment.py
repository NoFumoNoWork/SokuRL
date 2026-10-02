from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

from ken43 import (
    CombatState,
    FighterState,
    FrameEvent,
    FrameInput,
    FrameResolver,
    HeuristicDefenderBot,
    HeuristicOffenseBot,
    Ken43GymEnv,
    Ken43ParallelEnv,
    MashBot,
    PressureState,
    choose_challenge,
    classify_pressure_state,
)
from ken43.parallel_env import MovementState


def event(events, event_type):
    return next(item for item in events if item.type == event_type)


def action_is_available(env, observation, action):
    return bool(observation["action_mask"][int(action)])


def advance_until_idle(resolver, state, first_input):
    state, events = resolver.resolve_frame(state, (first_input, FrameInput()))
    all_events = list(events)
    while state.players[0].action_id is not None:
        state, events = resolver.resolve_frame(state)
        all_events.extend(events)
    return state, all_events


def test_updated_range_brackets_and_di_domain():
    resolver = FrameResolver()

    assert resolver.spacing.in_range("5MP", 131.30, "standing") is True
    assert resolver.spacing.in_range("5MP", 131.45, "standing") is True
    assert resolver.spacing.in_range("5MP", 131.46, "standing") is False
    assert resolver.spacing.effective_range("5MP", "standing") == pytest.approx(131.45)
    assert resolver._in_range("drive_impact", 188.8, "standing") is True
    assert resolver._in_range("drive_impact", 188.81, "standing") is False


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("hold2", 246.90), ("non_hold2", 218.31)],
)
def test_5hp_aggregate_whiff_displacement(mode, expected):
    resolver = FrameResolver()
    state, events = advance_until_idle(
        resolver,
        CombatState(distance=300),
        FrameInput(action="5HP", recovery_mode=mode),
    )

    assert state.distance == pytest.approx(expected)
    movement = [item for item in events if item.type == "Movement"]
    assert movement[-1].data["mode"] == "aggregate_whiff"


def test_block_damage_and_hit_gain_change_drive():
    resolver = FrameResolver()
    blocked = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5HP", action_frame=9, drive=5.0),
            FighterState(drive=6.0),
        ],
    )
    blocked, block_events = resolver.resolve_frame(
        blocked, (FrameInput(), FrameInput(guard="standing"))
    )
    assert blocked.players[1].drive == pytest.approx(5.5)
    assert event(block_events, "Block").data["drive_damage_defender"] == 0.5

    hit = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5HP", action_frame=9, drive=5.0),
            FighterState(drive=6.0),
        ],
    )
    hit, hit_events = resolver.resolve_frame(hit)
    assert hit.players[0].drive == pytest.approx(5.2)
    assert event(hit_events, "Hit").data["drive_gain_attacker"] == 0.2


def test_counter_hit_uses_120_percent_and_two_extra_stun_frames():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5LP", action_frame=3),
            FighterState(action_id="5HP", action_frame=1),
        ],
    )
    state, events = resolver.resolve_frame(state)
    hit = event(events, "Hit")

    assert hit.data["counter_context"] == "counter_hit"
    assert hit.data["damage"] == 360
    assert hit.data["stun"] == 16
    assert state.players[1].hp == 9640


def test_third_combo_move_uses_base_and_accumulated_starter_scaling():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(
                action_id="5LP",
                action_frame=3,
                combo_move_index=2,
                combo_penalty=0.20,
            ),
            FighterState(stun_remaining=5, stun_kind="hit"),
        ],
    )
    state, events = resolver.resolve_frame(state)
    hit = event(events, "Hit")

    assert hit.data["damage_scale"] == pytest.approx(0.60)
    assert hit.data["damage"] == 180
    assert state.players[0].combo_move_index == 3


def test_throw_tech_gives_both_players_half_a_bar():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="throw_forward", action_frame=4, drive=4.0),
            FighterState(action_id="throw_forward", action_frame=4, drive=5.0),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert event(events, "ThrowClash").data["drive_gain_each"] == 0.5
    assert state.players[0].drive == pytest.approx(4.5)
    assert state.players[1].drive == pytest.approx(5.5)


def test_l_tatsu_unknown_single_hit_damage_is_explicitly_incomplete():
    resolver = FrameResolver()
    state = CombatState(
        distance=130,
        players=[FighterState(action_id="tatsu_L", action_frame=3), FighterState()],
    )
    state.players[0].action_start_distance = 130
    state, events = resolver.resolve_frame(state)

    hit = event(events, "Hit")
    assert hit.data["hit_count"] == 1
    assert hit.data["damage"] == 0
    assert event(events, "DataIncomplete").data["field"] == "single_hit_damage"


def test_sacrifice_jump_is_throw_invulnerable_then_escapes_remaining_actives():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[FighterState(), FighterState(action_id="5HP", action_frame=5)],
    )
    state, _ = resolver.resolve_frame(
        state, (FrameInput(action="sacrifice_jump"), FrameInput())
    )
    assert resolver._throwable(state.players[0]) is False
    assert state.players[0].airborne is False
    for _ in range(4):
        state, _ = resolver.resolve_frame(state)
    assert state.players[0].sacrifice_phase == "airborne_escape"
    assert state.players[0].airborne is True
    assert state.distance == 70

    while state.players[0].sacrifice_phase == "airborne_escape":
        state, _ = resolver.resolve_frame(state)
    assert state.players[0].sacrifice_phase == "landing"
    for _ in range(3):
        state, _ = resolver.resolve_frame(state)
    assert state.players[0].action_id is None


def test_di_block_creates_72f_attacker_relative_wall_splat_window():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[FighterState(action_id="drive_impact", action_frame=25), FighterState()],
    )
    state, events = resolver.resolve_frame(
        state, (FrameInput(), FrameInput(guard="standing"))
    )
    assert event(events, "SpecialStun").data["kind"] == "wall_splat"

    while state.players[0].action_id is not None:
        state, _ = resolver.resolve_frame(state)
    assert state.players[1].special_stun == "wall_splat"
    assert state.players[1].special_stun_remaining == 72


def test_di_armor_conversion_uses_pc_numeric_profile_without_pc_flag():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(
                action_id="drive_impact",
                action_frame=25,
                drive=5.0,
                armor_hits_remaining=1,
                armor_absorbed_hit=True,
            ),
            FighterState(drive=3.0),
        ],
    )
    state, events = resolver.resolve_frame(state)
    hit = event(events, "Hit")

    assert hit.data["armor_conversion"] is True
    assert hit.data["counter_context"] == "normal"
    assert hit.data["damage"] == 960
    assert hit.data["drive_damage_defender"] == 1.5
    assert state.players[1].drive == pytest.approx(1.5)
    assert state.players[1].special_stun == "crumple"


def test_drive_reversal_can_spend_last_bars_and_resolve_burnout_after_contact():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(),
            FighterState(drive=2.0, stun_remaining=5, stun_kind="block", blocking=True),
        ],
    )
    state, events = resolver.resolve_frame(
        state, (FrameInput(), FrameInput(reversal="drive_reversal"))
    )
    assert event(events, "ReversalStarted").data["drive_cost"] == 2.0
    assert state.players[1].drive == 0.0
    assert state.terminal_reason is None

    while state.terminal_reason is None:
        state, _ = resolver.resolve_frame(state)
    assert state.terminal_reason == "burnout_neutral_reset"
    assert state.players[1].burnout is True


def test_parallel_environment_api_and_dynamic_masks():
    env = Ken43ParallelEnv(max_frames=12)
    observations, _ = env.reset()
    di_action = env.command_index["immediate:drive_impact"]
    assert action_is_available(env, observations["player_0"], di_action)

    env.core.combat_state.distance = 200
    assert not action_is_available(env, env._observations()["player_0"], di_action)
    parallel_api_test(Ken43ParallelEnv(max_frames=8), num_cycles=20)


def test_random_masked_rollout_has_finite_observations_and_rewards():
    rng = np.random.default_rng(7)
    env = Ken43ParallelEnv(max_frames=90)
    observations, _ = env.reset()
    for _ in range(90):
        actions = {}
        for agent in env.agents:
            actions[agent] = env.command_index["noop"]
        observations, rewards, terminated, truncated, infos = env.step(actions)
        assert all(np.isfinite(value) for value in rewards.values())
        assert all(np.all(np.isfinite(obs["observation"])) for obs in observations.values())
        assert not any(
            item["type"] == "DataIncomplete"
            for item in infos["player_0"]["events"]
        )
        if all(terminated.values()) or all(truncated.values()):
            break


def test_drive_reward_uses_signed_drive_change_and_is_zero_sum():
    env = Ken43ParallelEnv(
        drive_reward_weight=100.0,
        drive_regen_per_frame=0.0,
    )

    assert env._drive_change_reward(1.0, 0.5) == pytest.approx(0.5)
    assert env._drive_change_reward(2.0, 1.5) == pytest.approx(0.5)
    assert env._drive_change_reward(1.0, 1.5) == pytest.approx(-0.5)

    observations, _ = env.reset(options={"player_0_drive": 6.0, "player_1_drive": 1.0})
    env.core.combat_state.players[1].knockdown = False
    action = env.command_index["immediate:5HP"]
    guard = env.command_index["guard:standing"]
    for _ in range(20):
        observations, rewards, _, _, infos = env.step(
            {"player_0": action, "player_1": guard}
        )
        action = env.command_index["noop"]
        if infos["player_0"]["reward_components"]["drive"]:
            break

    components = infos["player_0"]["reward_components"]
    assert components["opponent_drive_change_reward"] == pytest.approx(0.5)
    assert components["own_drive_change_reward"] == 0.0
    assert components["drive"] == pytest.approx(50.0)
    assert rewards["player_0"] == pytest.approx(50.0)
    assert rewards["player_1"] == pytest.approx(-50.0)


def test_single_agent_gymnasium_facade_passes_checker():
    env = Ken43GymEnv(opponent=MashBot(), max_frames=12)
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)


def test_corner_action_catalog_includes_all_three_frame_kills():
    env = Ken43ParallelEnv(schedule_max_delay=3)
    assert "immediate:5MK" in env.command_index
    assert "immediate:shoryuken_L" in env.command_index
    assert "immediate:shoryuken_OD" in env.command_index
    assert "immediate:drive_impact" in env.command_index
    assert "immediate:hadoken_L" not in env.command_index
    assert "immediate:shoryuken_M" not in env.command_index
    assert "reversal:drive_reversal" not in env.command_index
    assert "immediate:2HK" in env.command_index
    assert "immediate:5HK" in env.command_index
    assert env.unavailable_requested_actions == {}


def test_flat_action_catalog_contains_only_complete_legal_commands():
    env = Ken43ParallelEnv(schedule_max_delay=3)
    observations, _ = env.reset()

    assert env.action_space("player_0").n == len(env.commands)
    assert observations["player_0"]["action_mask"].shape == (len(env.commands),)
    assert "schedule:3:5LP" in env.command_index
    assert "schedule:3:guard:crouching" not in env.command_index
    assert "schedule:0:5LP" not in env.command_index
    assert all(
        not label.startswith("schedule:") or label.split(":", 2)[2] in env._root_actions
        for label in env.commands
    )


def test_jhp_becomes_available_during_forward_jump():
    env = Ken43ParallelEnv(max_frames=80)
    observations, _ = env.reset()
    noop = env.command_index["noop"]
    observations, _, _, _, _ = env.step({
        "player_0": env.command_index["immediate:forward_jump"], "player_1": noop,
    })
    for _ in range(5):
        observations, _, _, _, _ = env.step({"player_0": noop, "player_1": noop})
    assert env.core.combat_state.players[0].airborne
    assert action_is_available(env, observations["player_0"], env.command_index["immediate:jHP"])


def test_drive_only_observation_has_no_react_or_recognition_fields():
    env = Ken43ParallelEnv(schedule_max_delay=2)
    observations, _ = env.reset()
    assert observations["player_0"]["observation"].shape == (32,)
    assert not any(key.startswith("react:") for key in env.command_index)


def test_schedule_samples_once_and_masks_replacement_until_execution():
    env = Ken43ParallelEnv(schedule_max_delay=5)
    observations, _ = env.reset(seed=9)
    command = env.command_index["schedule:5:5LP"]
    noop = env.command_index["noop"]
    observations, _, _, _, _ = env.step({"player_0": command, "player_1": noop})
    pending = env._pending[0]
    assert pending is not None
    assert pending.actual_delay in {3, 4, 5, 6, 7}
    assert observations["player_0"]["action_mask"][command] == 0
    for _ in range(8):
        observations, _, terminated, truncated, _ = env.step(
            {"player_0": command, "player_1": noop}
        )
        if env.core.combat_state.players[0].action_id == "5LP":
            break
        assert not any(terminated.values()) and not any(truncated.values())
    assert env.core.combat_state.players[0].action_id == "5LP"
    assert env._pending[0] is None


def test_schedule_is_limited_to_ten_frames_and_keeps_plus_minus_two_noise():
    with pytest.raises(ValueError, match="1..10"):
        Ken43ParallelEnv(schedule_max_delay=11)
    env = Ken43ParallelEnv(schedule_max_delay=10)
    env.reset(seed=4)
    action = env.command_index["schedule:10:5LP"]
    env.step({"player_0": action, "player_1": env.command_index["noop"]})
    assert env._pending[0].actual_delay in {8, 9, 10, 11, 12}
    assert "schedule:11:5LP" not in env.command_index


def test_jinrai_followup_delay_is_relative_to_first_branch_window():
    def branch_frame(delay):
        env = Ken43ParallelEnv(max_frames=120)
        observations, _ = env.reset()
        env.core.combat_state.players[1].knockdown = False
        noop = env.command_index["noop"]
        env.step({"player_0": env.command_index["immediate:jinrai_M"], "player_1": noop})
        env.step({"player_0": env.command_index[f"branch:{delay}:kazekama"], "player_1": noop})
        for _ in range(60):
            _, _, terminated, truncated, infos = env.step({"player_0": noop, "player_1": noop})
            started = [item for item in infos["player_0"]["events"] if item["type"] == "BranchStarted"]
            if started:
                return env.core.combat_state.frame
            assert not any(terminated.values()) and not any(truncated.values())
        raise AssertionError("Jinrai branch never started")

    assert branch_frame(2) - branch_frame(0) == 2


def test_jinrai_branch_claim_cannot_be_replaced_while_pending():
    env = Ken43ParallelEnv(max_frames=120)
    observations, _ = env.reset()
    env.core.combat_state.players[1].knockdown = False
    noop = env.command_index["noop"]
    env.step({"player_0": env.command_index["immediate:jinrai_M"], "player_1": noop})
    observations, _, _, _, _ = env.step({
        "player_0": env.command_index["branch:5:kazekama"], "player_1": noop,
    })
    assert env._pending_branch[0] is not None
    assert not action_is_available(
        env, observations["player_0"], env.command_index["branch:0:gorai"]
    )


def test_successful_od_dp_pays_cost_then_terminates_as_pressure_escape():
    env = Ken43ParallelEnv(max_frames=100, drive_regen_per_frame=0.0)
    env.reset()
    env.core.combat_state.players[1].knockdown = False
    noop = env.command_index["noop"]
    total = 0.0
    action = env.command_index["immediate:shoryuken_OD"]
    for _ in range(20):
        _, rewards, terminated, _, infos = env.step({"player_0": noop, "player_1": action})
        action = noop
        total += rewards["player_1"]
        if any(terminated.values()):
            break
    assert infos["player_1"]["terminal_reason"] == "pressure_escape"
    assert env.core.combat_state.players[1].drive == pytest.approx(4.0)
    assert total == pytest.approx(0.0)


def test_blocked_od_dp_has_no_refund_or_escape_bonus():
    env = Ken43ParallelEnv(max_frames=100, drive_regen_per_frame=0.0)
    env.reset()
    env.core.combat_state.players[1].knockdown = False
    guard = env.command_index["guard:standing"]
    noop = env.command_index["noop"]
    action = env.command_index["immediate:shoryuken_OD"]
    saw_block = False
    for _ in range(20):
        _, _, terminated, _, infos = env.step({"player_0": guard, "player_1": action})
        action = noop
        saw_block |= any(item["type"] == "Block" for item in infos["player_0"]["events"])
        assert infos["player_0"]["terminal_reason"] != "pressure_escape"
        assert not any(terminated.values())
    assert saw_block
    assert env.core.combat_state.players[1].drive == pytest.approx(4.0)
    assert env.core.combat_state.players[1].action_id == "shoryuken_OD"


def test_drive_regen_waits_for_configured_delay():
    env = Ken43ParallelEnv(drive_regen_per_frame=0.1, drive_regen_delay=2)
    env.reset(options={"player_0_drive": 5.0})
    env._regen_cooldown[0] = 2
    noop = env.command_index["noop"]
    for expected in (5.0, 5.0, 5.1):
        env.step({"player_0": noop, "player_1": noop})
        assert env.core.combat_state.players[0].drive == pytest.approx(expected)


def test_distance_beyond_2mk_marks_disengaged_without_terminating():
    env = Ken43ParallelEnv(initial_advantages=(26, 38, 43))
    observation, info = env.reset(options={"initial_advantage": 38, "distance": 186.01})
    assert info["player_0"]["initial_advantage"] == 38
    noop = env.command_index["noop"]
    _, rewards, terminated, _, infos = env.step({"player_0": noop, "player_1": noop})
    assert not any(terminated.values())
    assert infos["player_0"]["terminal_reason"] is None
    assert infos["player_0"]["disengaged"] is True
    assert rewards["player_1"] == rewards["player_0"] == 0.0


def test_attacker_pushback_disengage_is_not_a_defensive_win():
    env = Ken43ParallelEnv()
    env.reset()
    state = env.core.combat_state
    state.distance = 200.0
    events = [FrameEvent("Block", 0, 1, {"action": "5HP", "counter_context": "normal"})]
    bonuses = env._terminal_and_event_bonuses(
        state, events, ["immediate:noop", "immediate:guard:standing"], 180.0
    )
    assert state.terminal_reason is None
    assert env._disengaged is True
    assert env._disengage_cause == "attacker_contact_pushback"
    assert bonuses == [0.0, 0.0]


def test_defender_interrupt_that_disengages_terminates_with_small_bonus():
    env = Ken43ParallelEnv()
    env.reset()
    state = env.core.combat_state
    state.distance = 200.0
    events = [FrameEvent(
        "Hit", 1, 0,
        {"action": "5LP", "counter_context": "counter_hit"},
    )]
    bonuses = env._terminal_and_event_bonuses(
        state, events, ["immediate:5HP", "immediate:5LP"], 180.0
    )
    assert state.terminal_reason == "defender_disengage"
    assert bonuses[1] == pytest.approx(0.15)
    assert bonuses[0] == pytest.approx(-0.15)


def test_forward_walk_regen_accelerates_after_configured_streak():
    env = Ken43ParallelEnv(
        drive_regen_per_frame=0.1,
        drive_regen_delay=0,
        forward_walk_regen_after=11,
        forward_walk_regen_multiplier=1.5,
    )
    env.reset(options={"player_0_drive": 3.0})
    walk = env.command_index["immediate:forward_walk"]
    guard = env.command_index["guard:crouching"]
    for _ in range(10):
        env.step({"player_0": walk, "player_1": guard})
    assert env.core.combat_state.players[0].drive == pytest.approx(4.0)
    env.step({"player_0": walk, "player_1": guard})
    assert env.core.combat_state.players[0].drive == pytest.approx(4.15)


def test_attacking_does_not_passively_regenerate_drive():
    env = Ken43ParallelEnv(drive_regen_per_frame=0.1, drive_regen_delay=0)
    env.reset(options={"player_0_drive": 5.0})
    env.step({
        "player_0": env.command_index["immediate:5HP"],
        "player_1": env.command_index["guard:crouching"],
    })
    assert env.core.combat_state.players[0].drive == pytest.approx(5.0)


def test_defender_regen_offsets_attacker_drive_reward():
    env = Ken43ParallelEnv(drive_regen_per_frame=0.1, drive_regen_delay=0)
    env.reset(options={"player_1_drive": 5.0})
    guard = env.command_index["guard:crouching"]
    _, rewards, _, _, infos = env.step({"player_0": guard, "player_1": guard})
    assert env.core.combat_state.players[1].drive == pytest.approx(5.1)
    assert rewards["player_0"] == pytest.approx(-0.1)
    assert rewards["player_1"] == pytest.approx(0.1)
    assert infos["player_0"]["reward_components"]["opponent_drive_change_reward"] == pytest.approx(-0.1)


def test_terminal_info_contains_structured_episode_diagnostics_and_trace():
    env = Ken43ParallelEnv(max_frames=5, diagnostic_trace_frames=3)
    observations, _ = env.reset()
    noop = env.command_index["noop"]
    for _ in range(5):
        observations, _, _, truncated, infos = env.step(
            {"player_0": noop, "player_1": noop}
        )
    assert all(truncated.values())
    diagnostics = infos["player_0"]["episode_diagnostics"]
    assert diagnostics["terminal_reason"] == "max_frames"
    assert diagnostics["frames"] == 5
    assert len(diagnostics["last_trace"]) == 3
    assert diagnostics["frames_since_last_contact"] == 5
    assert diagnostics["activity_at_end"]["any"] is False
    assert diagnostics["timeout_analysis"]["classification"] == "hard_stall"
    assert diagnostics["reward_totals"][0]["event_bonus"] == pytest.approx(-0.5)
    assert diagnostics["reward_totals"][1]["event_bonus"] == pytest.approx(-0.5)
    assert set(diagnostics["reward_totals"][0]) == {"drive", "event_bonus", "total"}


def test_timeout_penalty_applies_equally_to_both_players():
    env = Ken43ParallelEnv(
        max_frames=1,
        timeout_penalty=0.4,
        drive_regen_per_frame=0.0,
    )
    env.reset()
    noop = env.command_index["noop"]
    _, rewards, terminated, truncated, infos = env.step(
        {"player_0": noop, "player_1": noop}
    )

    assert not any(terminated.values())
    assert all(truncated.values())
    assert rewards == {"player_0": pytest.approx(-0.4), "player_1": pytest.approx(-0.4)}
    assert infos["player_0"]["reward_components"]["timeout_penalty"] == pytest.approx(-0.4)
    assert infos["player_1"]["reward_components"]["timeout_penalty"] == pytest.approx(-0.4)


def test_quadratic_time_pressure_is_free_then_normalized():
    env = Ken43ParallelEnv(
        max_frames=600,
        timeout_penalty=0.0,
        time_pressure_start=360,
        time_pressure_total=0.5,
        time_pressure_power=2.0,
    )
    rewards = [env._time_pressure_reward(frame) for frame in range(1, 601)]

    assert all(value == 0.0 for value in rewards[:360])
    assert rewards[360] < 0.0
    assert abs(rewards[-1]) > abs(rewards[360])
    assert sum(rewards) == pytest.approx(-0.5)


def test_gym_facade_treats_penalized_timeout_as_terminal_for_ppo():
    env = Ken43GymEnv(
        max_frames=1,
        timeout_penalty=0.4,
        drive_regen_per_frame=0.0,
    )
    env.reset()
    _, reward, terminated, truncated, info = env.step(env.command_index["noop"])

    assert reward == pytest.approx(-0.4)
    assert terminated is True
    assert truncated is False
    assert info["terminal_reason"] == "max_frames"
    assert info["timeout_treated_as_terminal"] is True


def test_gym_facade_can_bootstrap_timeouts_for_ablation():
    env = Ken43GymEnv(
        max_frames=1,
        timeout_penalty=0.4,
        drive_regen_per_frame=0.0,
        bootstrap_timeouts=True,
    )
    env.reset()
    _, _, terminated, truncated, info = env.step(env.command_index["noop"])

    assert terminated is False
    assert truncated is True
    assert "timeout_treated_as_terminal" not in info


def test_schedule_diagnostics_preserve_claim_noise_and_trigger_frames():
    env = Ken43ParallelEnv(max_frames=30)
    env.reset(seed=9)
    noop = env.command_index["noop"]
    env.step({"player_0": env.command_index["schedule:5:5LP"], "player_1": noop})
    claim = next(item for item in env.diagnostics.timeline if item["type"] == "schedule_claim")
    assert claim["target_delay"] == 5
    assert claim["noise"] in {-2, -1, 0, 1, 2}
    for _ in range(10):
        env.step({"player_0": noop, "player_1": noop})
        triggers = [item for item in env.diagnostics.timeline if item["type"] == "schedule_trigger"]
        if triggers:
            break
    trigger = triggers[0]
    assert trigger["actual_delay"] == claim["actual_delay"]
    assert trigger["frame"] - trigger["submitted_frame"] == claim["actual_delay"]


def test_heuristic_defender_uses_one_persistent_wakeup_choice():
    bot = HeuristicDefenderBot()
    env = Ken43GymEnv(opponent=bot, max_frames=80)
    env.reset(seed=101)
    noop = env.command_index["noop"]

    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(noop)
        if bot.wakeup_decided or terminated or truncated:
            break

    assert bot.wakeup_decided is True
    assert bot.current_family in {"block", "tech", "reversal"}
    assert bot.pending_response is None


def test_heuristic_defender_does_not_mash_during_cancelable_light_pressure():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=4)
    block_event = {
        "type": "Block",
        "actor": 0,
        "target": 1,
        "data": {"action": "5LP"},
    }
    infos = {
        "player_0": {"submitted_command": "immediate:5LP"},
        "player_1": {"events": [block_event], "terminal_reason": None},
    }

    for _ in range(3):
        bot.observe(infos, env, actor=1)

    assert bot.blocked_light_count == 3
    assert bot.pending_response is None
    assert bot.tactics.last_block["assessment"] == "negative_but_cancelable"
    assert list(bot.recent_events)[-1] == "light_pressure_3"


def test_heuristic_defender_reduces_di_bias_after_counter_di():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=5)
    infos = {
        "player_0": {"submitted_command": "immediate:drive_impact"},
        "player_1": {
            "events": [{
                "type": "CounterDI",
                "actor": 0,
                "target": 1,
                "data": {"action": "drive_impact"},
            }],
            "terminal_reason": None,
        },
    }

    bot.observe(infos, env, actor=1)

    assert bot.attacker_counter_di_count == 1
    assert bot.biases["di"] == pytest.approx(0.5)
    assert list(bot.recent_events)[-1] == "attacker_counter_di"


def test_heuristic_defender_counter_di_waits_if_cancel_window_is_closed():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=5)
    own = env.core.combat_state.players[1]
    own.knockdown = False
    own.action_id = "5LP"
    own.action_frame = 4
    own.action_connected = True
    own.contact_action_frame = 4
    bot.pending_response = "immediate:drive_impact"
    observations = env._observations()

    action = bot.act_with_context(
        observations["player_1"], env.command_index,
        np.random.default_rng(5), env, actor=1,
    )

    assert env.commands[action] == "guard:crouching"
    assert bot.pending_response == "immediate:drive_impact"

    own.action_id = None
    own.action_frame = 0
    own.action_connected = False
    own.contact_action_frame = None
    observations = env._observations()
    action = bot.act_with_context(
        observations["player_1"], env.command_index,
        np.random.default_rng(5), env, actor=1,
    )

    assert env.commands[action] == "immediate:drive_impact"
    assert bot.pending_response is None


def test_heuristic_defender_repeated_di_removes_nonresponse_probability():
    bot = HeuristicDefenderBot()
    rng = np.random.default_rng(123)

    bot.opponent_di_seen_count = 3
    for _ in range(20):
        bot.pending_response = None
        bot._choose_di_response(action_frame=1, distance=70.0, rng=rng)
        assert bot.pending_response is not None


def test_heuristic_defender_first_di_always_claims_a_response():
    bot = HeuristicDefenderBot()
    rng = np.random.default_rng(123)

    bot.opponent_di_seen_count = 1
    for _ in range(20):
        bot.pending_response = None
        bot._choose_di_response(action_frame=1, distance=70.0, rng=rng)
        assert bot.pending_response is not None


@pytest.mark.parametrize(
    ("blocked_action", "expected_response"),
    [
        ("shoryuken_L", "immediate:5HP"),
        ("5HK", "immediate:5MP"),
    ],
)
def test_heuristic_defender_punishes_blocked_unsafe_attacks(
    blocked_action, expected_response,
):
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=6)
    infos = {
        "player_0": {"submitted_command": f"immediate:{blocked_action}"},
        "player_1": {
            "events": [{
                "type": "Block",
                "actor": 0,
                "target": 1,
                "data": {"action": blocked_action, "active_index": 1},
            }],
            "terminal_reason": None,
        },
    }

    bot.observe(infos, env, actor=1)

    assert bot.pending_response == expected_response
    assert bot.current_family == "punish"
    assert list(bot.recent_events)[-1] == f"guaranteed_punish:{blocked_action}"


def test_heuristic_defender_waits_out_cancel_window_before_punish():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=6)
    attacker = env.core.combat_state.players[0]
    attacker.action_id = "2HP"
    attacker.action_frame = 8
    infos = {
        "player_0": {"submitted_command": "noop"},
        "player_1": {
            "events": [{
                "type": "Block",
                "actor": 0,
                "target": 1,
                "data": {"action": "2HP", "active_index": 1, "stun": 21},
            }],
            "terminal_reason": None,
        },
    }
    bot.observe(infos, env, actor=1)

    assert bot.pending_response is None
    assert bot.tactics.last_block["assessment"] == "negative_but_cancelable"

    attacker.action_frame = 9
    observation = env._observations()["player_1"]
    action = bot.act_with_context(
        observation, env.command_index, np.random.default_rng(6), env, actor=1,
    )

    assert bot.current_family == "punish"
    assert bot.pending_response == "immediate:2MP"
    assert env.commands[action] == "guard:crouching"


def test_heuristic_defender_punishes_observed_throw_whiff():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=7)
    infos = {
        "player_0": {"submitted_command": "immediate:throw_forward"},
        "player_1": {
            "events": [{
                "type": "ThrowWhiff",
                "actor": 0,
                "target": 1,
                "data": {"reason": "target_unthrowable"},
            }],
            "terminal_reason": None,
        },
    }

    bot.observe(infos, env, actor=1)

    assert bot.pending_response == "immediate:5LP"
    assert bot.current_family == "mash"
    assert list(bot.recent_events)[-1] == "throw_whiff_punish"


def test_heuristic_defender_commits_post_jhp_response():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=8)
    infos = {
        "player_0": {"submitted_command": "immediate:jHP"},
        "player_1": {
            "events": [{
                "type": "Hit",
                "actor": 0,
                "target": 1,
                "data": {"action": "jHP"},
            }],
            "terminal_reason": None,
        },
    }

    bot.observe(infos, env, actor=1)

    assert bot.jhp_hit_count == 1
    assert bot.pending_response == "guard:crouching"
    assert list(bot.recent_events)[-1] == "post_jhp_response_1"


def test_heuristic_defender_commits_post_jhp_block_response():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=9)
    infos = {
        "player_0": {"submitted_command": "immediate:jHP"},
        "player_1": {
            "events": [{
                "type": "Block",
                "actor": 0,
                "target": 1,
                "data": {"action": "jHP", "active_index": 1, "stun": 14},
            }],
            "terminal_reason": None,
        },
    }

    bot.observe(infos, env, actor=1)

    assert bot.jhp_block_count == 1
    assert bot.pending_response == "guard:crouching"
    assert list(bot.recent_events)[-1] == "post_jhp_block_response_1"


def test_heuristic_defender_converts_successful_interrupt_into_escape():
    bot = HeuristicDefenderBot()
    env = Ken43ParallelEnv()
    env.reset(seed=10)
    bot.current_family = "mash"
    infos = {
        "player_0": {"submitted_command": "guard:crouching"},
        "player_1": {
            "events": [{
                "type": "Hit",
                "actor": 1,
                "target": 0,
                "data": {"action": "5LP"},
            }],
            "terminal_reason": None,
        },
    }

    bot.observe(infos, env, actor=1)

    assert bot.pending_response == "immediate:forward_jump"
    assert bot.current_family == "escape_conversion"
    assert list(bot.recent_events)[-1] == "interrupt_escape_conversion"


def test_gym_facade_can_train_player_1_against_player_0_bot():
    offense = HeuristicOffenseBot(forced_macro="safe_jump")
    env = Ken43GymEnv(
        opponent=offense,
        learning_agent="player_1",
        max_frames=80,
    )
    observation, info = env.reset(seed=31)

    assert observation.shape == env.observation_space.shape
    assert info["initial_advantage"] == 43
    assert env.action_masks().shape == (env.action_space.n,)

    _, reward, terminated, truncated, info = env.step(
        env.command_index["guard:crouching"]
    )
    assert np.isfinite(reward)
    assert not terminated and not truncated
    assert info["submitted_command"] == "guard:crouching"
    assert env.parallel._movements[0].kind == "forward_jump"


def test_safe_jump_offense_macro_commits_jhp_once_at_selected_frame():
    offense = HeuristicOffenseBot(forced_macro="safe_jump")
    env = Ken43GymEnv(
        opponent=offense,
        learning_agent="player_1",
        max_frames=100,
    )
    env.reset(seed=32)
    guard = env.command_index["guard:crouching"]
    jhp_starts = 0

    for _ in range(45):
        _, _, terminated, truncated, info = env.step(guard)
        jhp_starts += sum(
            event["type"] == "ActionStarted"
            and event["actor"] == 0
            and event["data"].get("action") == "jHP"
            for event in info["events"]
        )
        if terminated or truncated:
            break

    assert offense.jump_attack_frame in {32, 33, 34}
    assert jhp_starts == 1


def test_grounded_jump_startup_hit_cancels_movement_and_side_switch():
    env = Ken43ParallelEnv(max_frames=100)
    env.reset()
    env.core.combat_state.players[1].knockdown = False
    noop = env.command_index["noop"]
    attacker = env.command_index["immediate:5LP"]
    defender = env.command_index["immediate:forward_jump"]

    saw_hit = False
    saw_cancel = False
    for _ in range(10):
        _, _, terminated, _, infos = env.step({
            "player_0": attacker,
            "player_1": defender,
        })
        attacker = noop
        defender = noop
        events = infos["player_0"]["events"]
        saw_hit |= any(event["type"] == "Hit" and event["target"] == 1 for event in events)
        saw_cancel |= any(event["type"] == "MovementCancelled" for event in events)
        if saw_hit:
            break

    assert saw_hit is True
    assert saw_cancel is True
    assert env._movements[1] is None
    assert env.core.combat_state.players[1].airborne is False
    assert not any(terminated.values())

    for _ in range(50):
        _, _, terminated, truncated, infos = env.step({
            "player_0": noop,
            "player_1": noop,
        })
        if any(terminated.values()) or any(truncated.values()):
            break
    assert infos["player_0"]["terminal_reason"] != "side_switch"


def test_stun_prevents_pending_jump_movement_from_advancing():
    env = Ken43ParallelEnv(max_frames=100)
    env.reset()
    env.core.combat_state.players[1].knockdown = False
    jump = env.command_index["immediate:forward_jump"]
    noop = env.command_index["noop"]
    env.step({"player_0": noop, "player_1": jump})
    assert env._movements[1] is not None
    env._movements[1].frame = 45
    env.core.combat_state.players[1].stun_remaining = 5

    _, _, terminated, _, infos = env.step({"player_0": noop, "player_1": noop})

    assert env._movements[1] is None
    assert env.core.combat_state.players[1].airborne is False
    assert not any(terminated.values())
    assert infos["player_0"]["terminal_reason"] is None


@pytest.mark.parametrize("jump_kind", ["forward_jump", "back_jump"])
def test_heuristic_offense_reacts_once_to_jump(jump_kind):
    offense = HeuristicOffenseBot(forced_macro="stagger")
    env = Ken43ParallelEnv(max_frames=100)
    observations, _ = env.reset(seed=12)
    env.core.combat_state.players[1].knockdown = False
    env.core.combat_state.players[1].airborne = True
    env._movements[1] = MovementState(jump_kind, 5, 46)

    action = offense.act_with_context(
        observations["player_0"], env.command_index,
        np.random.default_rng(1), env, actor=0,
    )

    assert env.commands[action] in {
        "immediate:shoryuken_L", "immediate:2HP",
    }
    assert list(offense.recent_outcomes)[-1] == "anti_jump_claim"
    assert offense.reactive_response is None
    assert offense.opponent_jump_seen_count == 1


def test_pressure_classifier_separates_cancel_threat_and_gap_semantics():
    env = Ken43ParallelEnv()
    env.reset(seed=12)

    light = classify_pressure_state(env, "5LP")
    committed = classify_pressure_state(env, "shoryuken_L")
    overlap = classify_pressure_state(env, "5LP", gap=-1)
    zero_gap = classify_pressure_state(env, "5LP", gap=0)
    open_gap = classify_pressure_state(env, "5LP", gap=4)

    assert light.state is PressureState.NEGATIVE_BUT_CANCELABLE
    assert committed.state is PressureState.TURN_ENDED
    assert overlap.state is PressureState.TRUE_BLOCKSTRING
    assert zero_gap.state is PressureState.ZERO_GAP
    assert open_gap.state is PressureState.CHALLENGEABLE_GAP


def test_challenge_requires_a_real_gap_and_observed_repetition():
    env = Ken43ParallelEnv()
    env.reset(seed=12)

    assert choose_challenge(
        env,
        gap=0,
        distance=70.0,
        opponent_posture="standing",
        repetition=4,
        rng=np.random.default_rng(1),
    ) is None
    assert choose_challenge(
        env,
        gap=3,
        distance=70.0,
        opponent_posture="standing",
        repetition=1,
        rng=np.random.default_rng(1),
    ) is None
    assert choose_challenge(
        env,
        gap=4,
        distance=70.0,
        opponent_posture="standing",
        repetition=4,
        rng=np.random.default_rng(1),
    ) == "5LP"


def test_anti_air_recognition_delay_does_not_become_input_reading():
    offense = HeuristicOffenseBot(forced_macro="stagger")
    env = Ken43ParallelEnv(max_frames=100)
    observations, _ = env.reset(seed=15)
    env.core.combat_state.players[1].knockdown = False
    env.core.combat_state.players[1].airborne = False
    env._movements[1] = MovementState("forward_jump", 1, 46)

    action = offense.act_with_context(
        observations["player_0"], env.command_index,
        np.random.default_rng(1), env, actor=0,
    )

    assert env.commands[action] == "guard:crouching"
    assert offense.opponent_jump_seen_count == 0
    assert not offense.recent_outcomes


@pytest.mark.parametrize("jump_kind", ["forward_jump", "vertical_jump"])
def test_heuristic_defender_uses_shared_anti_air_selection_after_delay(jump_kind):
    defender = HeuristicDefenderBot()
    env = Ken43ParallelEnv(max_frames=100)
    observations, _ = env.reset(seed=16)
    env.core.combat_state.players[1].knockdown = False
    env.core.combat_state.players[0].knockdown = False
    env.core.combat_state.players[0].airborne = True
    env._movements[0] = MovementState(jump_kind, 5, 46)
    observations = env._observations()

    action = defender.act_with_context(
        observations["player_1"], env.command_index,
        np.random.default_rng(1), env, actor=1,
    )

    assert env.commands[action] == "immediate:shoryuken_L"
    assert defender.opponent_jump_seen_count == 1
    assert list(defender.recent_events)[-1] == "anti_air_claim"


def test_heuristic_offense_counter_di_claim_is_bound_to_that_di():
    offense = HeuristicOffenseBot(forced_macro="stagger")
    env = Ken43ParallelEnv(max_frames=100)
    observations, _ = env.reset(seed=13)
    defender = env.core.combat_state.players[1]
    defender.knockdown = False
    defender.action_id = "drive_impact"
    defender.action_frame = 1

    action = offense.act_with_context(
        observations["player_0"], env.command_index,
        np.random.default_rng(2), env, actor=0,
    )

    assert env.commands[action] == "immediate:drive_impact"
    assert list(offense.recent_outcomes)[-1] == "counter_di_claim"
    assert offense.reactive_response is None
    assert offense.opponent_di_seen_count == 1


def test_heuristic_offense_blocks_first_poke_then_di_checks_repetition():
    offense = HeuristicOffenseBot(forced_macro="stagger")
    env = Ken43ParallelEnv(max_frames=100)
    observations, _ = env.reset(seed=14)
    defender = env.core.combat_state.players[1]
    defender.knockdown = False
    defender.action_id = "5MP"
    defender.action_frame = 1

    first = offense.act_with_context(
        observations["player_0"], env.command_index,
        np.random.default_rng(3), env, actor=0,
    )
    assert env.commands[first] == "guard:crouching"
    assert offense.opponent_poke_seen_count == 1

    env.core.combat_state.frame = 20
    defender.action_frame = 1
    second = offense.act_with_context(
        observations["player_0"], env.command_index,
        np.random.default_rng(2), env, actor=0,
    )
    assert env.commands[second] == "immediate:drive_impact"
    assert offense.opponent_poke_seen_count == 2
    assert list(offense.recent_outcomes)[-1] == "anti_poke_claim"
