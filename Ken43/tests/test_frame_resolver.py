from __future__ import annotations

import pytest

from ken43 import (
    CombatState,
    FighterState,
    FrameInput,
    FrameResolver,
    KenOkiMicrogame,
    MovementProfile,
)


def event_types(events):
    return [event.type for event in events]


def advance_to_action_frame(
    resolver: FrameResolver,
    state: CombatState,
    action_id: str,
    target_frame: int,
    defender_input: FrameInput | None = None,
):
    state, events = resolver.resolve_frame(
        state,
        (FrameInput(action=action_id), defender_input or FrameInput()),
    )
    all_events = list(events)
    while state.players[0].action_id is not None and state.players[0].action_frame < target_frame:
        state, events = resolver.resolve_frame(
            state,
            (FrameInput(), defender_input or FrameInput()),
        )
        all_events.extend(events)
    return state, all_events


def test_normal_advances_and_blocks_on_its_first_active_frame():
    resolver = FrameResolver()
    state = CombatState(distance=70)
    state, events = advance_to_action_frame(
        resolver, state, "5LP", 4, FrameInput(guard="standing")
    )

    assert state.frame == 4
    assert state.players[0].action_frame == 4
    assert "Block" in event_types(events)
    assert state.players[1].hp == 10000
    assert state.players[1].stun_remaining == 8
    assert state.distance == pytest.approx(103.34)


def test_unguarded_normal_hits_and_applies_damage_and_stun():
    resolver = FrameResolver()
    state, events = advance_to_action_frame(
        resolver, CombatState(distance=70), "5LP", 4
    )

    assert "Hit" in event_types(events)
    assert state.players[1].hp == 9700
    assert state.players[1].stun_remaining == 13


def test_range_is_checked_before_contact_and_spacing():
    resolver = FrameResolver()
    state, events = advance_to_action_frame(
        resolver,
        CombatState(distance=150),
        "5LP",
        4,
        FrameInput(guard="standing"),
    )

    assert "RangeMiss" in event_types(events)
    assert "Block" not in event_types(events)
    assert state.distance == pytest.approx(150)

    while state.players[0].action_id is not None:
        state, later_events = resolver.resolve_frame(
            state, (FrameInput(), FrameInput(guard="standing"))
        )
        events.extend(later_events)
    assert "ActiveFrame" in event_types(events)
    assert "Whiff" in event_types(events)


def test_low_requires_crouching_guard():
    resolver = FrameResolver()
    standing, standing_events = advance_to_action_frame(
        resolver,
        CombatState(distance=70),
        "2LK",
        5,
        FrameInput(guard="standing"),
    )
    crouching, crouching_events = advance_to_action_frame(
        resolver,
        CombatState(distance=70),
        "2LK",
        5,
        FrameInput(guard="crouching"),
    )

    assert "Hit" in event_types(standing_events)
    assert standing.players[1].hp == 9800
    assert "Block" in event_types(crouching_events)
    assert crouching.players[1].hp == 10000


def test_simultaneous_strikes_trade_from_one_snapshot():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5LP", action_frame=3),
            FighterState(action_id="5LP", action_frame=3),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert event_types(events).count("Hit") == 2
    assert state.players[0].hp == 9700
    assert state.players[1].hp == 9700


def test_guard_input_cannot_block_while_attacking_or_in_hitstun():
    resolver = FrameResolver()
    attacking = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5LP", action_frame=3),
            FighterState(action_id="5LP", action_frame=1),
        ],
    )
    attacking, attacking_events = resolver.resolve_frame(
        attacking,
        (FrameInput(), FrameInput(guard="standing")),
    )
    assert "Hit" in event_types(attacking_events)
    assert attacking.players[1].hp == 9700

    hitstun = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5LP", action_frame=3),
            FighterState(stun_remaining=5, stun_kind="hit"),
        ],
    )
    hitstun, hitstun_events = resolver.resolve_frame(
        hitstun,
        (FrameInput(), FrameInput(guard="standing")),
    )
    assert "Hit" in event_types(hitstun_events)
    assert hitstun.players[1].hp == 9700


def test_strike_beats_throw_on_the_same_frame():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5LP", action_frame=3),
            FighterState(action_id="throw_forward", action_frame=4),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert "ThrowLostToStrike" in event_types(events)
    assert state.players[0].hp == 10000
    assert state.players[1].hp == 9700


def test_full_invulnerability_is_evaluated_before_simultaneous_commit():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5LP", action_frame=3),
            FighterState(action_id="shoryuken_OD", action_frame=5),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert "Invulnerable" in event_types(events)
    assert state.players[0].hp == 8400
    assert state.players[1].hp == 10000


def test_drive_impact_armor_absorbs_a_strike():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="5LP", action_frame=3),
            FighterState(
                action_id="drive_impact",
                action_frame=0,
                armor_hits_remaining=2,
            ),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert "ArmorAbsorb" in event_types(events)
    assert state.players[1].armor_hits_remaining == 1
    assert state.players[1].hp == 10000


def test_throw_failed_first_capture_does_not_reacquire():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="throw_forward", action_frame=4),
            FighterState(throw_invul_remaining=1),
        ],
    )
    state, first_events = resolver.resolve_frame(state)
    state, second_events = resolver.resolve_frame(state)

    assert "ThrowWhiff" in event_types(first_events)
    assert "Throw" not in event_types(second_events)
    assert state.players[1].hp == 10000


def test_contact_cancel_queues_child_for_the_next_global_frame():
    resolver = FrameResolver()
    state = CombatState(distance=70)
    state, _ = advance_to_action_frame(
        resolver, state, "2MP", 5, FrameInput(guard="standing")
    )
    state, events = resolver.resolve_frame(
        state,
        (
            FrameInput(cancel_to="jinrai_M"),
            FrameInput(guard="standing"),
        ),
    )

    assert "Block" in event_types(events)
    assert "CancelQueued" in event_types(events)
    assert state.players[0].action_id is None
    assert state.players[0].queued_action == "jinrai_M"

    state, next_events = resolver.resolve_frame(state)
    assert "ActionStarted" in event_types(next_events)
    assert state.players[0].action_id == "jinrai_M"
    assert state.players[0].action_frame == 1


def test_delayed_special_cancel_uses_recorded_contact_frame():
    resolver = FrameResolver()
    state, _ = advance_to_action_frame(
        resolver,
        CombatState(distance=70),
        "2MP",
        6,
        FrameInput(guard="standing"),
    )
    assert state.players[0].contact_action_frame == 6

    state, events = resolver.resolve_frame(
        state,
        (FrameInput(cancel_to="jinrai_M"), FrameInput(guard="standing")),
    )
    assert "CancelQueued" in event_types(events)
    assert state.players[0].queued_action == "jinrai_M"


def test_light_chain_delay_is_limited_to_remaining_active_frames():
    resolver = FrameResolver()
    state, _ = advance_to_action_frame(
        resolver,
        CombatState(distance=70),
        "5LP",
        4,
        FrameInput(guard="standing"),
    )
    state, events = resolver.resolve_frame(
        state,
        (FrameInput(cancel_to="2LP"), FrameInput(guard="standing")),
    )
    assert "CancelQueued" in event_types(events)

    late, _ = advance_to_action_frame(
        resolver,
        CombatState(distance=70),
        "5LP",
        4,
        FrameInput(guard="standing"),
    )
    late, _ = resolver.resolve_frame(late)
    late, _ = resolver.resolve_frame(late)
    late, late_events = resolver.resolve_frame(
        late,
        (FrameInput(cancel_to="2LP"), FrameInput()),
    )
    assert "CancelQueued" not in event_types(late_events)
    assert "InputRejected" in event_types(late_events)


def test_empirical_branch_timing_controls_child_start_frame():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[FighterState(action_id="jinrai_M", action_frame=31), FighterState()],
    )
    state, events = resolver.resolve_frame(
        state,
        (FrameInput(branch_to="kazekama"), FrameInput()),
    )

    assert "BranchStarted" in event_types(events)
    assert state.players[0].action_id == "kazekama"
    assert state.players[0].action_frame == 2


def test_quick_dash_empirical_override_can_queue_branch_until_next_frame():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[FighterState(action_id="quick_dash", action_frame=9), FighterState()],
    )
    state, events = resolver.resolve_frame(
        state,
        (FrameInput(branch_to="quick_dash_thunder_kick"), FrameInput()),
    )

    assert "BranchQueued" in event_types(events)
    assert state.players[0].queued_action == "quick_dash_thunder_kick"
    state, _ = resolver.resolve_frame(state)
    assert state.players[0].action_frame == 1


def test_5hp_full_and_cancelled_recovery_split_at_empirical_endpoints():
    resolver = FrameResolver()

    full, _ = advance_to_action_frame(
        resolver,
        CombatState(distance=70),
        "5HP",
        10,
        FrameInput(guard="standing"),
    )
    assert full.distance == pytest.approx(133.0)
    while full.players[0].action_id is not None:
        full, _ = resolver.resolve_frame(full)
    assert full.distance == pytest.approx(157.52)

    cancelled = CombatState(distance=70)
    cancelled, _ = advance_to_action_frame(
        resolver,
        cancelled,
        "5HP",
        9,
        FrameInput(guard="standing"),
    )
    cancelled, events = resolver.resolve_frame(
        cancelled,
        (
            FrameInput(cancel_to="jinrai_L"),
            FrameInput(guard="standing"),
        ),
    )
    assert "CancelQueued" in event_types(events)
    assert cancelled.distance == pytest.approx(133.0)


def test_known_per_frame_movement_profile_is_applied_before_contact():
    resolver = FrameResolver(
        movement_profile=MovementProfile({"5LP": {1: 2.5}})
    )
    state, events = resolver.resolve_frame(
        CombatState(distance=70),
        (FrameInput(action="5LP"), FrameInput()),
    )

    assert state.distance == pytest.approx(72.5)
    movement = [event for event in events if event.type == "Movement"]
    assert movement[0].data["mode"] == "per_frame"


def test_stun_expiry_grants_post_stun_throw_invulnerability():
    resolver = FrameResolver()
    state = CombatState(
        players=[FighterState(stun_remaining=1, stun_kind="block", blocking=True), FighterState()]
    )
    state, events = resolver.resolve_frame(state)

    assert "StunEnded" in event_types(events)
    assert state.players[0].stun_remaining == 0
    assert state.players[0].throw_invul_remaining == 2


def test_resolve_frame_does_not_mutate_its_input_state():
    resolver = FrameResolver()
    original = CombatState(
        distance=70,
        players=[FighterState(action_id="5LP", action_frame=3), FighterState()],
    )
    resolved, _ = resolver.resolve_frame(original)

    assert original.frame == 0
    assert original.players[0].action_frame == 3
    assert original.players[1].hp == 10000
    assert resolved.frame == 1
    assert resolved.players[0].action_frame == 4


def test_v0_masks_parry_and_drive_rush_but_keeps_drive_impact():
    mask = KenOkiMicrogame().action_mask()

    assert mask["drive_parry"] is False
    assert mask["cancel_drive_rush"] is False
    assert mask["drive_impact"] is True


@pytest.mark.parametrize(
    "action_id",
    [
        "kazekama",
        "gorai",
        "senka",
        "quick_dash_emergency_stop",
        "quick_dash_thunder_kick",
        "quick_dash_forward_step_kick",
        "chin_buster_2",
        "triple_flash_2",
        "triple_flash_3",
    ],
)
def test_contextual_followups_are_masked_and_rejected_as_direct_actions(action_id):
    env = KenOkiMicrogame()
    assert env.action_mask()[action_id] is False

    state, events = env.resolver.resolve_frame(
        CombatState(),
        (FrameInput(action=action_id), FrameInput()),
    )
    rejection = [event for event in events if event.type == "InputRejected"]
    assert rejection[0].data["reason"] == "not_directly_selectable"
    assert state.players[0].action_id is None


def test_target_combo_followup_remains_available_through_its_parent():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[FighterState(action_id="5MP", action_frame=4), FighterState()],
    )
    state, events = resolver.resolve_frame(
        state,
        (
            FrameInput(cancel_to="chin_buster_2"),
            FrameInput(guard="standing"),
        ),
    )

    assert "Block" in event_types(events)
    assert "CancelQueued" in event_types(events)
    assert state.players[0].queued_action == "chin_buster_2"


def test_drive_resource_is_inert_while_full_drive_system_is_deferred():
    resolver = FrameResolver()
    state = CombatState(players=[FighterState(drive=0), FighterState()])
    state, events = resolver.resolve_frame(
        state, (FrameInput(action="shoryuken_OD"), FrameInput())
    )

    assert "ActionStarted" in event_types(events)
    assert state.players[0].action_id == "shoryuken_OD"
    assert state.players[0].drive == 0


def test_simplified_multi_hit_counts_come_from_the_data_contract():
    resolver = FrameResolver()

    assert resolver.calc.simplified_hit_count("tatsu_L") == 2
    assert resolver.calc.simplified_hit_count("shoryuken_M") == 2
    assert resolver.calc.simplified_hit_count("shoryuken_H") == 3
    assert resolver.calc.simplified_hit_count("quick_dash_shoryuken") == 6
    assert resolver.calc.simplified_hit_count("5LP") == 1


def test_two_hit_move_is_fully_absorbed_by_two_hit_di_armor():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="shoryuken_M", action_frame=5),
            FighterState(
                action_id="drive_impact",
                action_frame=0,
                armor_hits_remaining=2,
            ),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert "ArmorAbsorb" in event_types(events)
    assert "Hit" not in event_types(events)
    assert state.players[1].armor_hits_remaining == 0
    assert state.players[1].hp == 10000


def test_three_hit_move_breaks_two_hit_di_armor_and_uses_aggregate_result():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="shoryuken_H", action_frame=6),
            FighterState(
                action_id="drive_impact",
                action_frame=0,
                armor_hits_remaining=2,
            ),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert "ArmorBreak" in event_types(events)
    assert "Hit" in event_types(events)
    assert state.players[1].hp == 8600
    assert state.players[1].knockdown is True


def test_drive_impact_bypasses_drive_impact_armor_as_counter_di():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="drive_impact", action_frame=25),
            FighterState(
                action_id="drive_impact",
                action_frame=0,
                armor_hits_remaining=2,
            ),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert "CounterDI" in event_types(events)
    assert "Hit" in event_types(events)
    assert "ArmorAbsorb" not in event_types(events)
    assert state.players[1].hp == 9200


def test_throw_bypasses_drive_impact_armor():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[
            FighterState(action_id="throw_forward", action_frame=4),
            FighterState(
                action_id="drive_impact",
                action_frame=0,
                armor_hits_remaining=2,
            ),
        ],
    )
    state, events = resolver.resolve_frame(state)

    assert "Throw" in event_types(events)
    assert "ArmorAbsorb" not in event_types(events)
    assert state.players[1].hp == 8800
    assert state.players[1].knockdown is True


def test_v0_wakeup_is_fixed_to_frame_43_and_allows_frame_1_reversal():
    env = KenOkiMicrogame()
    env.reset()

    for _ in range(42):
        env.resolve_frame()
    assert env.combat_state.players[1].knockdown is True

    state, events = env.resolve_frame(
        defender_input=FrameInput(action="shoryuken_OD")
    )
    assert state.frame == 43
    assert "ActionStarted" in event_types(events)
    assert state.players[1].action_id == "shoryuken_OD"
    assert state.players[1].action_frame == 1
    assert env.state.defender_invulnerable is True

    with pytest.raises(ValueError, match="fixed \\+43"):
        env.reset(knockdown_advantage=42)


def test_unmodeled_quick_dash_shoryuken_branch_is_rejected_without_clearing_parent():
    resolver = FrameResolver()
    state = CombatState(
        distance=70,
        players=[FighterState(action_id="quick_dash", action_frame=11), FighterState()],
    )
    state, events = resolver.resolve_frame(
        state,
        (FrameInput(branch_to="quick_dash_shoryuken"), FrameInput()),
    )

    assert "InputRejected" in event_types(events)
    assert state.players[0].action_id == "quick_dash"
    assert state.players[0].action_frame == 12
