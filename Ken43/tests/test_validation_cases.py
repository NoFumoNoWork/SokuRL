from __future__ import annotations

from ken43 import FrameCalculator, KenOkiMicrogame


def test_quick_dash_fastest_emergency_stop_total():
    assert FrameCalculator().quick_dash_fastest_emergency_stop_total() == 27


def test_plus43_5lp_whiff_quick_dash_thunder_last_active_advantage():
    contact = FrameCalculator().plus43_5lp_whiff_qd_thunder_contact()
    assert contact.contact_active_index == 3
    assert contact.advantage == 5


def test_plus42_safe_jump_jhp_inputs():
    calc = FrameCalculator()
    frame32 = calc.safe_jump_jhp(32)
    frame34 = calc.safe_jump_jhp(34)
    frame35 = calc.safe_jump_jhp(35)
    too_early = calc.safe_jump_jhp(28)

    assert frame32.contact is True
    assert frame32.remaining_usable_active == 2
    assert frame32.advantage == 9
    assert frame34.contact is True
    assert frame34.remaining_usable_active == 0
    assert frame34.advantage == 11
    assert frame35.contact is False
    assert too_early.contact is False
    assert too_early.advantage == -4


def test_2mp_first_active_block_advantage():
    assert FrameCalculator().raw_advantage("2MP", "block", 1) == 0


def test_2mp_cancel_m_jinrai_gaps():
    calc = FrameCalculator()
    fastest = calc.cancel_timing("2MP", "jinrai_M", "block", delay=0)
    delayed = calc.cancel_timing("2MP", "jinrai_M", "block", delay=1)

    assert fastest.gap == -1
    assert fastest.overlap == 1
    assert fastest.true_blockstring is True
    assert delayed.gap == 0
    assert delayed.overlap == 0
    assert delayed.true_blockstring is False


def test_5hp_last_active_cancel_h_jinrai_does_not_combo():
    calc = FrameCalculator()
    assert calc.raw_advantage("5HP", "hit", contact_active_index=5) == 7
    assert calc.cancel_gap("5HP", "jinrai_H", "hit") == 0
    assert calc.is_cancel_combo("5HP", "jinrai_H", "hit") is False


def test_l_jinrai_fastest_low_followup_loses_to_4f_normal():
    result = FrameCalculator().jinrai_followup_vs_defender_normal("jinrai_L", "kazekama", 4)
    assert result.attacker_active_frame == 37
    assert result.defender_active_frame == 35
    assert result.result == "DEFENDER_HITS_FIRST"


def test_plus6_wakeup_throw_attempt_whiffs_on_throw_invulnerability():
    attempt = FrameCalculator().plus_throw_attempt(6)
    assert attempt.first_active_relative_to_wakeup == 1
    assert attempt.captures is False
    assert "throw invulnerability" in attempt.reason


def test_late_active_increases_raw_advantage_but_not_cancel_gap():
    calc = FrameCalculator()
    for move_id, move in calc.data.moves.items():
        if move.get("kind") != "normal" or move.get("stun", {}).get("hit") is None:
            continue
        active_len = len(calc.active_frames(move_id))
        if active_len <= 1:
            continue

        first = calc.raw_advantage(move_id, "hit", 1)
        last = calc.raw_advantage(move_id, "hit", active_len)
        assert last - first == active_len - 1

        special = move.get("cancel", {}).get("special", {})
        if special.get("enabled"):
            assert calc.cancel_gap_after_contact(
                move_id, "jinrai_M", "hit", 1, 0
            ) == calc.cancel_gap_after_contact(
                move_id, "jinrai_M", "hit", active_len, 0
            )


def test_plus38_fastest_m_jinrai_light_followup_stand_and_crouch():
    calc = FrameCalculator()
    standing = calc.jinrai_followup_contact_on_wakeup("jinrai_M", "kazekama", 38, "standing")
    crouching = calc.jinrai_followup_contact_on_wakeup("jinrai_M", "kazekama", 38, "crouching")

    assert standing.contact_active_index == 3
    assert standing.outcome == "hit"
    assert standing.advantage == 5
    assert crouching.contact_active_index == 3
    assert crouching.outcome == "block"
    assert crouching.advantage == -3


def test_requested_oki_success_rates():
    calc = FrameCalculator()

    assert calc.quick_dash_thunder_success_rate(43, 0, prefix_move_ids=("5LP",)) == 1.0
    assert calc.quick_dash_thunder_success_rate(26, 0) == 0.0
    rates = {target: calc.quick_dash_thunder_success_rate(36, target) for target in range(1, 11)}
    assert max(rates.values()) == 0.4
    assert [target for target, rate in rates.items() if rate == 0.4] == [5, 6, 7, 8]
    assert calc.delayed_execution_outcomes(1).count(1) / 5 == 0.2


def test_environment_state_contains_required_surface_and_masks_unsupported_actions():
    env = KenOkiMicrogame()
    obs = env.reset(knockdown_advantage=43)
    mask = env.action_mask()

    for key in [
        "attacker_action",
        "attacker_action_frame",
        "defender_action",
        "defender_action_frame",
        "distance",
        "attacker_hp",
        "defender_hp",
        "attacker_drive",
        "defender_drive",
        "attacker_blocking",
        "defender_blocking",
        "attacker_airborne",
        "defender_airborne",
        "attacker_invulnerable",
        "defender_invulnerable",
        "attacker_knockdown",
        "defender_knockdown",
    ]:
        assert key in obs

    assert mask["5MK"] is True
    assert mask["5LP"] is True
