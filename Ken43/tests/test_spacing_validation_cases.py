from __future__ import annotations

import pytest

from ken43 import KenOkiMicrogame, SpacingCalculator


@pytest.fixture
def spacing() -> SpacingCalculator:
    return SpacingCalculator()


def test_01_5lp_posture_ranges(spacing: SpacingCalculator):
    expected = {
        141.5: (False, True),
        143.0: (False, True),
        146.0: (False, True),
        147.0: (False, True),
        148.5: (False, False),
    }
    for distance, (standing, crouching) in expected.items():
        assert spacing.in_range("5LP", distance, "standing") is standing
        assert spacing.in_range("5LP", distance, "crouching") is crouching


def test_5lp_geometry_uses_hurtbox_half_widths(spacing: SpacingCalculator):
    assert spacing.attack_reach("5LP") == pytest.approx(101.40)
    assert spacing.effective_range("5LP", "standing") == pytest.approx(141.20)
    assert spacing.effective_range("5LP", "crouching") == pytest.approx(148.40)
    assert (
        spacing.effective_range("5LP", "crouching")
        - spacing.effective_range("5LP", "standing")
    ) == pytest.approx(7.20)


def test_02_5mp_spacing_transition(spacing: SpacingCalculator):
    expected = {70: 128.93, 90: 132.5, 100: 142.5, 130: 172.5}
    for distance, final in expected.items():
        result = spacing.resolve_block("5MP", distance, "standing")
        assert result.contact_or_whiff == "block"
        assert result.final_distance == pytest.approx(final)


def test_03_two_blocked_2mk_attacks():
    env = KenOkiMicrogame()
    env.reset(distance=70)
    first = env.resolve_blocked_move("2MK", "standing", next_move_id="2MK")
    second = env.resolve_blocked_move("2MK", "standing")
    assert first.final_distance == pytest.approx(156.09)
    assert first.next_move_legal_or_whiff == "legal"
    assert second.contact_or_whiff == "block"
    assert second.final_distance == pytest.approx(209.22)


def test_04_5hp_full_vs_cancelled_recovery(spacing: SpacingCalculator):
    full = spacing.resolve_block("5HP", 70, "standing", "full")
    cancelled = spacing.resolve_block("5HP", 70, "standing", "cancelled")
    assert full.final_distance == pytest.approx(157.52)
    assert cancelled.final_distance == pytest.approx(133.0)
    assert "d + 9.89" in full.spacing_transition
    assert "d + 9.89" in cancelled.spacing_transition


def test_05_5hp_full_recovery_then_5lp_whiffs(spacing: SpacingCalculator):
    result = spacing.resolve_block(
        "5HP", 70, "standing", "non_hold2", next_move_id="5LP"
    )
    assert result.final_distance == pytest.approx(157.52)
    assert result.next_move_legal_or_whiff == "whiff"


def test_06_5hp_cancelled_recovery_then_5lp_connects(spacing: SpacingCalculator):
    result = spacing.resolve_block(
        "5HP", 70, "standing", "cancelled", next_move_id="5LP"
    )
    assert result.final_distance == pytest.approx(133.0)
    assert result.next_move_legal_or_whiff == "legal"


def test_07_5lp_2lk_5lp_crouching_hurtbox_effect():
    env = KenOkiMicrogame()
    env.reset(distance=70)
    env.resolve_blocked_move("5LP", "standing")
    second = env.resolve_blocked_move("2LK", "crouching")
    assert second.final_distance == pytest.approx(141.84)
    assert env.spacing.in_range("5LP", second.final_distance, "standing") is False
    assert env.spacing.in_range("5LP", second.final_distance, "crouching") is True


def test_08_2lp_2lp_5lp_sequence():
    env = KenOkiMicrogame()
    env.reset(distance=70)
    first = env.resolve_blocked_move("2LP", "standing", next_move_id="2LP")
    second = env.resolve_blocked_move("2LP", "standing", next_move_id="5LP")
    third = env.resolve_blocked_move("5LP", "standing")
    assert first.next_move_legal_or_whiff == "legal"
    assert second.next_move_legal_or_whiff == "legal"
    assert third.contact_or_whiff == "block"
    assert third.final_distance == pytest.approx(169.34)


def test_09_jinrai_root_distance_floor(spacing: SpacingCalculator):
    expected = {70: 122.5, 110: 122.5, 125: 125.0, 140: 140.0}
    for move_id in ("jinrai_L", "jinrai_M"):
        for distance, final in expected.items():
            result = spacing.resolve_block(move_id, distance, "standing")
            assert result.final_distance == pytest.approx(final)


def test_10_jinrai_lk_followup_fixed_increment(spacing: SpacingCalculator):
    expected = {122.5: 159.17, 127.9: 164.57, 140: 176.67}
    for distance, final in expected.items():
        result = spacing.resolve_block("kazekama", distance, "crouching")
        assert result.final_distance == pytest.approx(final)


def test_11_backwalk_changes_5mp_range(spacing: SpacingCalculator):
    expected = {1: True, 2: False, 3: False, 4: False, 5: False}
    expected_distances = {1: 128.8, 2: 132.0, 3: 135.2, 4: 138.4, 5: 141.6}
    for frames, contact in expected.items():
        distance = spacing.backwalk_distance(128, frames)
        assert distance == pytest.approx(expected_distances[frames])
        assert spacing.in_range("5MP", distance, "standing") is contact


def test_12_repeated_5mp_naturally_leaves_range():
    env = KenOkiMicrogame()
    env.reset(distance=70)
    first = env.resolve_blocked_move("5MP", "standing")
    second = env.resolve_blocked_move("5MP", "standing")
    third = env.resolve_blocked_move("5MP", "standing")
    assert first.contact_or_whiff == "block"
    assert first.final_distance == pytest.approx(128.93)
    assert second.contact_or_whiff == "block"
    assert second.final_distance == pytest.approx(171.43)
    assert third.contact_or_whiff == "whiff"
    assert third.final_distance == pytest.approx(171.43)
