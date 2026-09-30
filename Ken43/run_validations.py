from __future__ import annotations

from ken43.frame_math import FrameCalculator


def main() -> int:
    calc = FrameCalculator()
    fastest_2mp = calc.cancel_timing("2MP", "jinrai_M", "block", 0)
    delayed_2mp = calc.cancel_timing("2MP", "jinrai_M", "block", 1)
    rows = [
        ("quick_dash_fastest_emergency_stop", calc.quick_dash_fastest_emergency_stop_total()),
        ("plus43_5LP_whiff_QD_Thunder_hit_adv", calc.plus43_5lp_whiff_qd_thunder_contact().advantage),
        ("plus42_jHP_input32", calc.safe_jump_jhp(32).advantage),
        ("plus42_jHP_input34", calc.safe_jump_jhp(34).advantage),
        ("plus42_jHP_input35", calc.safe_jump_jhp(35).reason),
        ("plus42_jHP_too_early", calc.safe_jump_jhp(28).advantage),
        ("2MP_first_active_block_adv", calc.raw_advantage("2MP", "block", 1)),
        ("2MP_cancel_M_Jinrai_fastest", fastest_2mp),
        ("2MP_cancel_M_Jinrai_delay_1F", delayed_2mp),
        ("5HP_last_active_cancel_H_Jinrai_combo", calc.is_cancel_combo("5HP", "jinrai_H", "hit")),
        (
            "L_Jinrai_low_followup_vs_4F",
            calc.jinrai_followup_vs_defender_normal("jinrai_L", "kazekama", 4).result,
        ),
        ("plus6_throw_captures", calc.plus_throw_attempt(6).captures),
        (
            "plus38_M_Jinrai_low_vs_stand",
            calc.jinrai_followup_contact_on_wakeup("jinrai_M", "kazekama", 38, "standing").advantage,
        ),
        (
            "plus38_M_Jinrai_low_vs_crouch",
            calc.jinrai_followup_contact_on_wakeup("jinrai_M", "kazekama", 38, "crouching").advantage,
        ),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
