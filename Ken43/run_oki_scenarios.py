from __future__ import annotations

from ken43 import FrameCalculator


def _thunder_outcomes(
    calc: FrameCalculator,
    knockdown_advantage: int,
    target_delay: int,
    prefix_move_ids: tuple[str, ...] = (),
) -> list[str]:
    rows: list[str] = []
    for error, actual_delay in zip((-2, -1, 0, 1, 2), calc.delayed_execution_outcomes(target_delay)):
        contact = calc.quick_dash_thunder_contact_for_plus(
            knockdown_advantage,
            actual_delay,
            prefix_move_ids,
        )
        result = "whiff" if contact is None else f"active {contact.contact_active_index}, {contact.advantage:+d}F"
        rows.append(f"epsilon={error:+d}: delay={actual_delay}F -> {result}")
    return rows


def main() -> int:
    calc = FrameCalculator()
    scenarios = [
        (
            "+43F, 5LP whiff -> Quick Dash overhead, >=+4F",
            calc.quick_dash_thunder_success_rate(43, 0, prefix_move_ids=("5LP",)),
            _thunder_outcomes(calc, 43, 0, ("5LP",)),
        ),
        (
            "+26F, Quick Dash overhead, >=+4F",
            calc.quick_dash_thunder_success_rate(26, 0),
            _thunder_outcomes(calc, 26, 0),
        ),
        (
            "+36F, target delay 6F Quick Dash overhead, >=+4F",
            calc.quick_dash_thunder_success_rate(36, 6),
            _thunder_outcomes(calc, 36, 6),
        ),
    ]

    for name, rate, outcomes in scenarios:
        print(f"{name}: {rate:.0%}")
        for outcome in outcomes:
            print(f"  {outcome}")

    safe_jump_outcomes = calc.delayed_execution_outcomes(1)
    safe_jump_rate = safe_jump_outcomes.count(1) / len(safe_jump_outcomes)
    print(f"+43F, target delay 1F to +42F safe jump: {safe_jump_rate:.0%}")
    print(f"  actual delays for epsilon -2..+2: {safe_jump_outcomes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
