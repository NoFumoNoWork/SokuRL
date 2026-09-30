from __future__ import annotations

from ken43 import KenOkiMicrogame, SpacingCalculator, SpacingResult


def print_result(label: str, result: SpacingResult) -> None:
    print(label)
    print(f"initial_distance: {result.initial_distance:.2f}")
    print(f"defender_posture: {result.defender_posture}")
    if result.effective_range is None:
        print("range_check: not_available (contact assumed by scenario)")
    else:
        print(
            f"range_check: {result.initial_distance:.2f} <= {result.effective_range:.2f} "
            f"-> {str(result.range_check).lower()}"
        )
    print(f"contact_or_whiff: {result.contact_or_whiff}")
    print(f"spacing_transition: {result.spacing_transition}")
    print(f"final_distance: {result.final_distance:.2f}")
    print(f"next_move_legal_or_whiff: {result.next_move_legal_or_whiff}")


def main() -> int:
    spacing = SpacingCalculator()

    print("[1] 5LP posture ranges")
    for distance in (141.5, 143.0, 146.0, 147.0):
        for posture in ("standing", "crouching"):
            print_result(f"5LP d={distance:.1f} {posture}", spacing.resolve_block("5LP", distance, posture))

    print("[2] 5MP spacing transition")
    for distance in (70, 90, 100, 130):
        print_result(f"5MP d={distance}", spacing.resolve_block("5MP", distance, "standing"))

    print("[3] two blocked 2MK attacks")
    env = KenOkiMicrogame()
    env.reset(distance=70)
    print_result("2MK #1", env.resolve_blocked_move("2MK", "standing", next_move_id="2MK"))
    print_result("2MK #2", env.resolve_blocked_move("2MK", "standing"))

    print("[4-6] 5HP full/cancelled recovery and next 5LP")
    print_result("5HP full", spacing.resolve_block("5HP", 70, "standing", "full", "5LP"))
    print_result("5HP cancelled", spacing.resolve_block("5HP", 70, "standing", "cancelled", "5LP"))

    print("[7] 5LP -> 2LK -> 5LP posture effect")
    env.reset(distance=70)
    print_result("5LP", env.resolve_blocked_move("5LP", "standing"))
    second = env.resolve_blocked_move("2LK", "crouching")
    print_result("2LK", second)
    print_result(
        "third 5LP vs standing",
        spacing.resolve_block("5LP", second.final_distance, "standing"),
    )
    print_result(
        "third 5LP vs crouching",
        spacing.resolve_block("5LP", second.final_distance, "crouching"),
    )

    print("[8] 2LP -> 2LP -> 5LP")
    env.reset(distance=70)
    print_result("2LP #1", env.resolve_blocked_move("2LP", "standing", next_move_id="2LP"))
    print_result("2LP #2", env.resolve_blocked_move("2LP", "standing", next_move_id="5LP"))
    print_result("5LP #3", env.resolve_blocked_move("5LP", "standing"))

    print("[9] L/M Jinrai root distance floor")
    for move_id in ("jinrai_L", "jinrai_M"):
        for distance in (70, 110, 125, 140):
            print_result(f"{move_id} d={distance}", spacing.resolve_block(move_id, distance, "standing"))

    print("[10] Jinrai LK follow-up fixed increment")
    for distance in (122.5, 127.9, 140):
        print_result(f"kazekama d={distance}", spacing.resolve_block("kazekama", distance, "crouching"))

    print("[11] backwalk then 5MP")
    for frames in range(1, 6):
        distance = spacing.backwalk_distance(128, frames)
        print_result(
            f"backwalk {frames}F then 5MP",
            spacing.resolve_block("5MP", distance, "standing"),
        )

    print("[12] repeated 5MP")
    env.reset(distance=70)
    for index in range(1, 4):
        print_result(f"5MP #{index}", env.resolve_blocked_move("5MP", "standing"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
