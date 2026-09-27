from __future__ import annotations

import argparse

from bridge_shared import ACTION_INPUTS, BridgeClient, BridgeUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send one command to SokuRLBridge")
    parser.add_argument("action", choices=[*ACTION_INPUTS, "RELEASE"])
    parser.add_argument("frames", nargs="?", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "RELEASE":
        if args.frames is not None:
            raise SystemExit("RELEASE does not accept a frame count")
    elif args.frames is None:
        raise SystemExit(f"{args.action} requires a frame count")

    try:
        with BridgeClient() as client:
            sequence = client.release() if args.action == "RELEASE" else client.send_action(args.action, args.frames)
            snapshot = client.wait_for_ack(sequence)
    except (BridgeUnavailable, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(f"command_seq={sequence}")
    print(f"ack_seq={snapshot.ack_seq}")
    print(f"status={snapshot.result_name}")
    print(f"frames_remaining={snapshot.frames_remaining}")
    print(f"game_frame={snapshot.game_frame}")
    print(f"in_gameplay={int(snapshot.in_gameplay)}")
    if snapshot.ack_seq != sequence:
        print("error: bridge did not acknowledge the command before timeout")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
