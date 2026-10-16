"""Watch a catalogued REP through the native replay reader, without input injection."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import time
from unittest.mock import patch

from soku_rl.replay.review import reviewed_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--preview-frames", type=int,
                        help="optional bounded preview; omit to watch the complete recording")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path, entry, conditions = reviewed_replay(args.catalog, args.index)
    if args.timeout <= 0:
        raise ValueError("positive startup/response timeout required")
    horizon = entry["frames"]
    if args.preview_frames is not None:
        if not 1 <= args.preview_frames <= horizon:
            raise ValueError("preview must be within the recorded frame range")
        horizon = args.preview_frames
    print(f"Replay: {path}\nFrames: {entry['frames']}\nConditions: {conditions}")
    if args.dry_run:
        return
    from sokurl_platform.replay import launch_replay
    from sokurl_platform.bridge_shared import FIXED_WEATHER_IDS
    environment = {"SOKURL_HEADLESS_RENDER": "0", "SOKURL_FRESH_RESET": "0",
                   "SOKURL_CARDS_DISABLED": str(int(conditions["cards_disabled"]))}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, environment))
        game = launch_replay(path, args.timeout, unlimited=False, observation_mode="diagnostic_state")
        stack.callback(game.close)
        if conditions["fixed_weather"] is not None:
            sequence = game.client.set_fixed_weather(conditions["fixed_weather"])
            acknowledged = game.client.wait_for_ack(sequence, timeout=args.timeout)
            # Changing weather invalidates the original frame-zero checkpoint.
            # Viewing does not use restore/goto; verify weather on advanced frames.
            if acknowledged.ack_seq != sequence or acknowledged.result_name not in {
                    "COMPLETE", "CHECKPOINT_INVALIDATED"}:
                raise RuntimeError(f"weather configuration failed: ack={acknowledged.ack_seq}, "
                                   f"expected={sequence}, result={acknowledged.result_name}")
            print(f"Weather command: {acknowledged.result_name}; restore/goto not used.")
        sequence = game.client.run()
        acknowledged = game.client.wait_for_ack(sequence, timeout=args.timeout)
        if acknowledged.ack_seq != sequence or acknowledged.result_name != "ACCEPTED":
            raise RuntimeError(f"native replay run command failed: {acknowledged.result_name}")
        deadline = time.monotonic() + args.timeout + horizon / 60
        while time.monotonic() < deadline:
            if not game.process.is_running():
                raise RuntimeError("replay exited before recorded frame limit")
            snapshot = game.client.snapshot()
            game.client.drain_frames()
            weather = conditions["fixed_weather"]
            if weather is not None and snapshot.game_frame > 0:
                expected_weather = FIXED_WEATHER_IDS[weather]
                if (snapshot.latest.activeWeather != expected_weather
                        or snapshot.latest.displayedWeather != expected_weather):
                    raise RuntimeError("advanced replay state does not have requested weather")
            if snapshot.game_frame >= horizon:
                print(f"Reached preview/recording frame {horizon}. Playback is not a parity verdict.")
                print(f"Final observed HP: P1={snapshot.latest.p1.hp}, P2={snapshot.latest.p2.hp}")
                return
            time.sleep(.01)
        raise TimeoutError("native replay did not reach its recorded horizon")


if __name__ == "__main__":
    main()
