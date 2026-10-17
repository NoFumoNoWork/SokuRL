"""Record a checkpoint battle using existing native REP support; never train."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from soku_rl.replay.review import build_catalog, recording_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final.zip")
    parser.add_argument("--bc-context", type=Path, help="Typed multiscale BC manifest,not a plain PPO checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[918042743])
    parser.add_argument("--opponents", default="pressure")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seat", choices=("left", "right", "both"), default="left")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = recording_command(sys.executable, Path(os.environ["SOKURL_RL_ROOT"]) / "run.py",
                                args.training_directory, args.output, args.checkpoint,
                                args.seeds, args.opponents, args.num_envs, args.device, args.seat)
    if args.bc_context:
        manifest = args.bc_context.resolve(strict=True)
        if manifest.parent != args.training_directory.resolve(strict=True):
            raise ValueError('BC manifest must belong to the specified training directory')
        command += ['candidate.kind=human_multiscale',f'+candidate.path={manifest.as_posix()}']
    print(json.dumps({"command": command,
                      "games_per_opponent": (2 if args.seat == "both" else 1) * len(args.seeds)}, indent=2))
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    catalog = build_catalog(args.output)
    path = args.output / "catalog.json"
    with path.open("x", encoding="utf-8") as target:
        json.dump(catalog, target, indent=2)
    print(f"Saved {len(catalog['games'])} native recordings: {path.resolve()}")
    print("Playback parity NOT TESTED. Use watch_battle with this catalog.")


if __name__ == "__main__":
    main()
