"""Create a separately named world-coordinate PPO artifact; never overwrite it."""
import argparse
import json
from pathlib import Path
from soku_rl.rl.world_migration import migrate_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    if not output.is_relative_to(root/'rl/runs'):
        raise ValueError('migration output must remain under workspace rl/runs')
    print(json.dumps(migrate_world(args.checkpoint,args.config,output),indent=2))


if __name__ == '__main__':
    main()
