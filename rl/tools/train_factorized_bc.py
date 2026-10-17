"""Explicit bounded semantic BC training; separate from correctness and v1 runs."""
import argparse
from pathlib import Path

from soku_rl.rl.factorized_training import run_training


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,default=Path('rl/config/hierarchical_factorized_training.yaml'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args = p.parse_args()
    run_training(args.config,args.output,args.device)
