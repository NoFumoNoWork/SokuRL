"""Run the bounded A/B/C protocol using the existing training/evaluation CLIs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(os.environ['SOKURL_RL_ROOT'])
    report = dict(status='RUNNING', baseline=str(args.baseline.resolve()), stages=[])

    def save():
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(output / 'status.json')

    def run(name, tool, *parameters):
        row = dict(name=name, status='RUNNING', started=time.time())
        command = [sys.executable, '-B', str(root / 'run.py'), tool, *parameters]
        row['command'] = command
        report['stages'].append(row)
        save()
        print(name, 'RUNNING', flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row.update(exit_code=completed.returncode, seconds=time.time()-row['started'],
                   status='PASS' if completed.returncode == 0 else 'FAIL')
        save()
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect its retained log')
        print(name, 'PASS', flush=True)

    save()
    try:
        deadline = time.monotonic() + 7200
        while not (args.baseline / 'result.json').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('baseline did not finish within two hours')
            time.sleep(5)
        baseline = json.loads((args.baseline / 'result.json').read_text())
        if not baseline['success']:
            raise RuntimeError('baseline failed; training prohibited')
        games = baseline['result']['games']
        counts = {name: sum(g['opponent'] == name for g in games)
                  for name in ('rush', 'community_combo', 'pressure', 'bullet_wall')}
        if counts != dict.fromkeys(counts, 64) or len(games) != 256:
            raise RuntimeError(f'baseline matrix incomplete: {counts}')
        report['baseline_counts'] = counts
        save()
        for arm in ('c', 'a', 'b'):
            training = root / 'runs' / f'ppo-controlled-{arm}-20261014'
            run(f'train-{arm}', 'train', '--config-name', f'train_ppo_control_{arm}',
                f'output={training.as_posix()}')
            run(f'archive-{arm}', 'archive_training_sources', str(training))
            for steps in (65536, 131072):
                destination = root / 'runs/evaluation' / f'ppo-controlled-{arm}-{steps}-20261014'
                run(f'interim-{arm}-{steps}', 'benchmark_br', '--config-name',
                    'benchmark_ppo_fixed', f'training_directory={training.as_posix()}',
                    f'checkpoint=checkpoints/updated_{steps}_steps.zip', 'device=cuda:0',
                    'opponent_names=[pressure]', 'benchmark.world_seeds=[918042743,1897077702,244381756,3884668474]',
                    f'output={destination.as_posix()}')
            destination = root / 'runs/evaluation' / f'ppo-controlled-{arm}-final-20261014'
            run(f'final-{arm}', 'benchmark_br', '--config-name', 'benchmark_ppo_fixed',
                f'training_directory={training.as_posix()}', 'checkpoint=final.zip',
                'device=cuda:0', f'output={destination.as_posix()}')
        run('summary', 'summarize_ppo_control', '--runs', str(root / 'runs'),
            '--baseline', str(args.baseline.resolve()),
            '--output', str(output / 'comparison.json'))
        report['status'] = 'PASS'
        report['meaning'] = 'Execution complete; no automatic strength claim or model promotion'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
