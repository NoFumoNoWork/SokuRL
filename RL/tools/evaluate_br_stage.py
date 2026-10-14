"""Use existing benchmark CLI for paired and cross-opponent stage acceptance."""
import argparse
import json
import os
from pathlib import Path
import statistics
import hashlib
import subprocess
import sys


def evaluation_keys(games):
    keys = sorted((g['world_seed'],g['learner_seat'],json.dumps(g['match'],sort_keys=True),
                   tuple(g['policy_seeds'])) for g in games)
    if len(keys) != len(set(keys)):
        raise RuntimeError('duplicate evaluation trial identity')
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-directory',type=Path)
    args = parser.parse_args()
    training = args.training_directory.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    result = json.loads((training/'result.json').read_text())
    if result.get('success') is not True:
        raise RuntimeError('only completed successful training can enter acceptance')
    report = {'status':'RUNNING','training':str(training),'evaluations':{}}
    def save():
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    cases = [('initial','initial.zip','benchmark_platform_initial',
              '[918042743,1897077702,244381756,3884668474]'),
             ('final','final.zip','benchmark_platform_initial',
              '[918042743,1897077702,244381756,3884668474]'),
             ('cross','final.zip','benchmark_tokens_cross','[918042743,1897077702]')]
    try:
        paired = None
        for name,checkpoint,preset,seeds in cases:
            directory = output/name
            command = [sys.executable,'-B',str(Path(os.environ['SOKURL_RL_ROOT'])/'run.py'),
                'benchmark_br','--config-name',preset,
                f'training_directory={training.as_posix()}',f'checkpoint={checkpoint}',
                'device=cuda:0',f'benchmark.world_seeds={seeds}',f'output={directory.as_posix()}']
            previous = args.evidence_directory/name if args.evidence_directory else None
            if previous and (previous/'result.json').is_file():
                from omegaconf import OmegaConf
                config = OmegaConf.to_container(OmegaConf.load(previous/'config.yaml'),resolve=True)
                if config['training_config_sha256'] != hashlib.sha256((training/'config.yaml').read_bytes()).hexdigest():
                    raise RuntimeError('reused evidence training configuration mismatch')
                if config['checkpoint_sha256'] != hashlib.sha256((training/checkpoint).read_bytes()).hexdigest():
                    raise RuntimeError('reused evidence checkpoint mismatch')
                if config['benchmark']['world_seeds'] != json.loads(seeds):
                    raise RuntimeError('reused evidence seed set mismatch')
                directory = previous.resolve()
            else:
                with (output/f'{name}.log').open('w',encoding='utf-8') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            data = json.loads((directory/'result.json').read_text())
            if not data.get('success'):
                raise RuntimeError(f'{name} did not complete')
            games = data['result']['games']
            # Completion order differs when episodes end at different frames.
            keys = evaluation_keys(games)
            if name == 'initial':
                paired = keys
            elif name == 'final' and keys != paired:
                raise RuntimeError('paired evaluation condition mismatch')
            counts = dict(win=0,loss=0,time_limit=0,double_ko=0)
            opponents = {}
            for game in games:
                outcome = game['outcome']
                if outcome in ('p1_win','p2_win'):
                    outcome = 'win' if outcome==f"p{game['learner_seat']+1}_win" else 'loss'
                counts[outcome] += 1
                opponents.setdefault(game['opponent'],dict(win=0,loss=0,time_limit=0,double_ko=0))[outcome] += 1
            report['evaluations'][name] = {'games':len(games),'counts':counts,'by_opponent':opponents,
                'mean_frames':statistics.mean(g['frames'] for g in games),'seconds':data['seconds'],
                'public_hp_loss':statistics.mean(g['public_hp_loss_fraction_by_seat'][g['learner_seat']] for g in games),
                'opponent_public_hp_loss':statistics.mean(g['public_hp_loss_fraction_by_seat'][1-g['learner_seat']] for g in games),
                'source':str(directory/'result.json')}
            save()
            print(name,counts,flush=True)
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        save()

if __name__ == '__main__':
    main()
