"""Summarize bounded token/MLP smoke without making policy-strength claims."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {'scope':'single-seed2048-transition smoke;not a significant strength or speed study',
              'training':{},'evaluation':{}}
    for name,folder in [('tokens','tokens-smoke-20261013'),('mlp','mlp-tokens-control-20261013')]:
        source = root / 'runs' / folder
        result = json.loads((source/'result.json').read_text())
        timing = json.loads((source/'timing.json').read_text())
        assert result['success']
        steps = result['result']['steps']
        report['training'][name] = {'steps':steps,'environment_frames':steps*3,
            'ppo_update_epochs':sum(r['ppo_n_updates'] for r in timing['rollouts']),
            'e2e_seconds':result['total_seconds'],'e2e_transitions_per_second':steps/result['total_seconds'],
            'rollout_seconds':sum(r['rollout_seconds'] for r in timing['rollouts']),
            'update_seconds':sum(r['update_seconds'] for r in timing['rollouts']),
            'hash':result['result']['final_policy_hash'],'config':str(source/'config.yaml')}
    reference = None
    for name in ['tokens-initial','tokens-final','mlp-tokens-control']:
        source = root/'runs/evaluation'/f'{name}-20261013'
        result = json.loads((source/'result.json').read_text())
        assert result['success']
        games = result['result']['games']
        keys = [(g['world_seed'],g['learner_seat'],g['match'],g['policy_seeds']) for g in games]
        if reference is None:
            reference = keys
        assert keys == reference, 'evaluation conditions differ'
        report['evaluation'][name] = {'games':len(games),'outcomes':[g['outcome'] for g in games],
            'learner_final_public_hp_fraction':statistics.mean(g['public_hp_final_fraction_by_seat'][g['learner_seat']] for g in games),
            'learner_public_hp_loss_fraction':statistics.mean(g['public_hp_loss_fraction_by_seat'][g['learner_seat']] for g in games),
            'opponent_public_hp_loss_fraction':statistics.mean(g['public_hp_loss_fraction_by_seat'][1-g['learner_seat']] for g in games),
            'seconds':result['seconds'],'source':str(source/'result.json')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
