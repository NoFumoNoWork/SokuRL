"""Summarize actual continuation and paired evaluation across a BR curriculum."""
import argparse
import json
from pathlib import Path
import statistics

from evaluate_br_stage import evaluation_keys


def load(path):
    return json.loads(path.read_text())


def evaluation(directory):
    data = load(directory/'result.json')
    if not data['success']:
        raise RuntimeError('evaluation incomplete')
    games = data['result']['games']
    groups = {}
    for g in games:
        row = groups.setdefault(g['opponent'],dict(win=0,loss=0,time_limit=0,double_ko=0))
        outcome = g['outcome']
        if outcome in ('p1_win','p2_win'):
            outcome = 'win' if outcome==f"p{g['learner_seat']+1}_win" else 'loss'
        row[outcome] += 1
    return data,games,groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-stage',type=Path,required=True)
    parser.add_argument('--second-stage',type=Path,required=True)
    parser.add_argument('--initial-evaluation',type=Path,required=True)
    parser.add_argument('--final-evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    stages = []
    previous = None
    for directory in [args.first_stage,args.second_stage]:
        result,timing,progress = [load(directory/name) for name in ['result.json','timing.json','progress.json']]
        if not result['success']:
            raise RuntimeError('training incomplete')
        run = result['result']
        if previous:
            if run['initial_policy_hash']!=previous['final_policy_hash']:
                raise RuntimeError('initial parameter identity mismatch')
            kind = run['initial_policy']['kind']
            if kind=='checkpoint' and run['start_steps']!=previous['steps']:
                raise RuntimeError('checkpoint continuation counter mismatch')
            if kind=='weights' and (run['start_steps']!=0 or run['initial_policy']['source_steps']!=previous['steps']):
                raise RuntimeError('declared weights warmstart counter mismatch')
            if kind not in ('weights','checkpoint'):
                raise RuntimeError('unexpected initialization kind')
        stages.append({'directory':str(directory.resolve()),'start_steps':run['start_steps'],
            'additional_steps':run['additional_steps'],'steps':run['steps'],
            'initialization':run['initial_policy']['kind'],
            'final_hash':run['final_policy_hash'],'e2e_seconds':result['total_seconds'],
            'rollout_seconds':sum(r['rollout_seconds'] for r in timing['rollouts']),
            'update_seconds':sum(r['update_seconds'] for r in timing['rollouts']),
            'episode_counts':progress['episode_summary']['overall']['counts'],
            'by_opponent':progress['episode_summary']['groups']['opponent']})
        previous = run
    a,ag,ac = evaluation(args.initial_evaluation)
    b,bg,bc = evaluation(args.final_evaluation)
    if evaluation_keys(ag)!=evaluation_keys(bg):
        raise RuntimeError('paired conditions mismatch')
    if a['runtime']['fingerprints']['game_id']!=b['runtime']['fingerprints']['game_id']:
        raise RuntimeError('original runtime identity mismatch')
    report = {'status':'PASS','stages':stages,'paired_evaluation':{},
        'scope':'same-task before/after mixture;small seed set,not statistical superiority proof'}
    for name,games,counts in [('before',ag,ac),('after',bg,bc)]:
        report['paired_evaluation'][name] = {'games':len(games),'by_opponent':counts,
            'total_wins':sum(r['win'] for r in counts.values()),
            'mean_frames':statistics.mean(g['frames'] for g in games),
            'learner_public_hp_loss':statistics.mean(g['public_hp_loss_fraction_by_seat'][g['learner_seat']] for g in games)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['paired_evaluation'],indent=2))

if __name__=='__main__':
    main()
