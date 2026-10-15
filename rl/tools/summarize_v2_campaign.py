"""Summarize preserved campaign evidence without running environments or models."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def wins(games):
    return sum(g['outcome']==('p1_win' if g['learner_seat']==0 else 'p2_win') for g in games)


def summary(games):
    count=len(games)
    success=wins(games)
    p=success/count
    z=1.959963984540054
    center=(p+z*z/(2*count))/(1+z*z/count)
    radius=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    rewards=[g['combat_reward_summary_by_seat'][g['learner_seat']] for g in games]
    dealt=sum(r['damage_dealt'] for r in rewards)
    taken=sum(r['damage_taken'] for r in rewards)
    return dict(games=count,wins=success,win_rate=p,descriptive_wilson95=[center-radius,center+radius],
                damage_balance=mean(r['damage_dealt']-r['damage_taken'] for r in rewards),
                damage_exchange_ratio=dealt/taken if taken else None,
                mean_frames=mean(g['frames'] for g in games),
                mean_normalized_return=mean(g['returns'][g['learner_seat']] for g in games),
                extensions=mean(r['combo_extensions'] for r in rewards),
                unknown_lineage_hits=sum(r['unknown_lineage_hits'] for r in rewards))


def paired(reference, candidate):
    def key(g):return g['opponent'],g['learner_seat'],g['world_seed']
    before={key(g):g for g in reference}
    changes=[]
    for game in candidate:
        old=before[key(game)]
        if game['policy_seeds']!=old['policy_seeds'] or game['match']!=old['match']:
            raise ValueError('evaluation seed/match mismatch; comparison is not paired')
        seat=game['learner_seat']
        if game['strategy_ids'][1-seat]!=old['strategy_ids'][1-seat]:
            raise ValueError('opponent fingerprint differs')
        changes.append(wins([game])-wins([old]))
    return dict(games=len(changes),gained=sum(c==1 for c in changes),
                lost=sum(c==-1 for c in changes),unchanged=sum(c==0 for c in changes),
                meaning='matched world/actor seeds, character setup and opponent; not a significance test')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=dict(evaluations={},training={},limitations=[
        'Development selection uses four games per opponent; not evidence of significance.',
        'Wilson intervals treat games independently; paired seats share world seeds.',
        'Restarts, timing capacity migration and seat changes are declared confounds.'])
    datasets={}
    for path in sorted(args.campaign.glob('*/result.json')):
        data=json.loads(path.read_text())
        if not data.get('success'):continue
        games=data.get('result',{}).get('games')
        if not isinstance(games,list):continue
        datasets[path.parent.name]=games
        report['evaluations'][path.parent.name]={name:summary([g for g in games if g['opponent']==name])
            for name in sorted({g['opponent'] for g in games})}
        for foe,entry in report['evaluations'][path.parent.name].items():
            entry['by_seat']={str(seat):summary([g for g in games if g['opponent']==foe and g['learner_seat']==seat])
                              for seat in (0,1) if any(g['opponent']==foe and g['learner_seat']==seat for g in games)}
    report['paired']={name:paired(datasets['screen-65k'],games)
                      for name,games in datasets.items() if name.startswith('eval-')}
    references=[name for name in datasets if name.startswith('heldout-reference')]
    selected=[name for name in datasets if name.startswith('heldout-selected')]
    if references and selected:
        reference,candidate=datasets[references[-1]],datasets[selected[-1]]
        report['paired_heldout']={foe:paired([g for g in reference if g['opponent']==foe],
            [g for g in candidate if g['opponent']==foe]) for foe in {g['opponent'] for g in candidate}}
    initials=[name for name in datasets if name.startswith('heldout-initial')]
    if initials and selected:
        initial=datasets[initials[-1]]
        foes={g['opponent'] for g in initial}
        report['paired_vs_untrained']=paired(initial,
            [g for g in datasets[selected[-1]] if g['opponent'] in foes])
    keys=('approx_kl','clip_fraction','entropy_loss','explained_variance','value_loss')
    for path in sorted(args.campaign.glob('train-*/scalars/progress.csv')):
        rows=list(csv.DictReader(path.open()))
        metrics={}
        for key in keys:
            values=[float(r['train/'+key]) for r in rows if r.get('train/'+key)]
            values=[v for v in values if math.isfinite(v)]
            if values:
                metrics[key]=dict(n=len(values),mean=mean(values),median=median(values),
                                  minimum=min(values),maximum=max(values),last=values[-1])
        progress=path.parent.parent/'progress.json'
        if progress.exists():
            data=json.loads(progress.read_text())
            episodes=data['episodes']
            overall=data.get('episode_summary',{}).get('overall',{})
            metrics['episode_summary']=overall
            if episodes:
                metrics['native_reward_means']={key:mean(e['combat_reward_summary'][key] for e in episodes)
                    for key in episodes[0]['combat_reward_summary']}
        report['training'][path.parent.parent.name]=metrics
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(dict(output=str(args.output),evaluations=len(report['evaluations']),
                          training_runs=len(report['training']))))


if __name__=='__main__':main()
