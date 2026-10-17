"""Summarize the fixed human-BC experiment without tuning or model promotion."""
import argparse
from collections import Counter
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from soku_rl.replay.human_dataset import save_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def summarize_games(games):
    counts = Counter(win=0,loss=0,double_ko=0,time_limit=0)
    records = []
    for game in games:
        seat = game['learner_seat']
        outcome = game['outcome']
        win = 'p1_win' if seat==0 else 'p2_win'
        key = outcome if outcome in ('double_ko','time_limit') else 'win' if outcome==win else 'loss'
        counts[key] += 1
        ledger = game['combat_reward_summary_by_seat'][seat]
        records.append(dict(world_seed=game['world_seed'],seat=seat,outcome=outcome,
            damage_dealt=ledger['damage_dealt'],damage_taken=ledger['damage_taken'],
            damage_difference=ledger['damage_dealt']-ledger['damage_taken'],frames=game['frames'],
            native_hits=ledger['hits'],blocks=ledger['blocks'],counter_hits=ledger['counter_hits'],
            combo_extensions=ledger['combo_extensions'],later_segment_hits=ledger['later_segment_hits'],
            passive_frames=ledger['passive_frames']))
    return dict(games=len(games),counts=dict(counts),win_rate=counts['win']/len(games),
        means={k:float(np.mean([r[k] for r in records])) for k in (
            'damage_dealt','damage_taken','damage_difference','frames','native_hits','blocks',
            'counter_hits','combo_extensions','later_segment_hits','passive_frames')},records=records)


def plot_training(training, directory):
    epochs = [r['epoch'] for r in training['history']]
    svg = ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width='1080',height='300',viewBox='0 0 1080 300')
    ET.SubElement(svg,'rect',width='1080',height='300',fill='white')
    def text(x,y,value,size=11):
        ET.SubElement(svg,'text',x=str(x),y=str(y),fill='#303030',
                      attrib={'font-family':'Arial,sans-serif','font-size':str(size)}).text = value
    for panel,(key,title,maximum) in enumerate((('nll','All labelled frames: NLL',5),
            ('changed_nll','Input-change frames: NLL',5),('accuracy','Action accuracy',1))):
        left,top,width,height = 45+panel*355,45,285,165
        text(left,24,title,13)
        for tick in range(6):
            y = top+height-tick*height/5
            ET.SubElement(svg,'line',x1=str(left),x2=str(left+width),y1=str(y),y2=str(y),stroke='#e5e5e5')
            text(left-27,y+4,f'{tick*maximum/5:g}',10)
        for tick in range(0,11,2):
            text(left+tick*width/10-4,228,str(tick),10)
        text(left+110,248,'BC epoch')
        for split,color in [('train','#286eaf'),('validation','#30834b')]:
            values = [r[split][key] for r in training['history']]
            points = ' '.join(f'{left+epoch*width/10:.2f},{top+height-value*height/maximum:.2f}'
                              for epoch,value in zip(epochs,values))
            ET.SubElement(svg,'polyline',points=points,fill='none',stroke=color,attrib={'stroke-width':'2'})
        if key=='accuracy':
            for value,color,dash in ((training['constant_action_baseline']['validation_accuracy'],'#b66725','6 4'),
                    (training['history'][0]['validation']['previous_input_baseline']['accuracy'],'#777777','2 3')):
                y = top+height-value*height
                ET.SubElement(svg,'line',x1=str(left),x2=str(left+width),y1=str(y),y2=str(y),
                              stroke=color,attrib={'stroke-dasharray':dash})
    for x,label,color in ((45,'Train','#286eaf'),(190,'Validation','#30834b'),
                           (390,'Constant action baseline','#b66725'),(720,'Previous input baseline','#777777')):
        ET.SubElement(svg,'line',x1=str(x),x2=str(x+25),y1='277',y2='277',stroke=color,attrib={'stroke-width':'2'})
        text(x+32,281,label)
    path = directory/'bc-curves.svg'
    if path.exists():
        raise FileExistsError('preserve previous curve')
    ET.ElementTree(svg).write(path,encoding='utf-8',xml_declaration=True)
    return dict(path=str(path),sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous summary')
    training = read(args.run/'training/result.json')
    accepted = {name:read(args.run/path) for name,path in (
        ('train','pilot-acceptance-v2.json'),('validation','validation-acceptance.json'))}
    results, runtime_ids, opponent_ids, critical_hashes = {},set(),set(),None
    originals = {}
    for name in ('baseline','human-bc'):
        folders = sorted((args.evaluation/name).glob('seed-*'))
        if len(folders) != 8:
            raise ValueError('missing paired seed batches')
        games, catalogs = [], []
        for folder in folders:
            report = read(folder/'result.json')
            catalog = read(folder/'catalog.json')
            if not report['success'] or len(catalog['games']) != 2:
                raise ValueError('incomplete recording batch')
            runtime = report['runtime']
            if (not runtime['cards_disabled'] or runtime['fixed_weather'] != 'clear'
                    or runtime['fresh_reset'] or not runtime['record_replays']):
                raise ValueError('evaluation conditions changed')
            fingerprints = runtime['fingerprints']
            runtime_ids.add(fingerprints['game_id'])
            current = {k:v for k,v in fingerprints['source_hashes'].items() if k.startswith((
                'Platform/','RL/src/','RL/tools/game_runtime/'))}
            if critical_hashes is None:
                critical_hashes = current
            elif critical_hashes != current:
                raise ValueError('environment,policy or Platform source changed during comparison')
            expected_hash = (training['source']['sha256'] if name=='baseline'
                             else training['checkpoints']['best'])
            if report['candidate_fingerprint'] != expected_hash:
                raise ValueError('evaluation checkpoint identity differs')
            for game in report['result']['games']:
                opponent_ids.add(game['strategy_ids'][1-game['learner_seat']])
                key = (game['world_seed'],game['learner_seat'])
                if name=='baseline':
                    if key in originals:
                        raise ValueError('duplicate paired game')
                    originals[key] = game
                else:
                    prior = originals[key]
                    if (game['policy_seeds'] != prior['policy_seeds'] or game['match'] != prior['match']):
                        raise ValueError('policy randomness or physical matchup is not paired')
            games.extend(report['result']['games'])
            catalogs.append(dict(path=str(folder/'catalog.json'),sha256=sha256(folder/'catalog.json')))
        summary = summarize_games(games)
        summary['catalogs'] = catalogs
        results[name] = summary
    if len(runtime_ids) != 1 or len(opponent_ids) != 1:
        raise ValueError('original game artifacts or pressure opponent differ between batches')
    pairs = []
    old = {(r['world_seed'],r['seat']):r for r in results['baseline']['records']}
    for row in results['human-bc']['records']:
        baseline = old.pop((row['world_seed'],row['seat']))
        pairs.append(dict(world_seed=row['world_seed'],seat=row['seat'],
            damage_difference_delta=row['damage_difference']-baseline['damage_difference']))
    if old or len(pairs) != 16:
        raise ValueError('unpaired games')
    seeds = sorted({p['world_seed'] for p in pairs})
    blocks = np.array([np.mean([p['damage_difference_delta'] for p in pairs if p['world_seed']==seed]) for seed in seeds])
    rng = np.random.default_rng(20261017)
    bootstrap = rng.choice(blocks,(10000,len(blocks)),replace=True).mean(axis=1)
    initial = training['history'][0]
    best = training['history'][training['best_epoch']]
    tr_counts = np.array(initial['train']['label_counts'])
    val_counts = np.array(initial['validation']['label_counts'])
    marginal = (tr_counts+.5)/(tr_counts.sum()+45)
    report = dict(schema='human_bc_delivery_v1',status='COMPLETE_FIRST_PILOT_BC_COMPARISON',
        data={k:dict(rows=v['rows'],supervised=v['supervised']) for k,v in accepted.items()},
        training={k:training[k] for k in ('seconds','updates','epochs_limit','best_epoch','batch_size',
            'initial_parameter_sha256','final_parameter_sha256','value_coef','gpu_peak_allocated_bytes','checkpoints')},
        offline=dict(initial=initial,best=best,constant_baseline=training['constant_action_baseline'],
            posthoc_smoothed_training_frequency_validation_nll=float(-(val_counts*np.log(marginal)).sum()/val_counts.sum()),
            diagnostic='Best argmax predicts neutral for every validation row;NLL gain is nearly explained by class priors.'),
        evaluation=results,game_artifact_identity=next(iter(runtime_ids)),
        pressure_fingerprint=next(iter(opponent_ids)),
        shared_runtime_source_consistency='PASS_PLATFORM_POLICY_ENVIRONMENT_AND_WORKER',
        paired=dict(samples=16,world_seed_blocks=len(seeds),
            mean_damage_difference_delta=float(blocks.mean()),
            descriptive_world_block_bootstrap_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
            pairs=pairs,significance='Small descriptive pilot,not proof of stable improvement'),
        native_edits=0,new_hooks=0,optimizer='Fresh;source PPO optimizer not inherited',
        deterministic_deployment_warning='Validation argmax is always neutral;stochastic evaluation still samples non-neutral inputs.',
        sealed_test_used=False,ppo_finetuning=False,commit_or_push=False,
        limitations=['Only original-condition Clear/default-skill/no-card prefixes,not complete human matches.',
            'Raw source words,all masked rows and chronological history are preserved.',
            'Benchmark animation mode has no independent final-HP CombatMetrics;damage ledger is native HP damage,not scaled reward.',
            'Hit/combo/CH totals available;full per-hit source-type distribution is not logged in these benchmark results.',
            'Failed earlier4-worker CUDA reset batch is retained and excluded from CPU fresh-worker comparison.',
            'No automatic64-game expansion,extra BC search or PPO training.'])
    checks = args.run/'recording-checks/report.json'
    report['recording_checks'] = read(checks) if checks.exists() else dict(status='NOT_TESTED')
    report['training_curves'] = plot_training(training,args.run)
    save_json(args.output,report)
    print(json.dumps(dict(policies={k:dict(win_rate=v['win_rate'],means=v['means']) for k,v in results.items()},
        paired=report['paired']['mean_damage_difference_delta'])))


if __name__ == '__main__':
    main()
