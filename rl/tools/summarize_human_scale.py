"""Report a bounded Dscale-up,not a controlled data-versus-budget ablation."""
import argparse
import ast
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

from sokurl_platform.replay_format import Replay
from soku_rl.replay.human_dataset import sha256,save_json
from soku_rl.rl.learner import parameter_hash
from soku_rl.rl.multiscale_bc import load_actor
from summarize_human_bc import summarize_games


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path,required=True)
    args = parser.parse_args()
    run,reference = args.run.resolve(strict=True),args.reference_run.resolve(strict=True)
    read = lambda path:json.loads(path.read_text())
    if (run/'delivery.json').exists(): raise FileExistsError('preserve earlier report')
    training = read(run/'training/D/result.json')
    prior = read(reference/'delivery.json')
    alias = read(run/'model_alias.json')
    if training['variant']!='D' or training['objective']!='L0+lambda/4*sum(L1..L4)' or training['future_weight']!=1:
        raise ValueError('D must retain the accepted weighted-C loss')
    if training['status']!='PASS_OFFLINE_NOT_POLICY_IMPROVEMENT' or training['epochs']!=20:
        raise ValueError('expanded training incomplete')
    if (training['initialization']['sha256']!=alias['source_sha256']
            or sha256(reference/'C/best.json')!=alias['source_sha256']):
        raise ValueError('D does not start at named weighted-C checkpoint')
    _,warm,_ = load_actor(reference/'C/best.json','cpu')
    if parameter_hash(warm)!=training['initialization']['actor_state_sha256']:
        raise ValueError('context/future weights were not carried forward')
    audits = [read(p) for p in (run/'training/data_audit.json',reference/'data_audit.json')]
    if audits[0]['validation_identity']!=audits[1]['validation_identity']:
        raise ValueError('fixed validation window set changed')
    plan = read(run/'scale-plan.json')
    if len(audits[0]['train_cases'])!=plan['training_total']:
        raise ValueError('expanded training source count differs')
    original_games = {(g['world_seed'],g['learner_seat']):g for p in
        (reference/'paired-evaluation/C').glob('seed-*/result.json') for g in read(p)['result']['games']}
    games,catalogs = [],[]
    for folder in sorted((run/'paired-evaluation/D').glob('seed-*')):
        result,catalog = read(folder/'result.json'),read(folder/'catalog.json')
        historical = read(reference/'paired-evaluation/C'/folder.name/'result.json')
        if not result['success'] or result['candidate_fingerprint']!=sha256(run/'training/D/best.json'):
            raise ValueError('wrong/incomplete D evaluation')
        runtime = result['runtime']
        if (runtime['fingerprints']['game_id']!=historical['runtime']['fingerprints']['game_id']
                or not runtime['cards_disabled'] or runtime['fixed_weather']!='clear'
                or runtime['fresh_reset'] or not runtime['record_replays']):
            raise ValueError('native conditions changed')
        protected = lambda r:{k:v for k,v in r['fingerprints']['source_hashes'].items()
            if k.startswith(('Platform/','RL/src/soku_rl/env/','RL/src/soku_rl/policy/'))}
        if protected(runtime)!=protected(historical['runtime']):
            raise ValueError('Platform,observation,reward orpolicy-control code changed')
        for entry in catalog['games']:
            path = folder/entry['replay'];raw = path.read_bytes()
            if sha256(path)!=entry['sha256'] or Replay.decode(raw).encode()!=raw: raise ValueError('REP byte identity changed')
        for game in result['result']['games']:
            previous = original_games[(game['world_seed'],game['learner_seat'])]
            seat = game['learner_seat']
            if any(game[k]!=previous[k] for k in ('match','policy_seeds')) or game['strategy_ids'][1-seat]!=previous['strategy_ids'][1-seat]:
                raise ValueError('world/seat/opponent/policy seed not paired')
        games.extend(result['result']['games'])
        catalogs.append(dict(path=str(folder/'catalog.json'),sha256=sha256(folder/'catalog.json')))
    if len(catalogs)!=4 or len(games)!=8: raise ValueError('exactly4two-seat batches required')
    native = summarize_games(games)
    before = {(r['world_seed'],r['seat']):r for r in prior['new_c_native']['records']}
    pairs = [dict(world_seed=r['world_seed'],seat=r['seat'],damage_difference_delta=
        r['damage_difference']-before[(r['world_seed'],r['seat'])]['damage_difference']) for r in native['records']]
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(reference/'source-final.zip') as archive:
        old_tree = ast.parse(archive.read('rl/src/soku_rl/rl/multiscale_bc.py').decode())
    tree = ast.parse((root/'rl/src/soku_rl/rl/multiscale_bc.py').read_text())
    method = lambda t:ast.dump(next(m for c in t.body if getattr(c,'name',None)=='SequenceActor'
                                   for m in c.body if getattr(m,'name',None)=='logits'))
    if method(tree)!=method(old_tree): raise ValueError('actor forward computation changed beyond D label')
    with zipfile.ZipFile(run/'source-final.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for part in ('rl/src','rl/tools','rl/tests'):
            for path in sorted((root/part).rglob('*.py')): archive.write(path,path.relative_to(root).as_posix())
        for path in (root/'rl/README.md',root/'rl/run.py'): archive.write(path,path.relative_to(root).as_posix())
    svg = ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width='900',height='300')
    ET.SubElement(svg,'rect',width='900',height='300',fill='white')
    for panel,key in enumerate(('nll','changed_nll','accuracy')):
        left,top,width,height = 40+300*panel,40,240,200
        ET.SubElement(svg,'text',x=str(left),y='24').text = 'D validation current '+key
        maximum = 5 if 'nll' in key else 1
        points = ' '.join(f'{left+r["epoch"]*width/20:.2f},{top+height-r["validation"]["horizons"][0][key]*height/maximum:.2f}'
                          for r in training['history'])
        ET.SubElement(svg,'polyline',points=points,fill='none',stroke='#2166ac')
    ET.ElementTree(svg).write(run/'curves.svg',encoding='utf-8',xml_declaration=True)
    save_json(run/'delivery.json',dict(status='COMPLETE_BOUNDED_D_SCALEUP',model_alias=alias,
        training_source_count=len(audits[0]['train_cases']),validation_source_count=len(audits[0]['validation_cases']),
        train_anchors=training['train_anchors'],validation_anchors=training['validation_anchors'],
        epochs=training['epochs'],updates=training['updates'],anchor_visits=training['epochs']*training['train_anchors'],
        seconds=training['seconds'],best_epoch=training['best_epoch'],
        before_offline=training['history'][0]['validation'],after_offline=training['best_validation'],
        train_offline=training['best_train'],input_interventions=training['interventions'],
        native_before=prior['new_c_native'],native_after=native,paired=pairs,
        mean_paired_net_hp_delta=float(np.mean([p['damage_difference_delta'] for p in pairs])),
        sequence_baselines=prior['sequence_baselines'],catalogs=catalogs,
        recordings=dict(count=8,byte_roundtrip='PASS',full_playback_parity='NOT TESTED'),
        source_final_sha256=sha256(run/'source-final.zip'),
        reference=dict(path=str(reference/'delivery.json'),sha256=sha256(reference/'delivery.json')),
        limitations=['More data and more training jointly changed;not a factorial ablation.',
            'Only original-condition compatible prefixes,not complete human matches.',
            'Four paired diagnostic worlds,not statistical confirmation or a sealed test evaluation.',
            'Full-context weights continued with fresh Adam;old optimizer moments unavailable.',
            '1Fcontrol unchanged;future heads do not execute open-loop plans.']))
    print(json.dumps(dict(wins=native['counts']['win'],net_hp=native['means']['damage_difference'],
        nll=training['best_validation']['horizons'][0]['nll'],sequence_exact=training['best_validation']['sequence_exact_accuracy']),indent=2))


if __name__=='__main__':
    main()
