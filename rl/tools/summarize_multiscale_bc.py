"""Bounded paired BC comparison; never promote a model on a small win count."""
import argparse
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

from sokurl_platform.replay_format import Replay
from soku_rl.replay.human_dataset import save_json,sha256
from summarize_human_bc import summarize_games


def weighted_comparison(run,evaluation,reference):
    """Compare a C-only repeat to frozen paired evidence,without rerunning it."""
    import ast
    import torch
    read = lambda p:json.loads(p.read_text())
    new,old = read(run/'C/result.json'),read(reference/'C/result.json')
    prior = read(reference/'delivery.json')
    for key in ('initial_policy_parameter_sha256','initial_probability_sha256','anchor_identity',
                'train_anchors','validation_anchors','seed','epochs','batch_size','learning_rate','updates'):
        if new[key]!=old[key]: raise ValueError('unmatched C comparison:'+key)
    audits = [read(p/'data_audit.json') for p in (run,reference)]
    if any(audits[0][key]!=audits[1][key] for key in ('train_identity','validation_identity')):
        raise ValueError('window data differs from reference')
    if new['objective']!='L0+lambda/4*sum(L1..L4)': raise ValueError('wrong objective')
    for row in new['history'][1:]:
        if not np.isclose(np.dot(row['mean_batch_horizon_losses'],new['loss_coefficients']),
                          row['mean_batch_loss'],rtol=1e-5,atol=1e-6):
            raise ValueError('observed loss components do not sum to formula')
    initial = [torch.load(p/'C/initial-context.pt',map_location='cpu',weights_only=True) for p in (run,reference)]
    for section in ('context','future'):
        if initial[0][section].keys()!=initial[1][section].keys() or any(
                not torch.equal(v,initial[1][section][k]) for k,v in initial[0][section].items()):
            raise ValueError('context/future initialization differs')
    root = Path(__file__).resolve().parents[2]
    model_path = 'rl/src/soku_rl/rl/multiscale_bc.py'
    with zipfile.ZipFile(reference/'source-final.zip') as archive:
        previous = ast.parse(archive.read(model_path).decode())
    current = ast.parse((root/model_path).read_text())
    for name in ('HistoryContext','SequenceActor','load_actor','tensor_batch','score'):
        extract = lambda tree:ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name))
        if extract(previous)!=extract(current): raise ValueError('deployment semantics changed:'+name)
    base = {(r['world_seed'],r['seat']):r for r in prior['native']['C']['records']}
    original_games = {(g['world_seed'],g['learner_seat']):g for p in
        (reference/'paired-evaluation/C').glob('seed-*/result.json') for g in read(p)['result']['games']}
    folders = sorted((evaluation/'C').glob('seed-*'))
    if len(folders)!=4: raise ValueError('all four paired worlds required')
    games,catalogs = [],[]
    allowed = {'RL/src/soku_rl/rl/multiscale_bc.py','RL/tools/train_multiscale_bc.py','RL/tools/summarize_multiscale_bc.py'}
    for folder in folders:
        result,catalog = read(folder/'result.json'),read(folder/'catalog.json')
        if not result['success'] or len(catalog['games'])!=2 or result['candidate_fingerprint']!=sha256(run/'C/best.json'):
            raise ValueError('incomplete/wrong C evaluation')
        runtime = result['runtime']
        if (runtime['fixed_weather']!='clear' or not runtime['cards_disabled'] or runtime['fresh_reset']
                or not runtime['record_replays'] or runtime['fingerprints']['game_id'] not in prior['native_game_ids']):
            raise ValueError('native conditions changed')
        historical = read(reference/'paired-evaluation/C'/folder.name/'result.json')
        hashes,old_hashes = runtime['fingerprints']['source_hashes'],historical['runtime']['fingerprints']['source_hashes']
        changes = {k for k in hashes.keys()|old_hashes.keys() if hashes.get(k)!=old_hashes.get(k)}
        if changes-allowed: raise ValueError('unexpected runtime source changes:'+str(changes-allowed))
        for entry in catalog['games']:
            path = folder/entry['replay'];raw = path.read_bytes()
            if sha256(path)!=entry['sha256'] or Replay.decode(raw).encode()!=raw: raise ValueError('REP bytes changed')
        for game in result['result']['games']:
            old_game = original_games[(game['world_seed'],game['learner_seat'])]
            if (game['policy_seeds']!=old_game['policy_seeds'] or game['match']!=old_game['match']
                    or game['strategy_ids'][1-game['learner_seat']] not in prior['opponent_fingerprints']):
                raise ValueError('unpaired native setup/opponent/policy seeds')
        games.extend(result['result']['games'])
        catalogs.append(dict(path=str(folder/'catalog.json'),sha256=sha256(folder/'catalog.json')))
    native = summarize_games(games)
    if native['games']!=8: raise ValueError('exactly8 games required')
    pairs = [dict(world_seed=r['world_seed'],seat=r['seat'],
        damage_difference_delta=r['damage_difference']-base[(r['world_seed'],r['seat'])]['damage_difference'])
        for r in native['records']]
    with zipfile.ZipFile(run/'source-final.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for part in ('rl/src','rl/tools','rl/tests'):
            for path in sorted((root/part).rglob('*.py')): archive.write(path,path.relative_to(root).as_posix())
        for path in (root/'rl/README.md',root/'rl/run.py'): archive.write(path,path.relative_to(root).as_posix())
    save_json(run/'delivery.json',dict(status='COMPLETE_BOUNDED_C_WEIGHT_REPEAT',lambda_value=new['future_weight'],
        reference=dict(path=str(reference/'delivery.json'),sha256=sha256(reference/'delivery.json')),
        matched_initialization='PASS_POLICY_CONTEXT_FUTURE_AND_CURRENT_PROBABILITIES',matched_data='PASS',
        loss_formula_checks='PASS_ALL10_EPOCH_COMPONENT_SUMS',deployment_code_ast='UNCHANGED',
        old_c_offline=old['best_validation'],new_c_offline=new['best_validation'],
        old_c_native=prior['native']['C'],new_c_native=native,prior_b_native=prior['native']['B'],
        sequence_baselines=prior['sequence_baselines'],new_input_interventions=new['interventions'],
        paired_vs_old_c=pairs,mean_paired_damage_delta=float(np.mean([p['damage_difference_delta'] for p in pairs])),
        catalogs=catalogs,recordings=dict(count=8,byte_roundtrip='PASS',full_playback_parity='NOT TESTED'),
        source_final_sha256=sha256(run/'source-final.zip'),
        limitations=['Only4world blocks/8games;not significance or model promotion.',
            'lambda1 changes relative task weights AND absolute loss scale,no division by1+lambda.',
            'Same opening-prefix bias;no native/reward/observation/1Fcontrol changes.']))
    print(json.dumps(dict(old_nll=old['best_validation']['horizons'][0]['nll'],
        new_nll=new['best_validation']['horizons'][0]['nll'],native=native['counts'],
        net_hp=native['means']['damage_difference']),indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path)
    args = parser.parse_args()
    run,evaluation = args.run.resolve(strict=True),args.evaluation.resolve(strict=True)
    if (run/'delivery.json').exists():
        raise FileExistsError('preserve previous summary')
    if args.reference_run:
        return weighted_comparison(run,evaluation,args.reference_run.resolve(strict=True))
    reports = {v:json.loads((run/v/'result.json').read_text()) for v in 'ABC'}
    from soku_rl.rl.human_cloning import HumanDataset
    from soku_rl.replay.human_windows import WindowDataset
    data_root = Path(next(k for k in json.loads((run/'data_audit.json').read_text())['manifests']
                         if k.endswith('validation-20/manifest.json') or k.endswith('validation-20\\manifest.json'))).parent.parent
    windows = WindowDataset(HumanDataset(data_root/'validation-20/manifest.json',data_root/'validation-acceptance.json'))
    audit = json.loads((run/'data_audit.json').read_text())
    if windows.identity != audit['validation_identity']:
        raise ValueError('validation window identity changed')
    labels,mask = windows.data['labels'],windows.data['mask']
    full = mask.all(-1)
    previous = windows.data['previous_action']
    repeated = full & (previous>=0)
    sequence_baselines = dict(complete=int(full.sum()),
        constant_neutral_five_accuracy=float((labels[full]==40).all(-1).mean()),
        repeat_previous_five_accuracy=float((labels[repeated]==previous[repeated,None]).all(-1).mean()),
        note='Deterministic five-frame baselines on the exact complete validation sequences.')
    results,paired,opponents,games_ids = {},{},set(),set()
    source_hashes = None
    for variant in 'ABC':
        games,catalogs = [],[]
        folders = sorted((evaluation/variant).glob('seed-*'))
        if len(folders)!=4:
            raise ValueError('need all four paired worlds/group')
        for folder in folders:
            result = json.loads((folder/'result.json').read_text())
            catalog = json.loads((folder/'catalog.json').read_text())
            if not result['success'] or len(catalog['games'])!=2:
                raise ValueError('incomplete native comparison')
            runtime = result['runtime']
            if (not runtime['cards_disabled'] or runtime['fixed_weather']!='clear'
                    or runtime['fresh_reset'] or not runtime['record_replays']):
                raise ValueError('native conditions changed')
            if result['candidate_fingerprint']!=sha256(run/variant/'best.json'):
                raise ValueError('evaluated artifact identity changed')
            games_ids.add(runtime['fingerprints']['game_id'])
            hashes = runtime['fingerprints']['source_hashes']
            if source_hashes is None:
                source_hashes = hashes
            elif source_hashes != hashes:
                raise ValueError('runtime sources changed during comparison')
            for entry in catalog['games']:
                path = folder/entry['replay']
                raw = path.read_bytes()
                replay = Replay.decode(raw)
                if sha256(path)!=entry['sha256'] or replay.encode()!=raw:
                    raise ValueError('native REP hash or byte roundtrip failed')
            for game in result['result']['games']:
                key = (game['world_seed'],game['learner_seat'])
                opponents.add(game['strategy_ids'][1-game['learner_seat']])
                if variant=='A':
                    if key in paired: raise ValueError('duplicate world/seat')
                    paired[key] = game
                else:
                    previous = paired[key]
                    if game['policy_seeds']!=previous['policy_seeds'] or game['match']!=previous['match']:
                        raise ValueError('unpaired physical setup or policy seed')
            games.extend(result['result']['games'])
            catalogs.append(dict(path=str(folder/'catalog.json'),sha256=sha256(folder/'catalog.json')))
        results[variant] = summarize_games(games)
        results[variant]['catalogs'] = catalogs
    if len(opponents)!=1 or len(games_ids)!=1 or any(v['games']!=8 for v in results.values()):
        raise ValueError('opponent/game identity differs or comparison incomplete')
    comparisons = {}
    original = {(r['world_seed'],r['seat']):r for r in results['A']['records']}
    for variant in 'BC':
        differences = [dict(world_seed=r['world_seed'],seat=r['seat'],
            damage_difference_delta=r['damage_difference']-original[(r['world_seed'],r['seat'])]['damage_difference'])
            for r in results[variant]['records']]
        blocks = [np.mean([r['damage_difference_delta'] for r in differences if r['world_seed']==s])
                  for s in sorted({r['world_seed'] for r in differences})]
        comparisons[variant] = dict(pairs=differences,mean=float(np.mean(blocks)),
            limitation='Only four independent world blocks;no significance or model-promotion claim.')
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(run/'source-final.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [root/'rl/run.py',root/'rl/README.md']
        for part in ('rl/src','rl/tools','rl/tests'):
            paths.extend(sorted((root/part).rglob('*.py')))
        for path in paths:
            archive.write(path,path.relative_to(root).as_posix())
    # Plot comparable validation head0 and change-frame likelihood,not rewards.
    svg = ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width='900',height='300',viewBox='0 0 900 300')
    ET.SubElement(svg,'rect',width='900',height='300',fill='white')
    for panel,key in enumerate(('nll','changed_nll','accuracy')):
        left,top,width,height = 40+panel*300,40,245,200
        ET.SubElement(svg,'text',x=str(left),y='23').text = 'Validation current-action '+key
        maximum = 5 if 'nll' in key else 1
        for variant,color in zip('ABC',('#2166ac','#1b7837','#b2182b')):
            points = ' '.join(f'{left+r["epoch"]*width/10:.2f},{top+height-r["validation"]["horizons"][0][key]*height/maximum:.2f}'
                              for r in reports[variant]['history'])
            ET.SubElement(svg,'polyline',points=points,fill='none',stroke=color)
            ET.SubElement(svg,'text',x=str(left+60*'ABC'.index(variant)),y='275',fill=color).text = variant
    ET.ElementTree(svg).write(run/'curves.svg',encoding='utf-8',xml_declaration=True)
    save_json(run/'delivery.json',dict(status='COMPLETE_BOUNDED_EXPERIMENT_NOT_MODEL_PROMOTION',
        offline={v:dict(best_epoch=r['best_epoch'],updates=r['updates'],seconds=r['seconds'],
            best_validation=r['best_validation'],best_train=r['best_train'],
            interventions=r['interventions'],input_order_effect=r.get('input_order_effect'),
            source_parameter_sha256=r['initial_policy_parameter_sha256']) for v,r in reports.items()},
        native=results,paired_vs_A=comparisons,opponent_fingerprints=list(opponents),
        sequence_baselines=sequence_baselines,
        native_game_ids=list(games_ids),recordings=dict(count=24,byte_roundtrip='PASS',full_playback_parity='NOT TESTED'),
        source_final_sha256=sha256(run/'source-final.zip'),
        limitations=['Opening-prefix training data only;no sealed test tuning.',
            'Current-action selection and future-sequence prediction are separately reported.',
            'Horizon-specific parallel heads are ordered,but not an autoregressive joint sequence decoder.',
            'Distant context uses visible action/animation and object summaries,not separate contact-event tokens.',
            'Training raw history is consumed input;live history is own submitted commands from the existing wrapper,not new native acceptance evidence.',
            'Recorded config retains inert historical initial_policy/output metadata;actual initialization/source hash and all live outputs use explicit project-local overrides.',
            'CPU inference and four worlds/group;small-sample results,not statistical evidence.',
            'No PPO continuation or changed decision interface;B/C need their context sidecar.']))
    print(json.dumps({v:dict(wins=r['counts']['win'],**r['means']) for v,r in results.items()},indent=2))


if __name__=='__main__':
    main()
