"""Build immutable offline chains from accepted exports, without launching TH123."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from soku_rl.replay.action_chains import extract_chains, decision_examples, input_program, weather_eligible
from soku_rl.replay.human_dataset import sha256, save_json
from soku_rl.rl.human_cloning import HumanCollection, HumanDataset
from validate_human_export import raw_states


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--validation-acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root/'rl/runs') or output.exists():
        raise ValueError('new project-local run required')
    training = HumanCollection(args.collection)
    validation = HumanDataset(args.validation, args.validation_acceptance)
    if training.manifest['partition'] != 'train' or validation.manifest['partition'] != 'validation':
        raise ValueError('partition mismatch')
    for field in ('replay_id', 'source_sha256', 'set_id'):
        if {r[field] for r in training.manifest['replays']} & {r[field] for r in validation.manifest['replays']}:
            raise ValueError('train/validation overlap')
    if set(training.acceptance['players']) & set(validation.acceptance['players']):
        raise ValueError('shared human player')
    output.mkdir(parents=True)
    report = dict(schema='human_action_chains_v1', status='RUNNING', cases=[], sealed_test_used=False,
        observation_contract='1544 existing public features,not outcome/contact fields',
        decision_contract='Native accepted action IDs or raw movement runs;NOT90 primitive action labels',
        timing='Acceptance-time pre-state,not proven initial intention/command onset',
        training='NOT RUN', native_execution='NOT RUN', weather_training='BLOCKED_MISSING_WEATHER_OBSERVATION',
        limits=['Prefixes only;opening/selection bias remains.',
                'Observed succession is not a proven cancel/guaranteed combo.',
                'Movement runs are submitted inputs,not confirmed jumps/dashes.',
                'No whole-combo macros,no invented command notation orneutral missing inputs.',
                'Future outcomes remain in evidence JSON,never model inputs.',
                'No newly admitted weather rows orcard/skill modifications.'],
        sources={str(p.resolve()):sha256(p) for p in (args.collection,args.validation,args.validation_acceptance)},
        code={str(p.relative_to(root)):sha256(p) for p in
              (Path(__file__),root/'rl/src/soku_rl/replay/action_chains.py',
               root/'rl/src/soku_rl/env/input_lineage.py',root/'rl/src/soku_rl/replay/human_dataset.py')})
    save_json(output/'manifest.json', report)
    try:
        for partition, parts in [('train',training.parts),('validation',[validation])]:
            for part in parts:
                for case in part.manifest['replays']:
                    folder = part.path.parent/str(case['replay_id'])
                    for name,key in [('events.jsonl.gz','events_sha256'),('states.raw.gz','raw_sha256')]:
                        if sha256(folder/name) != case[key]:
                            raise ValueError('native evidence hash mismatch')
                    dense = []
                    for shard in case['shards']:
                        with np.load(folder/shard['path'],allow_pickle=False) as z:
                            dense.append({key:z[key] for key in z.files})
                    arrays = {k:np.concatenate([d[k] for d in dense]) for k in dense[0]}
                    with gzip.open(folder/'events.jsonl.gz','rt',encoding='utf-8') as stream:
                        records = [json.loads(line) for line in stream]
                    states = list(raw_states(folder/'states.raw.gz'))
                    if (len(states)!=len(arrays['actions'])+1 or
                            [int(s.frameId) for s in states]!=[0,*map(int,arrays['consumption_frames'])]):
                        raise ValueError('state/row chronology mismatch')
                    chains = extract_chains(arrays,records,case['seat'])
                    samples = decision_examples(arrays,chains)
                    # Separate visible condition metadata,not silently appended to1544/PPO.
                    samples['weather_before'] = np.array([int(states[int(i)].activeWeather)
                        for i in samples['observation_rows']],dtype=np.int32)
                    destination = output/partition/str(case['replay_id'])
                    destination.mkdir(parents=True)
                    np.savez_compressed(destination/'decisions.npz',**samples)
                    # Serialize only known contiguous runs;never fill no-poll gaps.
                    valid = arrays['compatible'] & arrays['raw_words_present'] & ~arrays['boundary']
                    runs, start = [], None
                    for i in range(len(valid)+1):
                        if i<len(valid) and valid[i] and start is None:
                            start=i
                        if start is not None and (i==len(valid) or not valid[i]):
                            runs.append(dict(first_frame=int(arrays['consumption_frames'][start]),
                                last_frame=int(arrays['consumption_frames'][i-1]), facing=1,
                                program=input_program(arrays['raw_words'][start:i,case['seat']])))
                            start=None
                    chains['absolute_input_programs'] = runs
                    save_json(destination/'chains.json',chains)
                    # Read existing state schema through the already validated reader.
                    weather, eligible = Counter(), 0
                    for state in states:
                        weather[int(state.activeWeather)]+=1
                        eligible+=int(weather_eligible(state.activeWeather))
                    statistics = dict(partition=partition,replay_id=case['replay_id'],
                        seat=case['seat'],source_manifest=str(part.path),source_manifest_sha256=sha256(part.path),
                        native_events_sha256=case['events_sha256'],raw_sha256=case['raw_sha256'],
                        decisions=len(chains['decisions']),movements=len(chains['movements']),
                        samples=len(samples['kind']),supervised_samples=int(samples['supervised'].sum()),
                        combos=len(chains['combos']),multi_action_combos=sum(c['independent_actions']>=2 for c in chains['combos']),
                        eligible_behavior_chains=sum(c['behavior_chain_eligible'] and
                            sum(s['kind']=='ACCEPTED_OPERATION' for s in c['behavior_steps'])>=2 for c in chains['combos']),
                        p0_range_damage_chains=sum(c['damage_sources_in_p0_attack_range'] and c['independent_actions']>=2 for c in chains['combos']),
                        non_attack_root_combos=sum(bool(c['non_attack_root_actions']) for c in chains['combos']),
                        contacts=len(chains['contacts']),unknown_root_contacts=sum(c['decision_id'] is None for c in chains['contacts']),
                        no_contact_operations=sum(not(d['outcomes']['hits'] or d['outcomes']['blocks']) for d in chains['decisions']),
                        simultaneous_acceptance_samples=sum(d['same_frame_acceptances']>1 for d in chains['decisions']),
                        weather_states=dict(weather),non_typhoon_state_candidates=eligible,
                        weather_scope='Candidate count only;doesnot certifycard/skill/historycompatibility',
                        artifacts={p.name:sha256(p) for p in destination.iterdir()})
                    report['cases'].append(statistics)
                    save_json(output/'manifest.json',report)
        report['totals'] = {partition:{key:sum(c[key] for c in report['cases'] if c['partition']==partition)
            for key in ('decisions','movements','samples','supervised_samples','combos','multi_action_combos',
                        'eligible_behavior_chains','p0_range_damage_chains','non_attack_root_combos',
                        'contacts','unknown_root_contacts','no_contact_operations','simultaneous_acceptance_samples')}
            for partition in ('train','validation')}
        report['status']='PASS_OFFLINE_EXTRACTION_NOT_POLICY_VALIDATION'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        save_json(output/'manifest.json',report)
    print(json.dumps(report['totals'],indent=2))


if __name__=='__main__':
    main()
