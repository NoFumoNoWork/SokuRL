"""Audit causal shards, source identity, repeated prefixes and native round isolation."""
import argparse
from collections import Counter, deque
import ctypes
import gzip
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from sokurl_platform.bridge_shared import RawFrameState
from sokurl_platform.probe_replay_commands import battle_diff
from sokurl_platform.replay_format import Replay
from soku_rl.env.encoding import decode_action
from soku_rl.replay.human import consumed_pair
from soku_rl.replay.human_dataset import policy_interface, save_json, sha256, validate_arrays
from soku_rl.replay.human_plans import export_plan


def raw_states(path):
    width = ctypes.sizeof(RawFrameState)
    with gzip.open(path,'rb') as stream:
        while data := stream.read(width):
            if len(data) != width:
                raise ValueError('truncated raw state')
            yield RawFrameState.from_buffer_copy(data)


def inspect_case(case, directory, config, prior):
    folder = directory/str(case['replay_id'])
    if case['status'] != 'PASS_EXPORT' or sha256(case['source']) != case['source_sha256']:
        raise ValueError('source or capture is not accepted')
    for name,key in [('states.raw.gz','raw_sha256'),('events.jsonl.gz','events_sha256')]:
        if sha256(folder/name) != case[key]:
            raise ValueError('raw evidence changed')
    replay = Replay.decode(Path(case['source']).read_bytes())
    words = replay.matches[0].input_words
    interface = policy_interface(config,tuple(p.character for p in replay.matches[0].players))
    seat = case['seat']
    blocked = set(case['masked_actions'])
    previous_commands = deque([[0]*8]*interface.config.action_history,
                              maxlen=interface.config.action_history)
    count = supervised = changed = changed_supervised = 0
    raw_groups, exclusions, excluded_words, labels = Counter(), Counter(), Counter(), Counter()
    examples, previous_inputs, seen_examples = [], None, set()
    for shard in case['shards']:
        path = folder/shard['path']
        if sha256(path) != shard['sha256']:
            raise ValueError('shard hash changed')
        with np.load(path,allow_pickle=False) as archive:
            data = {key:archive[key] for key in archive.files}
            validate_arrays(data, words, replay.header[8], seat, interface, blocked)
            history_tokens = data['observations'][:,:-64].reshape(len(data['actions']),8,-1)
            for offset, character in ((0,1),(10,0)):
                visible = history_tokens[:,:,offset] > 0
                if not np.all(history_tokens[:,:,offset+4][visible] == character):
                    raise ValueError('policy history has reversed character roles')
            for i in range(len(data['actions'])):
                if data['consumption_frames'][i] != count+1:
                    raise ValueError('shard ordering or coverage gap')
                tail = np.asarray(previous_commands,np.float32).reshape(-1)
                if not np.array_equal(data['observations'][i,-len(tail):],tail):
                    raise ValueError('own command history differs from past raw words')
                if not np.isclose(data['observations'][i,-65],count/interface.episode.max_frames):
                    raise ValueError('observation time is not the pre-action frame')
                count += 1
                supervised += int(data['supervised'][i])
                changed += int(data['changed'][i])
                changed_supervised += int(data['changed'][i] and data['supervised'][i])
                if data['supervised'][i]:
                    labels[int(data['actions'][i])] += 1
                else:
                    exclusions[str(data['reason'][i])] += 1
                if data['boundary'][i]:
                    previous_commands.append([0]*8)
                    continue
                pair = consumed_pair(words,replay.header[8],int(data['remaining_before'][i]),
                                     int(data['remaining_after'][i]))
                if pair is None:
                    previous_commands.append([0]*8)
                else:
                    inputs = pair['logical'][seat]
                    previous_commands.append(inputs)
                    categories = []
                    if previous_inputs is not None:
                        if inputs[:2] != previous_inputs[:2]:
                            categories.append('DIRECTION_CHANGE')
                        if any(a > b for a,b in zip(inputs[2:],previous_inputs[2:])):
                            categories.append('BUTTON_PRESS')
                        if any(a < b for a,b in zip(inputs[2:],previous_inputs[2:])):
                            categories.append('BUTTON_RELEASE')
                    for kind in categories:
                        if kind not in seen_examples:
                            examples.append(dict(kind=kind,observation_frame=count-1,
                                consumption_frame=count,raw_words=pair['raw_words'],inputs=inputs,
                                previous_inputs=previous_inputs,supervised=bool(data['supervised'][i])))
                            seen_examples.add(kind)
                    previous_inputs = inputs
                    raw_groups.update(name for name, yes in [
                        ('direction',any(inputs[:2])),('melee',inputs[2]),
                        ('shooting',any(inputs[3:5])),('dash',inputs[5]),('cards',any(inputs[6:]))] if yes)
                    if not data['supervised'][i]:
                        excluded_words[str(pair['raw_words'][seat])] += 1
    if count != case['rows'] or supervised != case['supervised']:
        raise ValueError('manifest/shard counts differ')
    evidence_rows = 0
    with gzip.open(folder/'events.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            evidence_rows += 1
            if record['frame'] != evidence_rows:
                raise ValueError('event evidence frame gap')
    states = list(raw_states(folder/'states.raw.gz'))
    if len(states) != count+1 or evidence_rows != count:
        raise ValueError('raw states/events do not cover every retained time row')
    if prior is not None:
        baseline = list(raw_states(prior/str(case['replay_id'])/'frames.raw.gz'))
        overlap = min(len(baseline),len(states))
        for frame in range(overlap):
            if bytes(baseline[frame]) == bytes(states[frame]):
                continue
            if diff := battle_diff(baseline[frame],states[frame]):
                raise ValueError(f'repeated original prefix differs at{frame}:{diff}')
    else:
        overlap = 0
    for example in examples:
        before,after = (states[example['observation_frame']],states[example['consumption_frame']])
        example['state_before'] = dict(actions=[int(before.p1.actionId),int(before.p2.actionId)],
            hp=[int(before.p1.hp),int(before.p2.hp)],hitstop=[int(before.p1.hitstop),int(before.p2.hitstop)])
        example['state_after'] = dict(actions=[int(after.p1.actionId),int(after.p2.actionId)],
            hp=[int(after.p1.hp),int(after.p2.hp)],hitstop=[int(after.p1.hitstop),int(after.p2.hitstop)])
    return dict(replay_id=case['replay_id'], rows=count, supervised=supervised,
        supervision_ratio=supervised/count, changed_frames=changed, changed_supervised=changed_supervised,
        changed_supervision_ratio=changed_supervised/changed if changed else None,
        exclusions=dict(exclusions), excluded_raw_words=dict(excluded_words),
        raw_input_groups=dict(raw_groups), represented_labels=dict(labels),examples=examples,
        native_events=case['native_event_counts'],
        repeated_native_prefix_frames=max(0,overlap-1),
        full_replay_parity='NOT_TESTED', accepted_scope='CAUSAL_ORIGINAL_CONDITION_PREFIX_ONLY')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--round-case',type=Path,required=True)
    parser.add_argument('--feasibility',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve earlier acceptance')
    report = dict(status='RUNNING',schema='human_replay_prefix_acceptance_v1',cases=[],
                  manifest_sha256=sha256(args.dataset/'manifest.json'),new_hooks=0)
    try:
        manifest = json.loads((args.dataset/'manifest.json').read_text())
        plan_path,plan = export_plan(args.audit,manifest['partition'],args.plan)
        if manifest['status'] != 'PASS_EXPORT_ONLY' or len(manifest['replays']) != len(plan['rows']):
            raise ValueError('requires all planned original captures')
        if (manifest['identity']['plan_sha256'] != sha256(plan_path)
                or [c['replay_id'] for c in manifest['replays']] != [r['id'] for r in plan['rows']]):
            raise ValueError('pilot differs from audited training-only selection')
        for case,row in zip(manifest['replays'],plan['rows']):
            if case['source_sha256'] != row['sha256']:
                raise ValueError('source differs from audited split')
        config = OmegaConf.to_container(OmegaConf.load(args.source/'config.yaml'),resolve=True)
        for case in manifest['replays']:
            report['cases'].append(inspect_case(case,args.dataset,config,args.prior))
        round_case = json.loads(args.round_case.read_text())
        last = round_case['shards'][-1]
        with np.load(args.round_case.parent/last['path'],allow_pickle=False) as data:
            if (not round_case['boundaries'] or round_case['stop_reason'] != 'ROUND_BOUNDARY'
                    or not data['boundary'][-1] or data['supervised'][-1]
                    or round_case['rows'] != round_case['boundaries'][-1]['frame']):
                raise ValueError('real round transition was not isolated')
        report['round_boundary'] = dict(source=str(args.round_case), sha256=sha256(args.round_case),
            boundaries=round_case['boundaries'], handling='STOP_NOT_NEXT_ROUND_REINITIALIZATION')
        feasibility = json.loads(args.feasibility.read_text())
        if (feasibility['status'] != 'PASS_MINIMUM_BC_FEASIBILITY_ZERO_NEW_HOOK'
                or any(c['raw_independent_replay']['status'] != 'PASS' for c in feasibility['cases'])):
            raise ValueError('independent raw input calibration missing')
        report['calibration'] = dict(source=str(args.feasibility),sha256=sha256(args.feasibility),
            scope='Three256F independent live-VS prefixes;not20 full-match equivalence')
        report.update(status='PASS_PREFIX_PILOT',rows=sum(c['rows'] for c in report['cases']),
            supervised=sum(c['supervised'] for c in report['cases']),sealed_test_used=False,
            partition=manifest['partition'],
            players=sorted({p['id'] for row in plan['rows'] for p in row['players'].values()}),
            limitations=['No proof of full hidden-state parity or whole-match equivalence.',
                'Later rounds are excluded,not joined or given a fabricated initial history.',
                'Skill compatibility uses original fresh start,card exclusion and existing skillLevels;no new internal map reads.',
                'Raw input groups describe buttons,not accepted attacks or defensive success.'])
    except BaseException as error:
        report.update(status='BLOCKED',error=repr(error))
        raise
    finally:
        save_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','rows','supervised')}))


if __name__ == '__main__':
    main()
