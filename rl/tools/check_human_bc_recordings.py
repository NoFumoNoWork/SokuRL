"""Check catalogued native REP HP projections using unmodified capture()."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from sokurl_platform import sokurl, probe_replay_commands as probe
from sokurl_platform.startup import title_configuration
from soku_rl.replay.review import reviewed_replay
from soku_rl.replay.human_dataset import save_json, sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous recording checks')
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING',cases=[],full_per_frame_live_parity='NOT_TESTED',
                  scope='One paired seed per policy: existing native damage ledger;final HP only if independently logged')
    original_launcher = probe.launch_replay_checkpoint
    try:
        with patch.dict(os.environ,{'SOKURL_REPLAY_INPUT_CAPTURE':'0','SOKURL_COMBAT_EVENTS':'1',
                'SOKURL_NATIVE_TRACE':'0','SOKURL_NATIVE_INPUT_COUNTERS':'0','SOKURL_FRESH_RESET':'0',
                'SOKURL_HEADLESS_RENDER':'1','SOKURL_UNLIMITED_PACING':'1',
                'SOKURL_ORIGINAL_REPLAY_RECORDING':'0','SOKURL_MUTE_AUDIO':'1'}):
            selections = []
            for name in ('baseline','human-bc'):
                catalogs = sorted((args.evaluation/name).glob('seed-*/catalog.json'))
                if len(catalogs) != 8:
                    raise ValueError('requires all eight paired seeds per policy')
                selections.extend((name,catalogs[0],index) for index in (0,1))
            for name,catalog,index in selections:
                replay,entry,conditions = reviewed_replay(catalog,index)
                expected = entry['trial']['combat_metrics_by_seat'][0]
                if expected['available']:
                    wanted_hp = [expected['own_final_hp'],expected['opponent_final_hp']]
                    wanted_damage = [expected['own_hp_loss'],expected['opponent_hp_loss']]
                    reference = 'EXISTING_PER_FRAME_HP_METRICS'
                else:
                    ledger = entry['trial']['combat_reward_summary_by_seat']
                    wanted_hp = None
                    wanted_damage = [p['damage_taken'] for p in ledger]
                    reference = 'EXISTING_NATIVE_DAMAGE_LEDGER_VALIDATED_AGAINST_HP_EACH_LIVE_FRAME'
                folder = args.output/f'{name}-seat-{entry["trial"]["learner_seat"]}'
                folder.mkdir()
                def launch(path):
                    return original_launcher(path,cards_disabled=conditions['cards_disabled'],
                                             weather=conditions['fixed_weather'])
                with title_configuration(sokurl.SKIPINTRO_INI,45.),patch.object(probe,'launch_replay_checkpoint',launch):
                    states,_,_ = probe.capture(replay,entry['frames'],folder)
                hp = [int(states[-1].p1.hp),int(states[-1].p2.hp)]
                damage = [sum(max(0,int(getattr(a,f'p{s+1}').hp)-int(getattr(b,f'p{s+1}').hp))
                              for a,b in zip(states,states[1:])) for s in (0,1)]
                case = dict(policy=name,frames=entry['frames'],replay=str(replay),sha256=sha256(replay),
                    status='PASS_DAMAGE_PROJECTION' if (wanted_hp is None or hp==wanted_hp) and damage==wanted_damage else 'FAIL',
                    actual_hp=hp,expected_hp=wanted_hp,actual_hp_loss=damage,expected_hp_loss=wanted_damage,
                    reference=reference,live_final_hp='NOT_INDEPENDENTLY_LOGGED' if wanted_hp is None else wanted_hp,
                    state_capture_sha256=sha256(folder/'native_frames.raw.gz'),
                    limitation='Live per-frame native states were not stored by benchmark;this is NOT full-state parity.')
                report['cases'].append(case)
                save_json(args.output/'report.json',report)
                if case['status']=='FAIL':
                    raise ValueError('recorded replay public projection differs')
        report['status'] = 'PASS_BOUNDED_RECORDING_PROJECTION'
    except BaseException as error:
        report.update(status='BLOCKED',error=repr(error))
        raise
    finally:
        save_json(args.output/'report.json',report)


if __name__=='__main__':
    main()
