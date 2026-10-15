"""Cold-start observer OFF/ON native parity, contact and Rate boundary checks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import hashlib

from bridge_shared import BridgeClient
from game_runtime.combat_capture import CombatCaptureClient
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS, encode_action
from validate_platform_integration import VISIBILITY


def run_replay_case(enabled, frames, path, replay):
    from replay_validation import launch_replay_checkpoint
    from soku_rl.env.input_lineage import InputLineage
    lineage=InputLineage()
    os.environ['SOKURL_COMBAT_EVENTS'] = '1' if enabled else '0'
    result = dict(enabled=enabled, status='RUNNING', states=[], events=[],
                  replay=str(replay.resolve()), replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
    # Native replay inputs/weather/cards remain authoritative; no live input injection.
    saved = {key:os.environ.get(key) for key in (
        'SOKURL_VS_BOOTSTRAP','SOKURL_CARDS_DISABLED','SOKURL_CAPTURE_IMAGES')}
    for key in saved:
        os.environ.pop(key,None)
    os.environ['SOKURL_CARDS_DISABLED']='0'
    instance = combat = None
    start=time.monotonic()
    try:
        instance=launch_replay_checkpoint(replay)
        if enabled: combat=CombatCaptureClient(instance.pid)
        for _ in range(frames):
            raw=instance.step_native()
            result['states'].append(dict(frame=int(raw.frameId),hash=f'{raw.stateHash:016X}',
                hp=[int(raw.p1.hp),int(raw.p2.hp)],spirit=[int(raw.p1.spirit),int(raw.p2.spirit)],
                characters=[int(raw.p1.characterId),int(raw.p2.characterId)],weather=int(raw.activeWeather)))
            if combat: result['events'].extend(lineage.observe(combat.drain(),
                (int(raw.p1.characterId),int(raw.p2.characterId))))
            if instance.client.snapshot().dropped_frames:
                raise RuntimeError('replay frame recording dropped')
        result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',error=repr(error))
        raise
    finally:
        if combat: combat.close()
        if instance: instance.close()
        for key,value in saved.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
        result['seconds']=time.monotonic()-start
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def run_case(enabled, frames, path, mode='state', reward_damage_ch=False):
    os.environ['SOKURL_COMBAT_EVENTS'] = '1' if enabled else '0'
    root = Path(os.environ['SOKURL_RL_ROOT'])
    backend = WorkerBackend([sys.executable,'-B',str(root/'tools/rollout_worker.py')],
        os.environ['SOKURL_PLATFORM_ROOT'],str(path.with_suffix('.worker.log')),120,40,True,
        os.environ['SOKURL_GAME_DIR'],fresh_reset=True,
        cards_disabled=True,fixed_weather='clear',combat_events=enabled)
    reward_config = dict(combo_beta=0,passive_b=0) if enabled and reward_damage_ch else None
    config = EpisodeConfig(7200,8 if mode=='animation' else 4,1,0,mode,VisibilityConfig(**VISIBILITY),LEGACY_MATCH,
                           combat_reward=reward_config)
    backend.configure_observation(config.backend_observation())
    env = TwoPlayerVectorEnv(backend,1,config)
    bridge = combat = None
    result = dict(enabled=enabled, status='RUNNING', states=[], events=[],reward_parts=[])
    start = time.monotonic()
    try:
        env.reset({0:1732})
        pid = backend.request('status',[0])[0]['pid']
        bridge = BridgeClient(pid)
        if enabled and mode!='animation': combat = CombatCaptureClient(pid)
        for frame in range(1,frames+1):
            # Wait out startup. Close distance, then alternate shooting/A.
            t = frame-600
            if t < 0:
                p1 = p2 = (0,)*8
            elif t < 60:
                p1,p2 = (1,0,0,0,0,0,0,0),(-1,0,0,0,0,0,0,0)
            else:
                phase=(t-60)%90
                p1=(0,0,int(phase<3),int(30<=phase<33),int(60<=phase<63),0,0,0)
                # Some blocks, some exposed states, no collision inference.
                p2=(1 if (t//180)%2 else 0,0,0,0,0,0,0,0)
            _,_,terms,truncs,infos = env.step({0:{AGENTS[0]:encode_action(p1),AGENTS[1]:encode_action(p2)}})
            raw=bridge.snapshot().latest
            result['states'].append(dict(frame=int(raw.frameId),hash=f'{raw.stateHash:016X}',
                hp=[int(raw.p1.hp),int(raw.p2.hp)], spirit=[int(raw.p1.spirit),int(raw.p2.spirit)]))
            if combat: result['events'].extend(combat.drain())
            if enabled and mode=='animation':
                result['events'].extend(infos[0][AGENTS[0]]['native_combat_events'])
                if reward_damage_ch:
                    result['reward_parts'].append([infos[0][agent]['combat_reward_parts'] for agent in AGENTS])
            if any(terms[0].values()) or any(truncs[0].values()): break
        result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        if combat: combat.close()
        if bridge: bridge.close()
        env.close()
        result['seconds']=time.monotonic()-start
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=2400)
    parser.add_argument('--mode',choices=('state','animation'),default='state')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--reward-damage-ch',action='store_true')
    args=parser.parse_args()
    if args.reward_damage_ch and (args.replay or args.mode!='animation'):
        parser.error('partial live reward requires animation P0, not replay')
    args.output.mkdir(parents=True,exist_ok=False)
    old=os.environ.get('SOKURL_COMBAT_EVENTS')
    report={'status':'RUNNING'}
    try:
        runner = (lambda enabled,path:run_replay_case(enabled,args.frames,path,args.replay)) if args.replay else (
            lambda enabled,path:run_case(enabled,args.frames,path,args.mode,args.reward_damage_ch))
        off=runner(False,args.output/'off.json')
        if args.baseline_only:
            report.update(status='PASS',scope='OFF_ONLY',frames=len(off['states']))
            return
        on=runner(True,args.output/'on.json')
        mismatch=next((i+1 for i,(a,b) in enumerate(zip(off['states'],on['states'])) if a!=b),None)
        report['first_difference_frame']=mismatch
        if mismatch or len(off['states'])!=len(on['states']):
            raise RuntimeError('observer changes native frame state; do not enable training')
        hits=[e for e in on['events'] if e['kind']=='HIT_RESOLUTION' and e['hp_after']<e['hp_before']]
        blocks=[e for e in on['events'] if e['kind'].startswith('BLOCK')]
        if not hits or not blocks: raise RuntimeError('contact coverage incomplete')
        if any(e['rate_at_damage'] is None or e['rate_read_count']<1 for e in hits):
            raise RuntimeError('damage-time Rate capture incomplete')
        actions={(e['segment'],e['attacker_seat'],e['attack_instance']) for e in on['events']
                 if e['kind']=='ACTION_TRANSITION' and e['accepted'] and e['lineage']=='DIRECT'}
        if not args.replay and any((e['segment'],e['attacker_seat'],e['attack_instance']) not in actions for e in hits):
            raise RuntimeError('hit root is not a captured accepted action instance')
        # Exact HP accounting: distinguish hit damage and block chip, each once.
        for seat in range(2) if not args.replay else ():
            observed=sum(max(0,e['hp_before']-e['hp_after']) for e in on['events']
                         if e['kind'] in ('HIT_RESOLUTION','BLOCK_HIGH_RESOLUTION','BLOCK_LOW_RESOLUTION')
                         and e['defender_seat']==seat)
            actual=10000-on['states'][-1]['hp'][seat]
            if observed!=actual: raise RuntimeError('native contact HP accounting incomplete')
        report.update(status='PASS',frames=len(on['states']),hit_events=len(hits),block_events=len(blocks),
                      hits_with_attack_instance=sum(e['attack_instance'] is not None for e in hits),
                      hits_with_source_instance=sum(e['source_instance'] is not None for e in hits),
                      independent_input_segments='UNKNOWN',punish_counter='UNKNOWN')
        report.update(first_ch_events=sum(e.get('counter_hit_first') is True for e in hits),
                      unknown_first_ch=sum(e.get('counter_hit_first') is None for e in hits),
                      punish_definition='FIRST_HIT_NATIVE_CH_PROXY',
                      scope='REPLAY_PREFIX_NATIVE_CONTACTS' if args.replay else 'P0_CLEAR_NO_CARDS',
                      replay_attribution='PARTIAL_NON_P0_VTABLES' if args.replay else 'P0_VALIDATED')
        if args.reward_damage_ch:
            if len(on['reward_parts'])!=len(on['states']):
                raise RuntimeError('live reward frame coverage incomplete')
            report['reward_profile']='PARTIAL_DAMAGE_CH_NO_COMBO_NO_PASSIVE'
            report['reward_frames']=len(on['reward_parts'])
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        if old is None: os.environ.pop('SOKURL_COMBAT_EVENTS',None)
        else: os.environ['SOKURL_COMBAT_EVENTS']=old
        (args.output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
