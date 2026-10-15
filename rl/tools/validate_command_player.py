"""Native two-instance command execution: the same player drives every move."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from bridge_shared import BridgeClient
from soku_rl.env.commands import CommandPlayer,compile_commands,NEUTRAL
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH,MatchConfig,PlayerSetup
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS,encode_action
from validate_platform_integration import VISIBILITY
from game_runtime.privileged import ProcessMemory
from game_runtime.action_state import NativeActionStateReader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-reward',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve previous evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    root=Path(os.environ['SOKURL_RL_ROOT'])
    backend=WorkerBackend([sys.executable,'-B',str(root/'tools/rollout_worker.py')],
        os.environ['SOKURL_PLATFORM_ROOT'],str(args.output.with_suffix('.worker.log')),120,40,True,
        os.environ['SOKURL_GAME_DIR'],fresh_reset=True,
        cards_disabled=True,fixed_weather='clear',combat_events=True,native_input_counters=True)
    reward=dict(combo_beta=.3,passive_b=.02,unknown_combo_policy='no_bonus') if args.full_reward else None
    config=EpisodeConfig(7200,8,1,0,'animation',VisibilityConfig(**VISIBILITY),LEGACY_MATCH,combat_reward=reward)
    backend.configure_observation(config.backend_observation())
    env=TwoPlayerVectorEnv(backend,2,config);clients={};memories={};report=dict(status='RUNNING',cases=[])
    start=time.monotonic();frame=0
    def advance(keys):
        nonlocal frame
        result=env.step({slot:{AGENTS[0]:encode_action(keys[slot]),AGENTS[1]:encode_action(NEUTRAL)} for slot in range(2)})
        frame+=1
        if any(any(x.values()) for group in result[2:4] for x in group.values()):
            raise RuntimeError('command validation ended before completion')
        return result[-1]
    try:
        env.reset_matchups({0:1732,1:1733},{0:LEGACY_MATCH,
            1:MatchConfig(PlayerSetup(0,0,0),PlayerSetup(1,0,0))})
        for slot,item in backend.request('status',[0,1]).items():
            clients[slot]=BridgeClient(item['pid']);memories[slot]=ProcessMemory(item['pid'])
        report['characters']=[int(c.snapshot().latest.p1.characterId) for c in clients.values()]
        for _ in range(600):advance({0:NEUTRAL,1:NEUTRAL})
        # Both slots have identical characters; independent seeds and command queues.
        for name,text in [('5B','5B'),('236B','236B'),('623B','623B'),('22B','22B'),
                          ('charge2B','2B{40}'),('AAA','AAA')]:
            if name=='AAA':
                for _ in range(250):
                    raw={s:c.snapshot().latest for s,c in clients.items()}
                    if all(abs(r.p1.x-r.p2.x)<45 for r in raw.values()):break
                    advance({s:(int(r.p1.facing),0,0,0,0,0,0,0)
                        if abs(r.p1.x-r.p2.x)>=45 else NEUTRAL for s,r in raw.items()})
                for _ in range(4):advance({0:NEUTRAL,1:NEUTRAL})
            players={slot:CommandPlayer(compile_commands(text,confirm=name=='AAA')) for slot in range(2)}
            entry=dict(name=name,text=text,first_frame=frame+1,events=[],states=[],input_samples=[],native_action_states=[],
                distance=[abs(c.snapshot().latest.p1.x-c.snapshot().latest.p2.x) for c in clients.values()])
            report['cases'].append(entry)
            while not all(p.done for p in players.values()):
                keys={slot:p.next(frame,int(clients[slot].snapshot().latest.p1.facing),segment=1) for slot,p in players.items()}
                infos=advance(keys)
                for slot,player in players.items():
                    player.observe_native(infos[slot][AGENTS[0]]['native_combat_events'],0)
                entry['events'].extend(dict(slot=slot,**event) for slot in range(2)
                    for event in infos[slot][AGENTS[0]]['native_combat_events'] if event['kind']=='ACTION_TRANSITION')
                entry['input_samples'].extend(dict(slot=slot,**event) for slot in range(2)
                    for event in infos[slot][AGENTS[0]]['native_combat_events']
                    if event['kind']=='NATIVE_INPUT_SAMPLE' and event['attacker_seat']==0)
                entry['states'].append([dict(action=int(clients[s].snapshot().latest.p1.actionId),
                    sequence=int(clients[s].snapshot().latest.p1.sequenceId),
                    frame_flags=int(clients[s].snapshot().latest.p1.frameFlags),
                    hitstop=int(clients[s].snapshot().latest.p1.hitstop),
                    untech=int(clients[s].snapshot().latest.p1.untech)) for s in range(2)])
                entry['native_action_states'].append([NativeActionStateReader(memories[s]).read(0) for s in range(2)])
            entry['last_frame']=frame
            for _ in range(120):
                infos=advance({0:NEUTRAL,1:NEUTRAL})
                entry['events'].extend(dict(slot=slot,**event) for slot in range(2)
                    for event in infos[slot][AGENTS[0]]['native_combat_events'] if event['kind']=='ACTION_TRANSITION')
            entry['accepted_actions']={str(slot):[e['action_after'] for e in entry['events']
                if e['slot']==slot and e['attacker_seat']==0 and e['accepted']] for slot in range(2)}
        by_name={case['name']:case for case in report['cases']}
        checks={}
        for slot in range(2):
            actions=by_name['AAA']['accepted_actions'][str(slot)]
            checks[f'AAA_slot{slot}']=all(value in actions for value in (300,320,321))
            for name,expected in [('5B',400),('236B',560 if slot==0 else 500),
                                  ('623B',520 if slot==0 else 560)]:
                checks[f'{name}_slot{slot}']=expected in by_name[name]['accepted_actions'][str(slot)]
            samples=[e['native_inputs_before'][3] for e in by_name['charge2B']['input_samples'] if e['slot']==slot]
            checks[f'hold_release_slot{slot}']=samples==list(range(1,41))+[0,0]
        report['checks']=checks
        report['charge_semantics']={str(slot):'NATIVE_CHARGE_FLAG_OBSERVED' if any(
            state[slot]['charged_attack'] for state in by_name['charge2B']['native_action_states'])
            else 'NO_CHARGE_FLAG_OBSERVED_NOT_CLAIMED' for slot in range(2)}
        if not all(checks.values()):raise RuntimeError('command/counter acceptance failed')
        report.update(status='PASS',frames=frame,instances=2,
            scope='COMMAND_EXECUTION_NOT_ALL_CHAIN_LEGALITY',counts='OPT_IN_COMMUNITY_COUNTER_ADAPTER')
        if args.full_reward:
            report['reward_summaries']={str(slot):episode.reward.totals for slot,episode in env.episodes.items()}
            if not all(summary[0]['combo_extensions']>0 for summary in report['reward_summaries'].values()):
                raise RuntimeError('controlled AAA did not validate native-scope combo extensions')
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        for client in clients.values():client.close()
        for memory in memories.values():memory.close()
        env.close();report['seconds']=time.monotonic()-start
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
