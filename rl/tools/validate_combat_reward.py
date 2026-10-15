"""Settle saved native contacts, explicitly distinguishing full/partial profiles."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from soku_rl.env.reward import CombatReward, CombatRewardConfig
from soku_rl.pomg import Outcome


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--combo-beta',type=float,default=.3)
    parser.add_argument('--passive-b',type=float,default=.02)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('preserve existing reward evidence')
    config=CombatRewardConfig(combo_beta=args.combo_beta,passive_b=args.passive_b)
    data=json.loads(args.capture.read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for event in data['events']:groups[event['frame']].append(event)
    segment=next(iter(data['events']))['segment']
    def timestep(frame,hp,events):
        return SimpleNamespace(frame=frame,diagnostics=dict(segment=segment,hp=hp,native_combat_events=events),
            outcome=Outcome.ONGOING,terminated=False,truncated=False,ended=False)
    ledger=CombatReward(config)
    ledger.reset(timestep(0,(10000,10000),()))
    report=dict(status='RUNNING',source_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        config=config.__dict__,profile='FULL_STRICT' if config.combo_beta and config.passive_b else 'PARTIAL_EXPLICIT',
        settled_frames=0,totals=[0.,0.])
    try:
        for state in data['states']:
            rewards,parts=ledger.step(timestep(state['frame'],state['hp'],groups[state['frame']]))
            report['settled_frames']+=1
            for seat,value in enumerate(rewards):report['totals'][seat]+=value
        report['status']='PASS'
    except NotImplementedError as error:
        report.update(status='BLOCKED',reason=str(error))
    except Exception as error:
        report.update(status='FAIL',reason=repr(error))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(2)


if __name__=='__main__':main()
