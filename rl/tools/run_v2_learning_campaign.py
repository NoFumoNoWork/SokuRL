"""Evidence-driven V2 training blocks using existing train/benchmark entry points."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import hashlib
from omegaconf import OmegaConf


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=Path(os.environ['SOKURL_RL_ROOT'])
    source=args.source.resolve(strict=True)
    if not json.loads((source/'result.json').read_text())['success']:
        raise ValueError('source training must have completed')
    output=args.output.resolve()
    if args.resume:
        state=json.loads((output/'status.json').read_text())
        if state['source']!=str(source):raise ValueError('resume source differs')
        shutil.copy2(output/'status.json',output/f'status-before-resume-{time.time_ns()}.json')
        state.update(status='RUNNING');state.pop('error',None)
    else:
        output.mkdir(parents=True,exist_ok=False)
        state=dict(status='RUNNING',source=str(source),stages=[],decisions=[],best_training=str(source))
    previous_seconds=state.get('seconds',0.)
    started=time.monotonic()
    def save():
        tmp=output/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
        tmp.replace(output/'status.json')
    def run(name,tool,*parameters):
        command=[sys.executable,'-B',str(root/'run.py'),tool,*parameters]
        previous=next((row for row in state['stages'] if row['name']==name and row['status']=='PASS'),None)
        if previous:
            if previous['command']!=command:raise RuntimeError(f'{name}: cached command differs')
            print(name,'REUSE_PASS',flush=True)
            return
        row=dict(name=name,status='RUNNING',started=time.time(),
                 command=command)
        state['stages'].append(row);save()
        print(name,'RUNNING',flush=True)
        with (output/f'{name}.log').open('x',encoding='utf-8') as log:
            result=subprocess.run(row['command'],stdout=log,stderr=subprocess.STDOUT)
        row.update(status='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,
                   seconds=time.time()-row['started']);save()
        if result.returncode:raise RuntimeError(f'{name} failed; retained log and checkpoints')
        print(name,'PASS',flush=True)
    def evaluate(name,training,checkpoint='final.zip',opponents='all',heldout=False):
        original_name=name
        for attempt in range(4):
            failed=any(row['name']==name and row['status']=='FAIL' for row in state['stages'])
            passed=any(row['name']==name and row['status']=='PASS' for row in state['stages'])
            if not failed or passed:break
            name=f'{original_name}-retry-{attempt+1}'
        else:
            raise RuntimeError(f'{original_name}: too many failed evaluations; inspect native runtime')
        destination=output/name
        migrated=False
        previous_pass=next((row for row in state['stages'] if row['name']==name and row['status']=='PASS'),None)
        legacy_pass=previous_pass and 'require_complete=false' not in previous_pass['command']
        if heldout and not legacy_pass:
            from soku_rl.rl.timing_migration import migrate_checkpoint
            model,config=migrate_checkpoint(training/checkpoint,training/'config.yaml',
                output/'timing-migrations'/name)
            training,checkpoint=model.parent,model.name
            migrated=True
        options=['--config-name','benchmark_combat_v2_control',
            f'training_directory={training.as_posix()}',f'checkpoint={checkpoint}',
            f'opponent_names={opponents}',f'output={destination.as_posix()}']
        if heldout:
            options.append('benchmark.world_seeds=[630281921,1416067620,228490115,3072144481,1924058730,836250417,2671095488,411028306]')
        if migrated:
            # A validated capacity migration is not a new completed training run.
            options.append('require_complete=false')
        run(name,'benchmark_br',*options)
        data=json.loads((destination/'result.json').read_text())
        if not data['success']:raise RuntimeError('incomplete evaluation')
        return data['result']['games']
    def metrics(games,opponent):
        group=[g for g in games if g['opponent']==opponent]
        wins=sum(g['outcome']==('p1_win' if g['learner_seat']==0 else 'p2_win') for g in group)
        rows=[g['combat_reward_summary_by_seat'][g['learner_seat']] for g in group]
        return dict(wins=wins,games=len(group),win_rate=wins/len(group),
            damage_balance=sum(r['damage_dealt']-r['damage_taken'] for r in rows)/len(rows),
            extensions=sum(r['combo_extensions'] for r in rows)/len(rows),
            unknown=sum(r['unknown_lineage_hits'] for r in rows))
    try:
        save()
        initial=evaluate('screen-initial',source,'initial.zip')
        baseline=evaluate('screen-65k',source)
        names=['rush','community_combo','pressure','bullet_wall']
        screen={name:dict(initial=metrics(initial,name),current=metrics(baseline,name)) for name in names}
        state['screen']=screen
        choices=[n for n in names if n!='rush' and screen[n]['current']['win_rate']<1]
        if not choices:choices=['pressure']
        focus=min(choices,key=lambda n:(abs(screen[n]['current']['win_rate']-.5),names.index(n)))
        state['focus']=focus
        state['decisions'].append(dict(reason='prefer non-saturated opponent near 50% descriptive win rate',focus=focus))
        best=source;best_metrics=metrics(baseline,focus);failures=0;low_lr=False
        save()
        control=None
        for block in range(1,5):
            if time.monotonic()-started>7200:raise RuntimeError('two-hour campaign safety limit; completed evidence retained')
            training=output/f'train-{block}'
            target_lr=0.00003 if low_lr else 0.0001
            previous_config=OmegaConf.load(best/'config.yaml')
            source_kind='checkpoint' if previous_config.rl.ppo.learning_rate==target_lr else 'weights'
            goal=(json.loads((best/'result.json').read_text())['result']['steps'] if source_kind=='checkpoint' else 0)+65536
            candidates=[training,*sorted(output.glob(f'train-{block}-retry-*'))]
            completed=[p for p in candidates if (p/'result.json').exists()
                       and json.loads((p/'result.json').read_text()).get('success') is True]
            budget=65536
            source_path=best/'final.zip';source_config=best/'config.yaml'
            training_seed=20261015+block
            if completed:
                training=completed[-1]
            elif training.exists():
                checkpoints=[p for directory in candidates for p in (directory/'checkpoints').glob('updated_*_steps.zip')]
                if not checkpoints:raise RuntimeError('failed block has no completed-update checkpoint')
                checkpoint=max(checkpoints,key=lambda p:int(p.stem.split('_')[1]))
                steps=int(checkpoint.stem.split('_')[1]);budget=goal-steps
                if budget<=0:raise RuntimeError('failed block reached target; inspect before recovery')
                retry=len(candidates)
                if retry>3:raise RuntimeError('three failed attempts; investigate native environment')
                training=output/f'train-{block}-retry-{retry}'
                source_path=checkpoint;source_config=checkpoint.parent.parent/'config.yaml';source_kind='checkpoint'
                training_seed+=10000*retry
                recovery=dict(block=block,checkpoint=str(checkpoint),
                    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    checkpoint_steps=steps,target_steps=goal,remaining=budget,
                    destination=str(training),seed=training_seed,
                    reason='resume completed PPO update; retain failed trace and restart native environments')
                state.setdefault('recoveries',[]).append(recovery);save()
            if not completed:
                from soku_rl.rl.timing_migration import migrate_checkpoint
                source_path,source_config=migrate_checkpoint(source_path,source_config,
                    output/'timing-migrations'/training.name)
            options=['--config-name','train_combat_v2_specialist','algorithm.timesteps=65536',
                f'seed={training_seed}',
                f'algorithm.opponents.0.name={focus}',f'algorithm.opponents.0.policy.name={focus}',
                f'algorithm.initial_policy.kind={source_kind}',
                f'+algorithm.initial_policy.path={source_path.as_posix()}',
                f'+algorithm.initial_policy.training_config={source_config.as_posix()}',
                f'output={training.as_posix()}']
            options[2]=f'algorithm.timesteps={budget}'
            options.append('+rl.ppo.policy_kwargs.features_extractor_kwargs.timing_vocab_size=8192')
            if low_lr:options.append('rl.ppo.learning_rate=0.00003')
            if not completed:run(training.name,'train',*options)
            if block==4:control=training
            actual=OmegaConf.load(training/'config.yaml')
            games=evaluate(f'eval-{block}',training,opponents=f'[{focus},rush]')
            current=metrics(games,focus)
            improved=(current['wins'],current['damage_balance'])>(best_metrics['wins'],best_metrics['damage_balance'])
            state['decisions'].append(dict(block=block,focus=focus,learning_rate=float(actual.rl.ppo.learning_rate),
                training_seed=int(actual.seed),source_kind=str(actual.algorithm.initial_policy.kind),
                player=str(actual.algorithm.player),training=str(training),
                previous=best_metrics,current=current,rush=metrics(games,'rush'),
                selected=improved,meaning='descriptive development selection,not statistical proof'))
            if improved:
                best,best_metrics=training,current;failures=0
            else:failures+=1
            state['best_training']=str(best);save()
            if failures>=2:low_lr=True
        reference=evaluate('heldout-reference',source,opponents=f'[{focus},rush]',heldout=True)
        if best==source:
            candidate=reference
            state['heldout_identical_model_reused']=True
        else:
            candidate=evaluate('heldout-selected',best,opponents=f'[{focus},rush]',heldout=True)
        state['heldout']={n:dict(reference=metrics(reference,n),selected=metrics(candidate,n)) for n in (focus,'rush')}
        # Pre-registered before heldout: retain block4 seed/initial weights and
        # sampling budget, vary only GAE. Reward telemetry is now opt-in by data.
        credit=output/'train-credit'
        control_config=OmegaConf.load(control/'config.yaml')
        if not (credit/'result.json').exists():
            run('train-credit','train','--config-name','train_combat_v2_credit',
                'algorithm.timesteps=65536',f'seed={int(control_config.seed)}',
                f'algorithm.opponents.0.name={focus}',f'algorithm.opponents.0.policy.name={focus}',
                'algorithm.initial_policy.kind=weights',
                f'+algorithm.initial_policy.path={(control/"initial.zip").as_posix()}',
                f'+algorithm.initial_policy.training_config={(control/"config.yaml").as_posix()}',
                f'output={credit.as_posix()}')
        credit_result=json.loads((credit/'result.json').read_text())
        if not credit_result.get('success'):raise RuntimeError('credit experiment failed; inspect retained evidence')
        original=json.loads((control/'result.json').read_text())
        if credit_result['result']['initial_policy_hash']!=original['result']['initial_policy_hash']:
            raise RuntimeError('credit experiment initial weights differ from control')
        development=evaluate('eval-credit',credit,opponents=f'[{focus},rush]')
        heldout_credit=evaluate('heldout-credit',credit,opponents=f'[{focus},rush]',heldout=True)
        state['credit_test']=dict(control=str(control),candidate=str(credit),
            meaning='same initial weights/seed/budget, GAE.95 vs.99; one training seed, no automatic promotion',
            development={n:metrics(development,n) for n in (focus,'rush')},
            heldout={n:metrics(heldout_credit,n) for n in (focus,'rush')})
        initial_control=evaluate('heldout-initial',source,'initial.zip',opponents=f'[{focus}]',heldout=True)
        state['heldout_initial']={focus:metrics(initial_control,focus)}
        state.update(status='PASS',meaning='bounded adaptive campaign complete; paired heldout evidence,not automatic model promotion')
    except BaseException as error:
        state.update(status='FAIL',error=repr(error));raise
    finally:
        state['seconds']=previous_seconds+time.monotonic()-started;save()


if __name__=='__main__':main()
