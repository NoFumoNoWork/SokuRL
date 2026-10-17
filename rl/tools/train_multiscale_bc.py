"""Bounded A/B/C/D BC using accepted human prefixes, never a new Hook."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
from omegaconf import OmegaConf
import torch

from soku_rl.replay.human_dataset import policy_interface, save_json, sha256
from soku_rl.replay.human_windows import WindowDataset
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.human_cloning import HumanDataset,HumanCollection
from soku_rl.rl.learner import create_learner, parameter_hash
from soku_rl.rl.multiscale_bc import SequenceActor, tensor_batch, sequence_loss, horizon_losses, score, load_actor


def fit(variant, config, source, train, validation, output, device, seed, epochs, batch_size,
        future_weight=1.,initial_artifact=None):
    directory = output/variant
    directory.mkdir()
    OmegaConf.save(OmegaConf.create(config), directory/'config.yaml')
    interface = policy_interface(config,(1,0))
    model, source_identity = create_learner(ObservationContractEnv(interface),interface,config['rl'],
        dict(kind='weights',path=str(source/'final.zip'),training_config=str(source/'config.yaml')),device,seed)
    from stable_baselines3 import PPO
    original = PPO.load(source/'final.zip',device='cpu')
    initial_hash = parameter_hash(model.policy)
    if initial_hash != parameter_hash(original.policy) or model.policy.optimizer.state:
        raise ValueError('exact source weights and fresh optimizer required')
    torch.manual_seed(seed)
    actor = SequenceActor(model.policy,variant).to(device)
    warm_actor = None
    initialization = dict(kind='original_ppo_weights',sha256=sha256(source/'final.zip'))
    if initial_artifact:
        from soku_rl.policy.contract import read_training_contract
        initial_artifact = Path(initial_artifact).resolve(strict=True)
        read_training_contract(initial_artifact.parent/'config.yaml',interface)
        _,warm_actor,_ = load_actor(initial_artifact,device)
        if variant not in ('C','D') or warm_actor.horizon!=5:
            raise ValueError('typed full-context warmstart requires matching5-head architecture')
        actor.load_state_dict(warm_actor.state_dict(),strict=True)
        if parameter_hash(actor)!=parameter_hash(warm_actor):
            raise ValueError('full actor/context/future warmstart failed')
        initial_hash = parameter_hash(model.policy)
        initialization = dict(kind='typed_bc_full_weights',path=str(initial_artifact),
            sha256=sha256(initial_artifact),actor_state_sha256=parameter_hash(actor),
            optimizer='NEW Adam;prior BC optimizer moments were not saved')
    optimizer = torch.optim.Adam(actor.parameters(),lr=config['rl']['ppo']['learning_rate'])
    before = {k:v.detach().cpu().clone() for k,v in actor.named_parameters()}
    report = dict(status='RUNNING',variant=variant,source=source_identity,seed=seed,
        initial_policy_parameter_sha256=initial_hash,initialization=initialization,train_anchors=train.count,
        validation_anchors=validation.count,anchor_identity=train.identity,
        epochs=epochs,batch_size=batch_size,learning_rate=config['rl']['ppo']['learning_rate'],
        optimizer='fresh Adam',value_coef=0,action_change_weight=1,decision_frames=1,
        target_horizon=actor.horizon,updates=0,history=[],best_epoch=0,
        objective='L0+lambda/4*sum(L1..L4)' if variant=='D' else 'legacy_pooled_mean',
        future_weight=future_weight if variant=='D' else None,
        loss_coefficients=[1.,*[future_weight/4]*4] if variant=='D' else [1./actor.horizon]*actor.horizon,
        mask_reduction='Per-horizon valid-label mean;empty future horizon0,no redistribution.' if variant=='D' else 'Pooled valid-label mean.',
        budget_note='Same anchors/order/updates;C has up to5 labels/anchor,not equal label exposure.',
        limitation='Opening prefixes only;context ZIP alone is NOT an equivalent deployable policy.')
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    try:
        initial_data = train.batch(np.arange(min(128,train.count)))
        actor.eval()
        with torch.no_grad():
            x = tensor_batch(initial_data,device)
            p = actor.logits(x)[:,0].softmax(-1)
            reference = (warm_actor.logits(x)[:,0].softmax(-1) if warm_actor else
                         model.policy.get_distribution(x['observations']).distribution.probs)
            delta = float((p-reference).abs().max())
            if delta > 1e-6:
                raise ValueError('initial current actor no longer matches source')
        report['initial_probability_max_error'] = delta
        report['initial_probability_sha256'] = __import__('hashlib').sha256(p.cpu().numpy().tobytes()).hexdigest()
        actor.save(model,directory,'initial')
        metrics = score(actor,validation,device,batch_size)
        report['history'].append(dict(epoch=0,validation=metrics))
        best = metrics['horizons'][0]['nll']
        actor.save(model,directory,'best')
        for epoch in range(1,epochs+1):
            actor.train()
            order = rng.permutation(train.count)
            losses,norms,per_horizon = [],[],[]
            for start in range(0,train.count,batch_size):
                data = train.batch(order[start:start+batch_size])
                h = actor.horizon
                logits = actor.logits(tensor_batch(data,device))
                labels = torch.as_tensor(data['labels'][:,:h],device=device,dtype=torch.long)
                mask = torch.as_tensor(data['mask'][:,:h],device=device,dtype=torch.bool)
                changed = torch.as_tensor(data['changed'][:,:h],device=device,dtype=torch.bool)
                loss = sequence_loss(logits,labels,mask,changed,
                                     future_weight=future_weight if variant=='D' else None)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite BC loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(actor.parameters(),model.max_grad_norm,error_if_nonfinite=True)
                optimizer.step()
                report['updates'] += 1
                losses.append(float(loss.detach()));norms.append(float(norm))
                per_horizon.append(horizon_losses(logits.detach(),labels,mask,changed).cpu().numpy())
            metrics = score(actor,validation,device,batch_size)
            report['history'].append(dict(epoch=epoch,mean_batch_loss=float(np.mean(losses)),
                mean_batch_horizon_losses=np.mean(per_horizon,axis=0).tolist(),
                gradient_norm_mean=float(np.mean(norms)),validation=metrics,
                seconds=time.perf_counter()-started))
            current = metrics['horizons'][0]['nll']
            if current < best:
                best = current; report['best_epoch'] = epoch
                actor.save(model,directory,'best')
            if epoch%5==0:
                actor.save(model,directory,f'epoch-{epoch:04d}')
            save_json(directory/'result.json',report)
            print(f'{variant} epoch={epoch} val0={current:.5f} val_change={metrics["horizons"][0]["changed_nll"]:.5f}',flush=True)
        actor.save(model,directory,'final')
        report['parameter_delta_l2'] = {k:float((v.detach().cpu()-before[k]).norm()) for k,v in actor.named_parameters()}
        _,best_actor,_ = load_actor(directory/'best.json',device)
        report['best_validation'] = score(best_actor,validation,device,batch_size)
        report['best_train'] = score(best_actor,train,device,batch_size)
        report['interventions'] = {name:score(best_actor,validation,device,batch_size,keys)
            for name,keys in [('permute_inputs',('inputs',)),('permute_states',('recent','distant'))]} if variant!='A' else {}
        data = validation.batch(np.arange(min(256,validation.count)))
        batch = tensor_batch(data,device)
        with torch.no_grad():
            actual = best_actor.logits(batch)[:,0].softmax(-1)
            if variant!='A':
                changed = dict(batch,inputs=batch['inputs'].flip(1))
                altered = best_actor.logits(changed)[:,0].softmax(-1)
                report['input_order_effect'] = dict(mean_abs=float((actual-altered).abs().mean()),
                                                   max_abs=float((actual-altered).abs().max()))
            original.policy.set_training_mode(False)
            source_probabilities = (warm_actor.logits(batch)[:,0].softmax(-1) if warm_actor else
                original.policy.get_distribution(torch.as_tensor(data['observations'],device='cpu')).distribution.probs.to(device))
            report['best_vs_initial_probability_l1'] = float((actual-source_probabilities).abs().sum(-1).mean())
        report.update(status='PASS_OFFLINE_NOT_POLICY_IMPROVEMENT',seconds=time.perf_counter()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
            best_artifact_sha256=sha256(directory/'best.json'))
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        save_json(directory/'result.json',report)
    return report


def baselines(dataset, train):
    truth,mask = dataset.data['labels'],dataset.data['mask']
    previous = dataset.data['previous_action']
    counts = np.bincount(train.data['labels'][:,0],minlength=90).astype(np.float64)+.5
    probabilities = counts/counts.sum()
    eligible = mask & (previous[:,None]>=0)
    return dict(constant_action=int(counts.argmax()),
        horizons=[dict(offset=j,class_prior_nll=float(-np.log(probabilities[truth[mask[:,j],j]]).mean()),
            constant_accuracy=float((truth[mask[:,j],j]==counts.argmax()).mean()),
            repeat_previous_accuracy=float(((truth[:,j]==previous)&eligible[:,j]).sum()/eligible[:,j].sum()),
            repeat_eligible=int(eligible[:,j].sum())) for j in range(5)])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--seed',type=int,default=20261017)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--variants',choices=('A','B','C','D'),nargs='+',default=['A','B','C'])
    parser.add_argument('--future-weight',type=float,default=1.)
    parser.add_argument('--train-collection',type=Path)
    parser.add_argument('--initial-artifact',type=Path)
    parser.add_argument('--max-anchor-visits',type=int,default=1000000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output,source = args.output.resolve(),args.source.resolve(strict=True)
    if (not output.is_relative_to(root/'rl/runs') or output.exists() or not 1<=args.epochs<=20
            or not np.isfinite(args.future_weight) or args.future_weight<0
            or not 1<=args.max_anchor_visits<=2000000
            or len(set(args.variants))!=len(args.variants)):
        raise ValueError('new project-local output,at most20epochs and explicit finite budget required')
    accepted_train = (HumanCollection(args.train_collection) if args.train_collection else
                      HumanDataset(args.data/'pilot-20/manifest.json',args.data/'pilot-acceptance-v2.json'))
    accepted_val = HumanDataset(args.data/'validation-20/manifest.json',args.data/'validation-acceptance.json')
    if accepted_train.manifest['partition']!='train' or accepted_val.manifest['partition']!='validation':
        raise ValueError('training and validation partitions cannot be interchanged')
    for key in ('replay_id','source_sha256','set_id'):
        if {x[key] for x in accepted_train.manifest['replays']} & {x[key] for x in accepted_val.manifest['replays']}:
            raise ValueError('partition leakage')
    if set(accepted_train.acceptance['players']) & set(accepted_val.acceptance['players']):
        raise ValueError('shared human player across partitions')
    for accepted in (accepted_train,accepted_val):
        if accepted.manifest['identity']['source_checkpoint_sha256'] != sha256(source/'final.zip'):
            raise ValueError('source checkpoint differs from accepted contract')
    train,validation = WindowDataset(accepted_train),WindowDataset(accepted_val)
    if train.count*args.epochs>args.max_anchor_visits:
        raise ValueError('small experiment budget exceeded')
    output.mkdir(parents=True)
    for name in ('temp','cuda-cache','torch-cache'):
        (output/name).mkdir()
    os.environ.update(TEMP=str(output/'temp'),TMP=str(output/'temp'),
                      CUDA_CACHE_PATH=str(output/'cuda-cache'),TORCH_HOME=str(output/'torch-cache'))
    config = OmegaConf.to_container(OmegaConf.load(source/'config.yaml'),resolve=True)
    config['runtime']['cwd'] = str(root)
    config['runtime']['command'] = [sys.executable,'-B',str(root/'rl/tools/rollout_worker.py')]
    config['output'] = str(output)
    config['algorithm']['initial_policy'] = dict(kind='weights',path=str(source/'final.zip'),
                                                training_config=str(source/'config.yaml'))
    config['human_bc'] = dict(variants=args.variants,future_weight=args.future_weight,
        initial_artifact=str(args.initial_artifact.resolve()) if args.initial_artifact else None,
        epochs=args.epochs,max_anchor_visits=args.max_anchor_visits)
    save_json(output/'data_audit.json',dict(status='PASS_CAUSAL_WINDOW_CONSTRUCTION',
        train_cases=train.cases,validation_cases=validation.cases,
        train_anchors=train.count,validation_anchors=validation.count,
        train_identity=train.identity,validation_identity=validation.identity,
        manifests={str(accepted.path):sha256(accepted.path) for accepted in (accepted_train,accepted_val)},
        own_input_source='Original consumed raw word of Marisa ownseat,all8 logical channels;no opponent input.',
        targets='Row i..i+4,individually supervised,within a complete compatible65-row window.',
        motion='Visible position differences only;no native velocity or inferred hit/graze events.',
        baselines=baselines(validation,train)))
    code = [p for part in ('rl/src','rl/tools','rl/config','platform/src','platform/tools')
            for p in sorted((root/part).rglob('*')) if p.is_file() and p.suffix in ('.py','.ps1','.yaml')]
    with zipfile.ZipFile(output/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path,path.relative_to(root).as_posix())
    from game_runtime.identity import fingerprints
    save_json(output/'identity.json',dict(source_sha256=sha256(source/'final.zip'),
        source_config_sha256=sha256(source/'config.yaml'),source_archive_sha256=sha256(output/'source.zip'),
        files={str(p.relative_to(root)):sha256(p) for p in code},runtime=fingerprints(),
        hub_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        hub_dirty=bool(subprocess.check_output(['git','status','--porcelain','-uno'],cwd=root,text=True).strip()),
        published_anchors=dict(platform='7bd1d6668984710dec1d5d8a4b883637a882a54e',RL='a5e634950df21300bb2888a74ebd4a4fd16aa4aa',
            note='Historical anchors;actual dirty sources preserved in hash-bound source.zip.')))
    torch.set_num_threads(4)
    results = {variant:fit(variant,config,source,train,validation,output,args.device,args.seed,args.epochs,args.batch_size,args.future_weight,args.initial_artifact)
               for variant in args.variants}
    if (len({r['initial_policy_parameter_sha256'] for r in results.values()})!=1
            or len({r['initial_probability_sha256'] for r in results.values()})!=1):
        raise ValueError('A/B/C initial current policies differ')
    save_json(output/'comparison.json',dict(status='PASS_OFFLINE_ONLY',results=results,
        native_evaluation='NOT TESTED',decision='No model promotion based on NLL alone.'))


if __name__=='__main__':
    main()
