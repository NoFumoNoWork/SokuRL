"""Bounded budget and inference-consistent natural validation, no game access."""
import numpy as np
import json
import pytest
import torch
from omegaconf import OmegaConf

from soku_rl.rl.factorized_training import validate_config, restore_continuation
from soku_rl.rl.semantic_sampling import stratified_order
from soku_rl.replay.human_dataset import sha256
from soku_rl.rl.hierarchical_training import evaluate


def config():
    return OmegaConf.to_container(OmegaConf.load('rl/config/hierarchical_factorized_training.yaml'),resolve=True)


def test_budget_is_finite_and_fixed_objective():
    cfg=config();validate_config(cfg)
    assert cfg['cycles']*cfg['draws_per_cycle']==40960
    assert cfg['max_updates']==640
    for key,value in [('cycles',11),('max_updates',639),('termination_weight',.5),
                      ('future_weight',.5),('learning_rate',.001),('row_cap',3)]:
        bad=dict(cfg);bad[key]=value
        with pytest.raises(ValueError):validate_config(bad)


class GreedyHierarchy:
    def eval(self):return self
    def __call__(self,x,**kwargs):
        n=len(x['state'])
        return dict(action=torch.tensor([[2.,0.]]).repeat(n,1),choice_id=torch.ones(n,dtype=torch.long),
            sequence=torch.tensor([[[2.,0.]]]).repeat(n,3,1),
            sequence_choice_ids=torch.ones(n,3,dtype=torch.long),termination=torch.zeros(n))


def test_metrics_use_actual_hierarchical_decisions_not_joint_map():
    data=dict(state=np.zeros((3,185),np.float32),target_action=np.ones(3,np.int64),
        target_action_mask=np.array([1,1,0],bool),target_termination=np.ones(3),
        target_termination_mask=np.array([1,0,0],bool),target_sequence=np.ones((3,3),np.int64),
        target_sequence_mask=np.array([[1,1,0],[0,0,0],[0,0,0]],bool))
    weights=torch.tensor([1.,4.]);catalog=[dict(continuous=True),dict(continuous=False)]
    a=evaluate(GreedyHierarchy(),data,catalog,2,'cpu',task_weights=(.3,.15),ce_weights=weights)
    b=evaluate(GreedyHierarchy(),data,catalog,3,'cpu',task_weights=(.3,.15),ce_weights=weights)
    assert a['action']['accuracy']==a['sequence_free_running']['accuracy']==1
    assert a['action']['weighted_nll']==pytest.approx(4*a['action']['nll'])
    assert a['predicted_commands']=={'1':2}
    assert a['selection_loss']==pytest.approx(a['action']['weighted_nll']+
        .3*a['termination']['nll']+.15*a['sequence_teacher_forced']['weighted_nll'])
    assert a['selection_loss']==pytest.approx(b['selection_loss'])


def test_resume_config_is_explicit_and_default_remains_fresh():
    cfg=config()
    assert cfg['optimizer']=='fresh_adam'
    cfg['resume_training_report']='report.json'
    with pytest.raises(ValueError,match='together'):validate_config(cfg)
    cfg['optimizer']='resume_adam_from_bound_selected_checkpoint'
    validate_config(cfg)
    del cfg['resume_training_report']
    with pytest.raises(ValueError,match='together'):validate_config(cfg)


@pytest.mark.parametrize('bound_rng',[False,True])
def test_continuation_exact_adam_and_sampler_next_step(tmp_path,bound_rng):
    first=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(first.parameters(),lr=3e-5)
    def update(model,opt):
        opt.zero_grad();model(torch.ones(2,2)).square().sum().backward();opt.step()
    update(first,optimizer)
    checkpoint=tmp_path/'checkpoint.json';checkpoint.write_text('{}')
    data_file=tmp_path/'data.json';data_file.write_text('{}')
    sidecar=checkpoint.with_suffix('.adam.pt');torch.save(optimizer.state_dict(),sidecar)
    cfg=config();cfg.update(data=str(data_file),draws_per_cycle=2,batch_size=2,
        initial_checkpoint=str(checkpoint),optimizer='resume_adam_from_bound_selected_checkpoint',
        resume_training_report=str(tmp_path/'result.json'))
    train=dict(supervision_strata=np.tile([True,False,False,False],(4,1)),
        replay_id=np.arange(4),chain_id=np.full(4,-1))
    rng=np.random.default_rng(cfg['seed'])
    order,roles,_=stratified_order(train['supervision_strata'],train['replay_id'],
        train['chain_id'],rng,2,row_cap=2,replay_cap=256,chain_cap=32)
    np.savez(tmp_path/'sampling-01.npz',indices=order,roles=roles)
    best=dict(path=str(checkpoint),sha256=sha256(checkpoint),optimizer_sha256=sha256(sidecar),cycle=1)
    if bound_rng:best['sampler_rng_state']=rng.bit_generator.state
    previous=dict(status='PASS_BOUNDED_FACTORIZED_TRAINING',config=cfg,best=best,
        data_sha256=sha256(data_file),class_weights=[1.,1.],cycles=[dict(cycle=1,updates=1)])
    report=tmp_path/'result.json';report.write_text(json.dumps(previous))
    second=torch.nn.Linear(2,1);second.load_state_dict(first.state_dict())
    resumed=torch.optim.Adam(second.parameters(),lr=3e-5)
    next_rng=np.random.default_rng(cfg['seed'])
    result=restore_continuation(resumed,cfg,train,torch.ones(2),next_rng)
    assert result['minimum_step']==result['maximum_step']==result['starting_updates']==1
    assert next_rng.bit_generator.state==rng.bit_generator.state
    update(first,optimizer);update(second,resumed)
    for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match='weights changed'):
        restore_continuation(resumed,cfg,train,torch.tensor([1.,2.]),next_rng)
    if not bound_rng:
        np.savez(tmp_path/'sampling-01.npz',indices=order+1,roles=roles)
        with pytest.raises(ValueError,match='reproduction'):
            restore_continuation(resumed,cfg,train,torch.ones(2),np.random.default_rng(cfg['seed']))
