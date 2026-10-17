"""Formal continuation uses full masked denominators and an explicit budget."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from soku_rl.rl.hierarchical_training import evaluate, materialize, validate_budget, epoch_order, restore_optimizer


class FixedActor:
    def eval(self):return self

    def __call__(self, batch, **kwargs):
        n = len(batch['state'])
        return dict(action=torch.tensor([[2.,0.]]).repeat(n,1),
                    termination=torch.ones(n),
                    sequence=torch.tensor([[[2.,0.]]]).repeat(n,3,1))


def test_full_validation_mask_denominators_and_free_sequence():
    data = dict(state=np.zeros((5,185), np.float32),
        target_action=np.array([0,1,1,0,-1]),
        target_action_mask=np.array([1,1,0,1,0],bool),
        target_termination=np.array([1,0,1,0,0]),
        target_termination_mask=np.array([1,1,0,0,0],bool),
        target_sequence=np.tile([0,1,-1],(5,1)),
        target_sequence_mask=np.tile([1,1,0],(5,1)).astype(bool))
    catalog=[dict(continuous=True),dict(continuous=False)]
    a=evaluate(FixedActor(),data,catalog,2,'cpu')
    b=evaluate(FixedActor(),data,catalog,5,'cpu')
    assert a['action']['count']==3 and a['action']['accuracy']==2/3
    assert a['attack']['count']==1 and a['movement']['count']==2
    assert a['termination']['count']==2
    assert a['stop_confusion']==dict(tp=1,fp=1,tn=0,fn=0)
    assert a['sequence_free_running']['count']==10
    assert a['always_continue_accuracy']==.5
    assert a['attack_confusion']==dict(tp=0,fp=0,tn=2,fn=1)
    assert a['constant_majority_accuracy']==2/3
    assert a['balanced_action_accuracy']==.5
    assert a['selection_loss']==pytest.approx(b['selection_loss'],rel=1e-6)


def test_materialize_retains_all_rows_and_clears_cache():
    class Dataset:
        ends=[2,5]
        cache={0:True}
        def batch(self,indices):return dict(state=np.asarray(indices))
    d=Dataset()
    assert materialize(d)['state'].tolist()==list(range(5))
    assert d.cache=={}


def test_bounded_training_profile():
    cfg=OmegaConf.to_container(OmegaConf.load('rl/config/hierarchical_training.yaml'))
    validate_budget(cfg)
    for key,value in [('epochs',11),('max_updates',5001),('learning_rate',.001),('sequence_weight',1.),
                      ('optimizer','resume_adam_from_bound_selected_checkpoint')]:
        invalid=dict(cfg);invalid[key]=value
        with pytest.raises(ValueError):validate_budget(invalid)


def test_balanced_choices_preserve_context_masks_and_epoch_size():
    data=dict(target_action=np.array([1]+[0]*9+[-1]*10),
              target_action_mask=np.array([1]*10+[0]*10,bool))
    before={k:v.copy() for k,v in data.items()}
    catalog=[dict(continuous=True),dict(continuous=False)]
    order,stats=epoch_order(data,catalog,np.random.default_rng(7),'balanced_attack_movement')
    assert len(order)==20
    assert stats['draws']==dict(attack=5,movement=5,context=10)
    assert stats['unique']==dict(attack=1,movement=5,context=10)
    assert sorted(order[~data['target_action_mask'][order]])==list(range(10,20))
    assert all(np.array_equal(data[k],v) for k,v in before.items())
    again,_=epoch_order(data,catalog,np.random.default_rng(7),'balanced_attack_movement')
    assert np.array_equal(order,again)


def test_uniform_sampler_preserves_original_permutation_and_unknown_is_not_attack():
    data=dict(target_action=np.array([0,1,1,-1]),target_action_mask=np.array([1,1,0,0],bool))
    catalog=[dict(continuous=True),dict(continuous=False)]
    order,stats=epoch_order(data,catalog,np.random.default_rng(3))
    assert np.array_equal(order,np.random.default_rng(3).permutation(4))
    assert stats['draws']==dict(attack=1,movement=1,context=2)
    invalid=dict(data,target_action_mask=np.array([1,0,0,0],bool))
    with pytest.raises(ValueError):epoch_order(invalid,catalog,np.random.default_rng(3),'balanced_attack_movement')


def test_optimizer_restore_binds_checkpoint_sidecar_and_preserves_next_update(tmp_path):
    import json
    from soku_rl.replay.human_dataset import sha256
    torch.manual_seed(13)
    first=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(first.parameters(),lr=3e-5)
    x=torch.tensor([[1.,2.]])
    def update(model,opt):
        opt.zero_grad();model(x).square().sum().backward();opt.step()
    update(first,optimizer)
    checkpoint=tmp_path/'epoch.json';checkpoint.write_text('{}')
    sidecar=checkpoint.with_suffix('.adam.pt');torch.save(optimizer.state_dict(),sidecar)
    report=tmp_path/'result.json'
    report.write_text(json.dumps(dict(status='PASS_BOUNDED_SUPERVISED_TRAINING',best=dict(
        path=str(checkpoint),sha256=sha256(checkpoint),optimizer_sha256=sha256(sidecar)))))
    second=torch.nn.Linear(2,1);second.load_state_dict(first.state_dict())
    resumed=torch.optim.Adam(second.parameters(),lr=3e-5)
    result=restore_optimizer(resumed,checkpoint,report,3e-5)
    assert result['minimum_step']==result['maximum_step']==1
    update(first,optimizer);update(second,resumed)
    for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError):restore_optimizer(resumed,checkpoint,report,1e-3)
    sidecar.write_bytes(sidecar.read_bytes()+b'tampered')
    with pytest.raises(ValueError):restore_optimizer(resumed,checkpoint,report,3e-5)
