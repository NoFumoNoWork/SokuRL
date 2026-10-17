"""Bounded budget and inference-consistent natural validation, no game access."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from soku_rl.rl.factorized_training import validate_config
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
