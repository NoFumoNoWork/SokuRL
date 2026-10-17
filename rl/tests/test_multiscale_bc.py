"""No native launch: causal windows, masks, head order and live history parity."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from soku_rl.replay.human_windows import prepare_replay, window_inputs
from soku_rl.rl.multiscale_bc import SequenceActor, sequence_loss, tensor_batch
from soku_rl.policy.multiscale_bc import MultiscaleEpisode


def dense(n=90):
    observations = np.zeros((n,1544),np.float32)
    for i in range(n):
        frame = observations[i,1295:1480]
        frame[:10] = [1,i/640,0,1,1,400,0,0,i,0]
        frame[10:20] = [1,1-i/640,0,-1,0,410,0,0,i,0]
        frame[-5:] = [1,1,1,1,i/7200]
        observations[i,-8:] = [0,0,int((i-1)%2),0,0,0,0,0]
    return dict(observations=observations,actions=np.arange(n,dtype=np.int64)%2+40,
        supervised=np.ones(n,bool),compatible=np.ones(n,bool),boundary=np.zeros(n,bool),
        raw_words_present=np.ones(n,bool),raw_words=np.stack((np.arange(n)%2*16,np.ones(n)*32),1).astype(np.int64),
        observation_frames=np.arange(n),consumption_frames=np.arange(n)+1,seats=np.zeros(n,np.int64),
        round_before=np.zeros(n,np.int64),round_after=np.zeros(n,np.int64),
        changed=np.ones(n,bool),previous_action=np.full(n,40,np.int64))


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.share_features_extractor = True
        self.observation_space = SimpleNamespace(shape=(1544,))
        self.action_space = SimpleNamespace(n=90)
        self.features = nn.Linear(1544,128)
        self.mlp_extractor = SimpleNamespace(forward_actor=lambda x:x)
        self.action_net = nn.Linear(128,90)
        self.register_buffer('_blocked_actions',torch.zeros(90,dtype=torch.bool))
        self._blocked_actions[[5,6]] = True

    def extract_features(self,x):
        return self.features(x/8192)


def test_exact_time_and_order():
    data = prepare_replay(dense())
    assert data['anchors'][0]==60
    assert data['recent'].shape[1:] == (5,185)
    assert data['distant'].shape[1:] == (3,40)
    assert data['inputs'].shape[1:] == (60,8)
    np.testing.assert_allclose(data['recent'][0,:,1]*640,[40,45,50,55,60],atol=1e-4)
    np.testing.assert_array_equal(data['inputs'][0,:,2],np.arange(60)%2)
    np.testing.assert_array_equal(data['labels'][0],40+np.arange(60,65)%2)
    assert np.all(data['recent'][0,:-1,-5:-1]==0)
    assert data['recent'][0,-1,-5]==1


def test_future_input_and_state_cannot_leak():
    raw = dense()
    first = prepare_replay(raw)
    raw['raw_words'][60:,0] ^= 16
    raw['observations'][61:,:] = 2
    second = prepare_replay(raw)
    for key in ('observations','recent','distant','inputs'):
        np.testing.assert_array_equal(first[key][0],second[key][0])


def test_opponent_input_is_not_a_feature():
    raw = dense()
    before = prepare_replay(raw)
    raw['raw_words'][:,1] = 1023
    after = prepare_replay(raw)
    for key in ('observations','recent','distant','inputs'):
        np.testing.assert_array_equal(before[key],after[key])


def test_unrepresentable_history_is_preserved_and_future_mask_is_individual():
    raw = dense()
    raw['raw_words'][10,0] = 16+32+64+512
    raw['actions'][62] = -1
    raw['supervised'][62] = False
    output = prepare_replay(raw)
    np.testing.assert_array_equal(output['inputs'][0,10],[0,0,1,1,1,0,0,1])
    np.testing.assert_array_equal(output['mask'][0],[1,1,0,1,1])
    assert output['labels'][0,2]==-1


@pytest.mark.parametrize('bad', ['boundary','round','gap','missing','seat','incompatible'])
def test_cannot_cross_invalid_history(bad):
    raw = dense()
    if bad=='boundary': raw['boundary'][40]=True
    if bad=='round': raw['round_before'][40:]=raw['round_after'][40:]=1
    if bad=='gap':
        raw['observation_frames'][40:] += 1
        raw['consumption_frames'][40:] += 1
    if bad=='missing': raw['raw_words_present'][40]=False
    if bad=='seat': raw['seats'][40]=1
    if bad=='incompatible': raw['compatible'][40]=False
    assert len(prepare_replay(raw)['anchors'])==0


def test_current_source_actor_equal_and_future_heads_ordered():
    torch.set_num_threads(1)
    batch = tensor_batch(prepare_replay(dense()),'cpu')
    policy = SmallPolicy()
    for variant in 'ABC':
        actor = SequenceActor(policy,variant)
        reference = policy.action_net(policy.extract_features(batch['observations']))
        reference = reference.masked_fill(policy._blocked_actions,float('-inf'))
        torch.testing.assert_close(actor.logits(batch)[:,0],reference,rtol=0,atol=0)
        if variant=='C':
            with torch.no_grad(): actor.future[0].bias[0] += 3
            assert torch.equal(actor.logits(batch)[:,0],reference)
            assert not torch.equal(actor.logits(batch)[:,1],reference)
            torch.testing.assert_close(actor.logits(batch)[:,2],reference)


def test_future_mask_does_not_create_neutral_training_target():
    logits = torch.zeros(1,5,90,requires_grad=True)
    labels = torch.tensor([[40,41,-1,43,44]])
    mask = labels>=0
    loss = sequence_loss(logits,labels,mask,torch.zeros_like(mask))
    loss.backward()
    assert torch.count_nonzero(logits.grad[0,2])==0
    assert torch.count_nonzero(logits.grad[0,1])>0


def test_history_has_order_sensitive_gradient_after_zero_residual_warmup():
    torch.manual_seed(7)
    actor = SequenceActor(SmallPolicy(),'C')
    data = prepare_replay(dense())
    batch = tensor_batch(data,'cpu')
    optimizer = torch.optim.Adam(actor.parameters(),lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        sequence_loss(actor.logits(batch),torch.as_tensor(data['labels']),
                      torch.as_tensor(data['mask']),torch.as_tensor(data['changed'])).backward()
        optimizer.step()
    assert actor.context.inputs[0].weight.grad.abs().sum()>0
    altered = dict(batch,inputs=batch['inputs'].flip(1))
    assert not torch.equal(actor.logits(batch),actor.logits(altered))


def test_live_history_matches_offline_and_is_isolated():
    from collections import deque
    raw = dense()
    seen = []
    class RecordingActor:
        def logits(self,batch):
            seen.append({k:v.clone() for k,v in batch.items()})
            return torch.zeros(1,1,90)
    policy = SimpleNamespace(horizon=7200,device='cpu',actor=RecordingActor())
    episode = MultiscaleEpisode(policy,np.random.default_rng(1))
    for observation in raw['observations'][:61]:
        episode.act(observation)
    expected = prepare_replay(raw)
    for key in ('observations','recent','distant','inputs'):
        np.testing.assert_allclose(seen[-1][key][0].numpy(),expected[key][0])
    assert bool(seen[-1]['context_known'][0])
    fresh = MultiscaleEpisode(policy,np.random.default_rng(1))
    fresh.act(raw['observations'][0])
    assert not bool(seen[-1]['context_known'][0])
    with pytest.raises(ValueError,match='frame gap'):
        episode.act(raw['observations'][65])


def test_loss_has_no_teacher_forcing_and_uses_horizon_order():
    logits = torch.randn(2,5,90)
    labels = torch.tensor([[40,41,42,43,44],[41,42,43,44,45]])
    mask = torch.ones(2,5,dtype=torch.bool)
    assert sequence_loss(logits,labels,mask,mask) != sequence_loss(logits,labels.flip(1),mask,mask)


@pytest.mark.parametrize('weight',[0.,.25,1.,2.])
def test_c_current_and_future_formula_with_individual_masks(weight):
    logits = torch.randn(3,5,90,dtype=torch.float64,requires_grad=True)
    labels = torch.tensor([[40,41,42,43,44],[41,42,-1,44,45],[42,-1,-1,45,46]])
    mask = labels>=0
    expected = [nn.functional.cross_entropy(logits[mask[:,j],j],labels[mask[:,j],j]) for j in range(5)]
    actual = sequence_loss(logits,labels,mask,torch.zeros_like(mask),future_weight=weight)
    torch.testing.assert_close(actual,expected[0]+weight/4*sum(expected[1:]),rtol=1e-12,atol=1e-12)


def test_c_current_gradient_is_four_times_each_future_at_lambda_one():
    logits = torch.zeros(2,5,90,dtype=torch.float64,requires_grad=True)
    labels = torch.full((2,5),40)
    mask = torch.ones(2,5,dtype=torch.bool)
    sequence_loss(logits,labels,mask,mask,future_weight=1.).backward()
    for j in range(1,5):
        torch.testing.assert_close(logits.grad[:,0],4*logits.grad[:,j],rtol=0,atol=0)


def test_c_empty_future_does_not_redistribute_weight_or_invent_labels():
    logits = torch.zeros(2,5,90,requires_grad=True)
    with torch.no_grad(): logits[:,:,5] = float('-inf')
    labels = torch.tensor([[40,41,-1,43,44],[40,42,-1,43,44]])
    mask = labels>=0
    expected = sum(nn.functional.cross_entropy(logits[:,j],labels[:,j])*w
                   for j,w in [(0,1.),(1,.25),(3,.25),(4,.25)])
    loss = sequence_loss(logits,labels,mask,mask,future_weight=1.)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:,2])==0
    assert torch.isfinite(logits.grad).all()


def test_c_lambda_zero_has_only_current_gradient():
    logits = torch.randn(2,5,90,requires_grad=True)
    labels = torch.full((2,5),40)
    mask = torch.ones_like(labels,dtype=torch.bool)
    loss = sequence_loss(logits,labels,mask,mask,future_weight=0.)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:,1:])==0


@pytest.mark.parametrize('weight',[-1.,float('nan'),float('inf')])
def test_c_rejects_invalid_lambda(weight):
    with pytest.raises(ValueError,match='lambda'):
        sequence_loss(torch.zeros(1,5,90),torch.zeros(1,5,dtype=torch.long),
                      torch.ones(1,5,dtype=torch.bool),torch.zeros(1,5,dtype=torch.bool),future_weight=weight)
