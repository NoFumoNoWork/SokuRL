"""Typed event BC. Independent from primitive PPO and the frozen D heads."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from soku_rl.rl.multiscale_bc import HistoryContext
from soku_rl.replay.human_dataset import sha256,save_json


class CurrentSpatial(nn.Module):
    """Reuse learned entity embeddings/spatial attention; no old temporal stack."""
    def __init__(self, original):
        super().__init__()
        for name in ('position','categories','roles','spatial'):
            setattr(self,name,deepcopy(getattr(original,name)))
        self.output=nn.Sequential(nn.Linear(69,128),nn.Tanh())
        with torch.no_grad():
            self.output[0].weight.copy_(original.output[0].weight[:,:69])
            self.output[0].bias.copy_(original.output[0].bias)

    def forward(self, frame):
        if frame.shape[-1]!=185:raise ValueError('16-object public frame required')
        tokens=frame[...,:180].reshape(-1,18,10)
        encoded=self.position(tokens[...,:4])
        for i,embedding in enumerate(self.categories):
            values=tokens[...,4+i];ids=values.long()
            if torch.any(values!=ids) or torch.any(ids<0) or torch.any(ids>=embedding.num_embeddings):
                raise ValueError('unsupported public event state category')
            encoded=encoded+embedding(ids)
        roles=torch.zeros_like(tokens[...,0],dtype=torch.long)
        roles[:,1]=1;roles[:,2:]=2+tokens[:,2:,9].long()
        mask=tokens[...,0]<=0;mask[:,0]=False
        feature=self.spatial(encoded+self.roles(roles),src_key_padding_mask=mask)[:,0]
        return self.output(torch.cat((feature,frame[...,-5:].reshape(-1,5)),-1))


class HierarchicalActor(nn.Module):
    def __init__(self, spatial_source, actions, input_cnn=None):
        super().__init__()
        if actions<2:raise ValueError('explicit high-level catalog required')
        self.actions=actions
        self.spatial=CurrentSpatial(spatial_source)
        self.input_cnn=deepcopy(input_cnn if input_cnn is not None else HistoryContext().inputs)
        self.kind=nn.Embedding(8,128)
        self.native_action=nn.Embedding(2048,128)
        self.owner=nn.Embedding(2,128)
        self.event_numbers=nn.Linear(3,128)
        self.positions=nn.Parameter(torch.zeros(17,128))
        layer=nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',
            batch_first=True,norm_first=True)
        self.events=nn.TransformerEncoder(layer,3,enable_nested_tensor=False)
        self.state_input=nn.Sequential(nn.Linear(192,128),nn.Tanh())
        self.fusion=nn.Sequential(nn.Linear(256,128),nn.Tanh())
        self.action_head=nn.Linear(128,actions)
        self.action_embedding=nn.Embedding(actions+1,128)
        self.stop_head=nn.Sequential(nn.Linear(257,128),nn.Tanh(),nn.Linear(128,1))
        self.sequence_cell=nn.GRUCell(128,128)
        self.sequence_head=nn.Linear(128,actions)
        self.register_buffer('causal',torch.ones(17,17,dtype=torch.bool).triu(1),persistent=False)

    def context(self,batch):
        current=self.spatial(batch['state'])
        b=current.shape[0]
        if batch['event_states'].shape!=(b,16,185) or batch['inputs'].shape!=(b,60,8):
            raise ValueError('event/input history contract mismatch')
        event_state=self.spatial(batch['event_states'].reshape(-1,185)).reshape(b,16,128)
        ids=batch['event_ids'].long()
        tokens=(event_state+self.kind(ids[...,0])+self.native_action(ids[...,1])+
                self.owner(ids[...,2])+self.event_numbers(batch['event_numbers']))
        tokens=torch.cat((tokens,current[:,None]),1)+self.positions
        padding=torch.cat((~batch['event_mask'].bool(),torch.zeros(b,1,device=current.device,dtype=torch.bool)),1)
        # A left-padded query otherwise has no visible key under a causal mask.
        # Native MHA inference can produce NaNs in that row and poison later
        # layers. Give only padded queries their own diagonal;real queries still
        # cannot attend to any padding orfuture event.
        blocked=self.causal.unsqueeze(0).expand(b,-1,-1)|padding[:,None,:]
        diagonal=torch.eye(17,device=current.device,dtype=torch.bool)[None]
        blocked=blocked & ~(diagonal & padding[:,:,None])
        blocked=blocked[:,None].expand(-1,4,-1,-1).reshape(b*4,17,17)
        summary=self.events(tokens,mask=blocked,is_causal=False)[:,-1]
        own=self.input_cnn(batch['inputs'].transpose(1,2))
        return self.fusion(torch.cat((self.state_input(torch.cat((current,own),-1)),summary),-1))

    def forward(self,batch, *, selected=None, teacher=None):
        context=self.context(batch)
        candidates=batch['candidates'].bool()
        if candidates.shape!=(len(context),self.actions) or not candidates.any(-1).all():
            raise ValueError('empty/invalid command candidate set')
        logits=self.action_head(context).masked_fill(~candidates,float('-inf'))
        active=batch['active_action'].long()
        elapsed=torch.log1p(batch['elapsed'].clamp_min(0))[:,None]/8
        stop=self.stop_head(torch.cat((context,self.action_embedding(active),elapsed),-1)).squeeze(-1)
        selected=logits.argmax(-1) if selected is None else selected.long()
        hidden=context;previous=selected;future=[]
        if teacher is not None and teacher.shape!=(len(context),3):raise ValueError('3 ordered event targets required')
        for i in range(3):
            hidden=self.sequence_cell(self.action_embedding(previous),hidden)
            step=self.sequence_head(hidden)
            future.append(step)
            previous=(teacher[:,i].clamp_min(0) if teacher is not None else step.argmax(-1))
        return dict(action=logits,termination=stop,sequence=torch.stack(future,1))


def hierarchical_loss(outputs,targets):
    def ce(logits,labels,mask):
        mask=mask.bool()
        if not mask.any():return logits[torch.isfinite(logits)].sum()*0
        values=F.cross_entropy(logits[mask],labels[mask].long())
        if not torch.isfinite(values):raise ValueError('nonfinite output or target excluded by candidate contract')
        return values
    action=ce(outputs['action'],targets['action'],targets['action_mask'])
    mask=targets['termination_mask'].bool()
    termination=(F.binary_cross_entropy_with_logits(outputs['termination'][mask],
        targets['termination'][mask].float()) if mask.any() else outputs['termination'].sum()*0)
    sequence=ce(outputs['sequence'],targets['sequence'],targets['sequence_mask'])
    return dict(total=action+.5*termination+.5*sequence,action=action,
                termination=termination,sequence=sequence)


def save_hierarchical(actor,path,source,catalog,metadata):
    path=Path(path)
    if path.exists():raise FileExistsError('preserve checkpoint')
    weights=path.with_suffix('.pt')
    if weights.exists():raise FileExistsError('preserve weights')
    torch.save(actor.state_dict(),weights)
    save_json(path,dict(format='hierarchical_bc_v1',actions=actor.actions,catalog=catalog,
        source=str(Path(source).resolve()),source_sha256=sha256(source),
        weights=weights.name,weights_sha256=sha256(weights),metadata=metadata,
        boundary='Not a90-action PPO/D checkpoint;no optimizer/value-function compatibility claimed'))


def load_hierarchical(path,device='cpu'):
    from stable_baselines3 import PPO
    path=Path(path).resolve(strict=True);data=json.loads(path.read_text())
    if data['format']!='hierarchical_bc_v1':raise ValueError('not a hierarchical checkpoint')
    weights=(path.parent/data['weights']).resolve(strict=True)
    if weights.parent!=path.parent or sha256(weights)!=data['weights_sha256'] or sha256(data['source'])!=data['source_sha256']:
        raise ValueError('hierarchical source/weights changed')
    source=PPO.load(data['source'],device='cpu')
    actor=HierarchicalActor(source.policy.features_extractor,data['actions']).to(device)
    actor.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True)
    actor.eval()
    return actor,data
