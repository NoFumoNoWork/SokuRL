"""Hash-bound, lazy event shards. Unknown labels never compress history."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch

from soku_rl.replay.human_dataset import sha256


class EventDataset:
    expected_format='hierarchical_dataset_v1'
    expected_status='PASS_OFFLINE_CAUSAL_LABELS_LIMITED_LEGALITY'
    def __init__(self,path,partition):
        self.path=Path(path).resolve(strict=True);self.manifest=json.loads(self.path.read_text())
        if self.manifest.get('format')!=self.expected_format or self.manifest['status']!=self.expected_status:
            raise ValueError('explicit hierarchical dataset required')
        if partition not in ('train','validation'):raise ValueError('sealed test forbidden')
        self.cases=[c for c in self.manifest['cases'] if c['partition']==partition and c['samples']]
        self.ends=np.cumsum([c['samples'] for c in self.cases]);self.cache=OrderedDict()
        self.count=int(self.ends[-1]) if len(self.ends) else 0
        if not self.count:raise ValueError('no event samples')
        for c in self.cases:
            p=(self.path.parent/c['path']).resolve(strict=True)
            if not p.is_relative_to(self.path.parent) or sha256(p)!=c['sha256']:raise ValueError('event shard changed')

    def batch(self,indices):
        indices=np.asarray(indices,np.int64)
        if np.any(indices<0) or np.any(indices>=self.count):raise ValueError('sample index outside dataset')
        owners=np.searchsorted(self.ends,indices,side='right');result={}
        for owner in np.unique(owners):
            if owner not in self.cache:
                with np.load(self.path.parent/self.cases[owner]['path'],allow_pickle=False) as z:
                    self.cache[owner]={k:z[k] for k in z.files}
                while len(self.cache)>4:self.cache.popitem(last=False)
            self.cache.move_to_end(owner)
            selected=np.flatnonzero(owners==owner);local=indices[selected]-(self.ends[owner-1] if owner else 0)
            for key,value in self.cache[owner].items():
                if key not in result:result[key]=np.empty((len(indices),*value.shape[1:]),value.dtype)
                result[key][selected]=value[local]
        return result


class SemanticDataset(EventDataset):
    expected_format='semantic_event_dataset_v2'
    expected_status='PASS_CAUSAL_DERIVATIVE_LIMITED_FEEDBACK'


class PostOpeningDataset(EventDataset):
    expected_format='semantic_post_opening_dataset_v3'
    expected_status='PASS_OFFLINE_SELECTION_ORIGINAL_ARRAYS_EXACT'

    def batch(self, indices):
        data = super().batch(indices)
        eligible = data.get('training_eligible')
        frames = data.get('observation_frame')
        minimum = self.manifest.get('selection', {}).get('min_observation_frame')
        if (eligible is None or frames is None or type(minimum) is not int or minimum < 0
                or eligible.shape != (len(indices),) or eligible.dtype != np.bool_
                or frames.shape != eligible.shape or frames.dtype.kind not in 'iu'
                or (frames[eligible] < minimum).any()):
            raise ValueError('post-opening eligibility/time contract missing or invalid')
        return data


def semantic_training_dataset(path,partition):
    kind=json.loads(Path(path).read_text()).get('format')
    if kind==SemanticDataset.expected_format:return SemanticDataset(path,partition)
    if kind==PostOpeningDataset.expected_format:return PostOpeningDataset(path,partition)
    raise ValueError('explicit supported semantic dataset required')


def tensors(data,device):
    inputs={k:torch.as_tensor(v,device=device) for k,v in data.items() if not k.startswith('target_')}
    targets={k[7:]:torch.as_tensor(v,device=device) for k,v in data.items() if k.startswith('target_')}
    return inputs,targets
