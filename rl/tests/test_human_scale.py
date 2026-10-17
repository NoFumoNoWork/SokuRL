"""Export plans and D share existing semantics;collections reference provenance."""
import json

import numpy as np
import pytest
import torch

from soku_rl.replay.human_dataset import sha256
from soku_rl.replay.human_plans import export_plan
from soku_rl.rl.human_cloning import HumanCollection
from soku_rl.rl.multiscale_bc import SequenceActor,tensor_batch
from test_multiscale_bc import SmallPolicy,dense
from soku_rl.replay.human_windows import prepare_replay


def test_explicit_export_plan_cannot_include_validation_or_modify_metadata(tmp_path):
    (tmp_path/'audit.json').write_text(json.dumps({'status':'PASS'}))
    split = tmp_path/'splits.json'
    row = {'id':1,'sha256':'known','file':'original.rep'}
    split.write_text(json.dumps(dict(train=[row],validation=[{'id':2}])))
    path = tmp_path/'plan.json'
    plan = dict(schema='human_replay_export_plan_v1',partition='train',split_sha256=sha256(split),rows=[row])
    path.write_text(json.dumps(plan))
    assert export_plan(tmp_path,'train',path)[1]['rows']==[row]
    for bad in ([row,row],[{'id':2}], [dict(row,file='different.rep')]):
        path.write_text(json.dumps(dict(plan,rows=bad)))
        with pytest.raises(ValueError): export_plan(tmp_path,'train',path)
    with pytest.raises(ValueError): export_plan(tmp_path,'test',path)


def test_d_is_an_exact_full_weight_alias_of_weighted_c():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    c,d = SequenceActor(SmallPolicy(),'C'),SequenceActor(SmallPolicy(),'D')
    with torch.no_grad(): c.context.residual.weight.fill_(.01)
    d.load_state_dict(c.state_dict(),strict=True)
    batch = tensor_batch(prepare_replay(dense()),'cpu')
    assert c.horizon==d.horizon==5
    torch.testing.assert_close(c.logits(batch),d.logits(batch),rtol=0,atol=0)


def make_part(path,index,partition='train',contract='contract'):
    path.mkdir()
    raw = path/'source.rep';raw.write_bytes(str(index).encode())
    (path/str(index)).mkdir()
    shard = path/str(index)/'shard.npz'
    np.savez(shard,observations=np.zeros((1,4)),actions=[40],supervised=[True],changed=[False],previous_action=[40])
    manifest = path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='human_replay_dataset_v1',status='PASS_EXPORT_ONLY',partition=partition,
        identity=dict(source_checkpoint_sha256=contract),replays=[dict(replay_id=index,status='PASS_EXPORT',
        source=str(raw),source_sha256=sha256(raw),shards=[dict(path='shard.npz',sha256=sha256(shard),supervised=1)])])))
    acceptance = path/'acceptance.json'
    acceptance.write_text(json.dumps(dict(status='PASS_PREFIX_PILOT',manifest_sha256=sha256(manifest),players=[index])))
    return dict(manifest=str(manifest),manifest_sha256=sha256(manifest),
                acceptance=str(acceptance),acceptance_sha256=sha256(acceptance))


def test_collection_preserves_shard_paths_and_rejects_duplicates_or_tamper(tmp_path):
    first,second = make_part(tmp_path/'first',1),make_part(tmp_path/'second',2)
    path = tmp_path/'collection.json'
    path.write_text(json.dumps(dict(schema='human_replay_collection_v1',datasets=[first,second])))
    data = HumanCollection(path)
    assert len(data.shards)==2 and data.shards[0].parent.parent.name=='first'
    assert data.acceptance['players']==[1,2]
    path.write_text(json.dumps(dict(schema='human_replay_collection_v1',datasets=[first,first])))
    with pytest.raises(ValueError,match='duplicates'): HumanCollection(path)
    path.write_text(json.dumps(dict(schema='human_replay_collection_v1',datasets=[dict(first,manifest_sha256='wrong')])))
    with pytest.raises(ValueError,match='changed'): HumanCollection(path)


@pytest.mark.parametrize('partition,contract',[('validation','contract'),('train','different')])
def test_collection_does_not_merge_partitions_or_model_contracts(tmp_path,partition,contract):
    first,second = make_part(tmp_path/'first',1),make_part(tmp_path/'second',2,partition,contract)
    path = tmp_path/'collection.json'
    path.write_text(json.dumps(dict(schema='human_replay_collection_v1',datasets=[first,second])))
    with pytest.raises(ValueError,match='mixes'): HumanCollection(path)
