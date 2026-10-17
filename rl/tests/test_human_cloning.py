import json

import numpy as np
import pytest

from soku_rl.replay.human_dataset import sha256
from soku_rl.rl.human_cloning import HumanDataset


def dataset(tmp_path):
    raw = tmp_path/'original.rep'
    raw.write_bytes(b'test provenance only,not a native replay')
    folder = tmp_path/'1'
    folder.mkdir()
    shard = folder/'shard.npz'
    np.savez_compressed(shard,observations=np.arange(12,dtype=np.float32).reshape(3,4),
        actions=np.array([41,-1,40]),supervised=np.array([True,False,True]),
        changed=np.array([True,True,False]),previous_action=np.array([40,41,-1]))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='human_replay_dataset_v1',status='PASS_EXPORT_ONLY',
        partition='train',replays=[dict(replay_id=1,status='PASS_EXPORT',source=str(raw),
            source_sha256=sha256(raw),shards=[dict(path='shard.npz',sha256=sha256(shard),supervised=2)])])))
    acceptance = tmp_path/'acceptance.json'
    acceptance.write_text(json.dumps(dict(status='PASS_PREFIX_PILOT',manifest_sha256=sha256(manifest))))
    return manifest,acceptance,shard


def test_human_loader_filters_labels_not_history(tmp_path):
    manifest,acceptance,_ = dataset(tmp_path)
    data = HumanDataset(manifest,acceptance)
    batch = data.batch(np.array([1,0,1]))
    assert data.count == 2
    assert batch['observations'].tolist() == [[8,9,10,11],[0,1,2,3],[8,9,10,11]]
    assert batch['actions'].tolist() == [40,41,40]
    assert 'returns' not in batch


def test_human_loader_requires_hash_bound_acceptance(tmp_path):
    manifest,acceptance,_ = dataset(tmp_path)
    manifest.write_text(manifest.read_text()+' ')
    with pytest.raises(ValueError):
        HumanDataset(manifest,acceptance)


def test_human_loader_rejects_changed_shards_and_index(tmp_path):
    manifest,acceptance,shard = dataset(tmp_path)
    data = HumanDataset(manifest,acceptance)
    with pytest.raises(ValueError):
        data.batch(np.array([2]))
    with pytest.raises(ValueError):
        data.batch(np.array([0.5]))
    shard.write_bytes(shard.read_bytes()+b'changed')
    with pytest.raises(ValueError):
        HumanDataset(manifest,acceptance)
