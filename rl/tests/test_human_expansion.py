"""Expansion freezes all prior partitions and needs no gameplay inference."""
from copy import deepcopy
from omegaconf import OmegaConf

from plan_human_expansion import candidates
from soku_rl.rl.factorized_training import validate_config


def row(i, a=10, b=11, set_id=None, digest=None):
    return dict(id=i, status='ok', sha256=digest or str(i), set_id=set_id or str(i),
        time_utc='2025-04-20T00:00:00Z', players=dict(
            server=dict(id=a, character_id=1), client=dict(id=b, character_id=0)))


def test_reserved_players_sets_hashes_ids_and_unknown_players_excluded():
    splits=dict(train=[row(1)], validation=[row(2,20,21)], test=[row(3,30,31)],
                excluded=[row(4,40,41)])
    old=deepcopy(splits)
    rows=[row(1),row(4),row(5,20,11),row(6,10,31),row(7,set_id='2'),
          row(8,digest='3'),row(9,a=None),row(10),row(11,digest='10')]
    pool, _=candidates(dict(games={str(r['id']):r for r in rows}),splits)
    assert [r['id'] for r in pool]==[10]
    assert pool[0]['sampling']['focal_side']=='server'
    assert splits==old and 'sampling' not in rows[-2]


def test_seat_is_derived_from_character_not_assumed_left():
    r=row(1)
    r['players']['server']['character_id']=0
    r['players']['client']['character_id']=1
    splits=dict(train=[],validation=[],test=[],excluded=[])
    pool,_=candidates(dict(games={'1':r}),splits)
    assert pool[0]['sampling']['focal_side']=='client'
    assert pool[0]['sampling']['split']=='train'


def test_missing_set_and_explicit_test_assignment_are_not_admitted():
    a,b=row(1),row(2)
    a['set_id']=None
    b['sampling']=dict(split='test')
    pool,_=candidates(dict(games={'1':a,'2':b}),
                      dict(train=[],validation=[],test=[],excluded=[]))
    assert not pool


def test_expanded_profile_changes_only_dataset():
    old=OmegaConf.to_container(OmegaConf.load('rl/config/hierarchical_factorized_post_opening.yaml'))
    new=OmegaConf.to_container(OmegaConf.load('rl/config/hierarchical_factorized_expanded.yaml'))
    validate_config(new)
    assert old.pop('data') != new.pop('data')
    assert old == new
