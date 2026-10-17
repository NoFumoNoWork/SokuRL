from types import SimpleNamespace

import numpy as np
import pytest

from soku_rl.replay.human import consumed_pair
from soku_rl.replay.human_dataset import LabelHistory, validate_arrays


@pytest.fixture
def interface():
    commands = tuple(axis*64+b for axis in range(9) for b in (0,1,2,4,8,16,32,9,10,12))
    return SimpleNamespace(inverse={c:i for i,c in enumerate(commands)},
        observation_space=SimpleNamespace(shape=(4,)), action_space=SimpleNamespace(n=90))


def test_missing_labels_do_not_compress_command_history(interface):
    history = LabelHistory(2)
    pair = consumed_pair([0x10, 0], 3, 2, 0)
    assert history.label(None,0,interface,set(),True)['reason'] == 'NO_NATIVE_POLL'
    assert not history.label(pair,0,interface,set(),True)['supervised']
    assert not history.label(pair,0,interface,set(),True)['supervised']
    assert history.label(pair,0,interface,set(),True)['supervised']
    assert not history.label(None,0,interface,set(),True)['supervised']
    assert not history.label(pair,0,interface,set(),True)['supervised']


def test_first_command_not_falsely_counted_as_known_change(interface):
    history = LabelHistory(0)
    pair = consumed_pair([0x10,0],3,2,0)
    assert not history.label(pair,0,interface,set(),True)['changed']
    assert not history.label(pair,0,interface,set(),True)['changed']
    assert not history.label(pair,0,interface,set(),False)['supervised']
    # A new round creates a fresh history,never carries over previous commands.
    assert LabelHistory(2).label(pair,0,interface,set(),True)['reason'] == 'UNKNOWN_COMMAND_HISTORY'


def arrays():
    return dict(observations=np.zeros((2,4),np.float32),actions=np.array([41,40]),
        supervised=np.ones(2,dtype=bool),history_known=np.ones(2,dtype=bool),
        compatible=np.ones(2,dtype=bool),boundary=np.zeros(2,dtype=bool),
        seats=np.zeros(2,int),remaining_before=np.array([4,2]),remaining_after=np.array([2,0]),
        poll_index=np.array([0,1]),raw_words=np.array([[0x10,0],[0,0]]),
        raw_words_present=np.ones(2,dtype=bool),observation_frames=np.array([69,70]),
        consumption_frames=np.array([70,71]))


def test_valid_source_and_no_fake_returns(interface):
    data = arrays()
    validate_arrays(data,[0x10,0,0,0],3,0,interface,set())
    assert not {'returns','rewards'} & data.keys()


@pytest.mark.parametrize('key,value', [
    ('seats',[1,1]), ('consumption_frames',[71,72]),
    ('remaining_after',[0,0]), ('raw_words',[[0,0],[0,0]]),
    ('history_known',[False,True]), ('actions',[41,-1]),
    ('boundary',[True,False]), ('compatible',[True,False]),
])
def test_rejects_wrong_seat_shift_source_truncation_history_and_boundary(interface,key,value):
    data = arrays()
    data[key] = np.asarray(value)
    with pytest.raises(ValueError):
        validate_arrays(data,[0x10,0,0,0],3,0,interface,set())


def test_boundary_is_retained_but_never_supervised(interface):
    data = arrays()
    data['boundary'][1] = True
    data['supervised'][1] = False
    data['raw_words_present'][1] = False
    data['remaining_after'][1] = 4  # A new stream must not be decoded with old words.
    validate_arrays(data,[0x10,0,0,0],3,0,interface,set())
