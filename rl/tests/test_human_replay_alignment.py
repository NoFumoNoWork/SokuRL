import pytest
from soku_rl.replay.human import consumed_pair, verify_pair, supervision_action


def test_count_not_frame_identifies_poll():
    pair = consumed_pair([0, 0, 0x10, 0x20, 0, 0], 3, 4, 2)
    assert pair['poll_index'] == 1
    assert pair['logical'][0][2] == 1
    assert pair['logical'][1][3] == 1
    assert consumed_pair([0, 0], 3, 2, 2) is None


def test_existing_decoder_priority_preserves_original_opposing_bits():
    pair = consumed_pair([0xC,0x3],3,2,0)
    assert pair['raw_words'] == [0xC,0x3]
    assert pair['logical'][0][:2] == (-1,0)
    assert pair['logical'][1][:2] == (0,-1)


@pytest.mark.parametrize('before,after', [(3, 1), (2, 4), (6, 0), (8, 6)])
def test_reject_unknown_or_truncated_boundary(before, after):
    with pytest.raises(ValueError):
        consumed_pair([0] * 6, 3, before, after)


def test_wrong_frame_and_seat_rejected():
    pair = consumed_pair([0x10, 0x20], 3, 2, 0)
    events = [dict(frame=42, attacker_seat=s, native_inputs_before=p) for s, p in enumerate(pair['logical'])]
    verify_pair(pair, events, 42)
    with pytest.raises(ValueError):
        verify_pair(pair, events, 41)
    events[0]['attacker_seat'] = 1
    with pytest.raises(ValueError):
        verify_pair(pair, events, 42)


def test_masked_and_unrepresentable_inputs_have_no_replacement():
    from types import SimpleNamespace
    commands = [a*64+b for a in range(9) for b in (0,1,2,4,8,16,32,9,10,12)]
    interface = SimpleNamespace(inverse={c:i for i,c in enumerate(commands)})
    blocked = {i for i,c in enumerate(commands) if c & 48}
    assert supervision_action((0,0,1,0,0,0,0,0),interface,blocked) == (41,None)
    assert supervision_action((0,0,0,0,0,0,1,0),interface,blocked) == (None,'POLICY_MASKED')
    assert supervision_action((0,0,1,1,0,0,0,0),interface,blocked) == (None,'UNREPRESENTABLE')


def test_unsupervised_frame_keeps_history_and_round_reset_clears_it():
    from types import SimpleNamespace
    import numpy as np
    from soku_rl.env.observation_history import ObservationHistory
    from soku_rl.env.observation.animation import AnimationObservation
    config = SimpleNamespace(observation_mode='animation',encode=lambda o:np.asarray(o.values))
    history = ObservationHistory(config,('player_0','player_1'))
    def observations(f,value):
        return (AnimationObservation(f,(value,)),)*2
    history.reset(0,observations(0,0))
    history.append(1,observations(1,1))  # The label may be masked, not the frame.
    history.append(2,observations(2,2))
    assert history.observations()['player_0'][-2:].tolist() == [1,2]
    with pytest.raises(RuntimeError):
        history.append(4,observations(4,4))
    history.reset(0,observations(0,10))
    assert np.all(history.observations()['player_0']==10)
