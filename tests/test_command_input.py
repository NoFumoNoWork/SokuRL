import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import pytest
from command_input import compile_commands,CommandPlayer,WaitFor,Keys,NEUTRAL


def play(text,facing=1):
    player=CommandPlayer(compile_commands(text));output=[]
    for frame in range(1000):
        if player.done:break
        output.append(player.next(frame,facing))
    return output


def test_same_engine_for_aaa_motion_double_down_and_charge():
    aaa=play('AAA')
    assert [x[2] for x in aaa]==[1,1,0,0]*3
    assert play('5B')==[(0,0,0,1,0,0,0,0)]*2+[NEUTRAL]*2
    motion=play('236B')
    assert [x[:2] for x in motion[:6]]==[(0,1)]*2+[(1,1)]*2+[(1,0)]*2
    double=play('2(2)2B')
    assert [x[:2] for x in double[:6]]==[(0,1)]*2+[(0,0)]*2+[(0,1)]*2
    assert double==play('22B')
    charge=play('2B{40}')
    assert len(charge)==42 and all(x[1]==1 and x[3]==1 for x in charge[:40])
    assert charge==play('2HB')


def test_facing_latched_per_command_not_per_frame_or_character():
    first=play('236B',1);second=play('236B',-1)
    assert all(a[0]==-b[0] and a[1:]==b[1:] for a,b in zip(first,second))
    player=CommandPlayer(compile_commands('236B'))
    assert player.next(0,1)[:2]==(0,1)
    player.next(1,-1)
    assert player.next(2,-1)[:2]==(1,1)


def test_chain_uses_bounded_generic_native_gates_and_no_frame_tables():
    program=compile_commands('A')+(WaitFor('hit',timeout=3),)+compile_commands('236B')
    player=CommandPlayer(program)
    for frame in range(4):player.next(frame,1)
    assert player.next(4,1)==NEUTRAL
    assert player.next(5,1,events=[dict(kind='HIT',command_id=1)])[:2]==(0,1)
    waiting=CommandPlayer((WaitFor('cancel_window',timeout=1),Keys(5,'B',2,1)))
    assert waiting.next(0,1,state=dict(frame_flags=0))==NEUTRAL
    with pytest.raises(TimeoutError):waiting.next(1,1,state=dict(frame_flags=0))


def test_invalid_and_nonconsecutive_programs_fail_explicitly():
    for text in ('236Q','B{0}','(0)','HB{999999}','__import__(x)'):
        with pytest.raises(ValueError):compile_commands(text)
    player=CommandPlayer(compile_commands('B'))
    player.next(0,1)
    with pytest.raises(RuntimeError):player.next(2,1)


def test_confirmed_chain_waits_for_native_input_phase_acceptance():
    player=CommandPlayer(compile_commands('AAA',confirm=True))
    assert player.next(0,1)[2]==1
    player.observe_native([dict(kind='ACTION_TRANSITION',attacker_seat=0,attack_instance=7,
        accepted=True,native_input_scope=3,lineage='DIRECT',native_inputs_before=[0,0,1,0,0,0,0,0],
        buffered_inputs_before=[0]*8,released_inputs_before=[0]*6)],0)
    for frame in range(1,4):player.next(frame,1)
    assert player.next(4,1)[2]==1  # second press, after first accepted
    for frame in range(5,8):player.next(frame,1)
    assert player.next(8,1)[2]==1  # bounded full-command retry until accepted
    player.observe_native([dict(kind='ACTION_TRANSITION',attacker_seat=0,attack_instance=8,
        accepted=True,native_input_scope=4,lineage='DIRECT',native_inputs_before=[0]*8,
        buffered_inputs_before=[0,0,1,0,0,0,0,0],released_inputs_before=[0]*6)],0)
    assert player.next(9,1)[2]==1


def test_confirmation_timeout_is_simulation_bounded_and_replays_full_motion():
    stages=compile_commands('236B')
    player=CommandPlayer(stages+(WaitFor('accepted',timeout=2,retry=True),))
    original=[player.next(frame,1) for frame in range(8)]
    assert player.next(8,1)==original[0]
    assert player.next(9,1)==original[1]
    with pytest.raises(TimeoutError):player.next(10,1)
