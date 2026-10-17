import ctypes
import struct
import pytest
from game_runtime.combat_capture import decode_event, EVENT, HEADER, CAPACITY, CombatCaptureClient
from soku_rl.env.combat_events import counter_hit_bonus


def bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def hit():
    data=[0]*34
    data[0]=703;data[2]=1;data[3]=2;data[4]=1;data[6]=1
    data[8]=800;data[13]=9800;data[14]=9453
    data[17]=bits(.98);data[18]=bits(1.);data[19]=bits(.93)
    data[20]=1;data[21]=1;data[28]=1;data[29]=1
    data[30]=3;data[31]=9;data[32]=1;data[33]=0
    return data


def test_damage_rate_is_not_contact_entry_or_exit_rate():
    event=decode_event(hit())
    assert event['rate_at_damage']==1 and event['rate_before']<1 and event['rate_after']<1
    assert event['combo_count_at_damage']==0
    assert event['attack_instance']==9 and event['source_instance']==3
    assert event['hit_or_block']=='HIT'
    assert event['punish_counter'] is None and event['independent_input_segment'] is None
    old=decode_event(hit()[:30],version=1)
    assert old['attack_instance'] is None and old['combo_count_at_damage'] is None


def test_real_key_edges_do_not_prove_independent_combo_segment():
    data=hit();data[4]=4;data[27]=1;data[33]=0x1234
    data[13:19]=[1,9,0,0,0,0]
    data[19]=0xFFFFFFFF;data[20]=0
    event=decode_event(data)
    assert event['native_inputs_before']==(-1,0,1,9,0,0,0,0)
    assert event['native_button_edges']==(0,)
    assert event['independent_input_segment'] is None
    assert decode_event(data,version=2)['native_inputs_before'] is None


def test_unknown_or_malformed_rate_fails_closed():
    data=hit();data[18]=0xFFFFFFFF
    assert decode_event(data)['damage_rate'] is None
    data[18]=0x7FC00000
    with pytest.raises(ValueError,match='nonfinite'):
        decode_event(data)
    data=hit()[:30];data[4]=4
    with pytest.raises(ValueError,match='ABI2'):
        decode_event(data,version=1)


def test_ring_consumes_each_event_once_and_rejects_drop():
    buffer=ctypes.create_string_buffer(HEADER.size+CAPACITY*EVENT.size)
    HEADER.pack_into(buffer,0,0x31564543,3,EVENT.size,CAPACITY,1,0,0,0)
    EVENT.pack_into(buffer,HEADER.size,*hit())
    client=CombatCaptureClient.__new__(CombatCaptureClient)
    client.view=ctypes.addressof(buffer);client.event=EVENT;client.version=3
    assert len(client.drain())==1
    assert client.drain()==[]
    ctypes.c_uint32.from_address(client.view+24).value=1
    with pytest.raises(RuntimeError,match='lost'):
        client.drain()


def test_ch_proxy_only_rewards_first_damage_in_combo():
    data=hit()+[0x20]
    event=decode_event(data,version=4)
    assert event['counter_hit_first'] is True and event['punish_counter'] is True
    assert counter_hit_bonus(event)==pytest.approx(.2*347)
    data[33]=1
    event=decode_event(data,version=4)
    assert event['counter_hit_first'] is False
    assert counter_hit_bonus(event)==0
    data[33]=0;data[34]=0
    assert counter_hit_bonus(decode_event(data,version=4))==0
    with pytest.raises(NotImplementedError):
        counter_hit_bonus(decode_event(hit(),version=3))
    data[34]=0x20;data[29]=2
    assert decode_event(data,version=4)['counter_hit_first'] is None


def test_input_scope_preserves_buffer_release_and_command_evidence():
    data=hit()+[0]*18
    data[4]=4;data[27]=1;data[35]=7
    data[13]=2
    data[36:44]=[0,-1,0,1,0,0,0,0]
    data[44:50]=[1,0,0,0,0,0]
    data[50]=2;data[51]=4
    event=decode_event(data,version=5)
    assert event['native_input_scope']==7
    assert event['buffered_inputs_before']==(0,-1,0,1,0,0,0,0)
    assert event['released_inputs_before']==(1,0,0,0,0,0)
    assert event['command_combination_before']==2
    assert event['movement_combination_before']==4
    assert event['independent_input_segment'] is None


def test_original_replay_boundary_is_separate_from_processed_player_keys():
    data=[0]*52
    data[0]=133;data[2]=1;data[4]=8;data[5]=data[6]=1
    data[7]=0x1234;data[8]=1;data[9]=42
    data[10]=data[11]=0x5678;data[12]=100;data[13]=99;data[14]=132
    data[20:28]=[0xFFFFFFFF,0,1,0,0,0,0,0]
    event=decode_event(data,version=9)
    assert event['kind']=='REPLAY_INPUT_BOUNDARY'
    assert event['frame']==133 and event['bridge_frame_before']==132
    assert event['remaining_before']==100 and event['remaining_after']==99
    assert event['original_key_manager_inputs']==(-1,0,1,0,0,0,0,0)
    assert event['phase']=='SIMULATION' and event['call_serial']==42
    assert 'native_inputs_before' not in event
    data[5]=data[6]=2
    unknown=decode_event(data,version=9)
    assert unknown['attacker_seat'] is None and unknown['seat_status']=='UNKNOWN'
    with pytest.raises(ValueError,match='ABI9'):
        decode_event(data,version=8)
    data[8]=2
    with pytest.raises(ValueError,match='ABI9'):
        decode_event(data,version=9)
