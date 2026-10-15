import struct
from game_runtime.action_state import NativeActionStateReader


def test_native_timing_fields_not_guessed_startup_recovery():
    values={0x8985DC:struct.pack('<I',0x1000),0x1040:struct.pack('<2I',0x2000,0x2008),
        0x2000:struct.pack('<I',0x3000),0x3158:struct.pack('<I',0x4000),
        0x315C:struct.pack('<I',0x5000)}
    for address,kind,value in [(0x313C,'H',402),(0x313E,'H',1),(0x4008,'h',3),
        (0x404C,'I',0x20),(0x5010,'h',30),(0x5012,'h',40),(0x3196,'h',5),
        (0x34BA,'h',25),(0x37F4,'B',1),(0x37AC,'B',8)]:
        values[address]=struct.pack('<'+kind,value)
    class Memory:
        def begin_frame(self):pass
        def read(self,address,size):
            assert len(values[address])==size
            return values[address]
    result=NativeActionStateReader(Memory()).read(0)
    assert result['phase']=='UNKNOWN'
    assert result['hitstop']==5 and result['untech']==25
    assert result['frame_duration']==3 and result['action_lock']==40
    assert result['charged_attack'] is True
