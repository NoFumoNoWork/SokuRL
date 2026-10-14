"""Read known native action timing/lock fields, without inferring recovery phase."""
import struct


class NativeActionStateReader:
    def __init__(self,memory):self.memory=memory

    def value(self,address,kind):
        return struct.unpack('<'+kind,self.memory.read(address,struct.calcsize('<'+kind)))[0]

    def read(self,seat):
        if seat not in (0,1):raise ValueError('two-player seat required')
        self.memory.begin_frame()
        root=self.value(0x8985DC,'I')
        first,last=struct.unpack('<2I',self.memory.read(root+0x40,8))
        if last-first!=8:raise RuntimeError('expected two original character managers')
        player=self.value(first+seat*4,'I')
        frame=self.value(player+0x158,'I');sequence=self.value(player+0x15C,'I')
        if not frame or not sequence:raise RuntimeError('native action frame/sequence unavailable')
        return dict(action=self.value(player+0x13C,'H'),sequence=self.value(player+0x13E,'H'),
            frame_duration=self.value(frame+0x8,'h'),frame_flags=self.value(frame+0x4C,'I'),
            move_lock=self.value(sequence+0x10,'h'),action_lock=self.value(sequence+0x12,'h'),
            hitstop=self.value(player+0x196,'h'),untech=self.value(player+0x4BA,'h'),
            charged_attack=bool(self.value(player+0x7F4,'B')),
            buffer_timer=self.value(player+0x7AC,'B'),
            phase='UNKNOWN',evidence='COMMUNITY_NATIVE_LAYOUT')
