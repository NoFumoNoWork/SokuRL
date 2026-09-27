# SokuRLBridge

`SokuRLBridge.dll` is a Win32/x86 SWRSToys module that injects primitive P1
logical inputs through `SokuLib::KeyInput`. It does not synthesize Windows
keyboard events.

## Hook

The module follows TrialMode's `KeymapManager::SetInputs` hook at `0x40A45D`.
It calls the original function first and only overrides the P1 manager at
`0x008989A0` while the game is in local Practice battle.

## Shared memory ABI

The mapping is named `Local\SokuRLBridge`. The control block is version 1,
packed to 4-byte alignment, and exactly 80 bytes:

```text
offset  type      field
0       uint32    magic (0x554B4F53)
4       uint32    version (1)
8       uint32    structSize (80)
12      uint32    commandSeq
16      uint32    ackSeq
20      uint32    commandType (1=input, 2=release)
24      int32     horizontalAxis
28      int32     verticalAxis
32      uint32    a
36      uint32    b
40      uint32    c
44      uint32    d
48      uint32    durationFrames
52      uint32    framesRemaining
56      uint64    gameFrame
64      uint32    connected
68      uint32    inGameplay
72      uint32    resultCode
76      uint32    reserved
```

The client writes command payload fields first and changes `commandSeq` last.
The DLL acknowledges consumption through `ackSeq`. One frame is consumed per
P1 `SetInputs` call during Practice battle. After the final active frame, the
next P1 input call explicitly injects neutral input.
