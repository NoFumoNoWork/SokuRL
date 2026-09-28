# SokuRLBridge

Each th123 process publishes a PID-qualified shared-memory mapping:

```text
Local\SokuRLBridge_<pid>
```

This keeps commands and state isolated when MemoryPatch allows multiple game
instances. `MenuConfirm` injects one frame of the game's logical A/confirm
input while the local character-select scene is active; it does not synthesize
an OS key event or write a scene ID.

`SokuRLBridge.dll` is a Win32/x86 SWRSToys module for Practice-mode logical
input injection, raw frame capture, simulation pause/step, and validated
checkpoint reconstruction. It does not synthesize Windows keyboard events or
write scene IDs or partial battle-manager objects.

## Simulation boundary

The canonical frame boundary is `SokuLib::VTable_BattleManager.onProcess`, the
same battle-manager process function used by ReplayInputView+ for pause and
frame step. A SokuRL frame increments only after one call to the original
battle process. Rendering, Python polling, and paused process calls do not
increment it.

The existing `KeymapManager::SetInputs` hook at `0x40A45D` remains the logical
input boundary. The hook records the effective P1 and P2 `KeyInput` values and
can replace both values with one atomic `StepWithInputs` command while paused.
Normal bridge actions still control only P1.

## Checkpoint model

`EstablishCheckpoint` is armed in the normal character-select scene. The bridge
fixes the requested match seed after `Select::onProcess`, captures frame zero
before the first battle-manager simulation update, and freezes there. GOTO is
implemented by `tools/frame_validation.py` as a fresh SkipIntro Practice
process followed by recorded P1/P2 logical-input replay. The proven-crashing
active-battle `SCENE_LOADING` route remains disabled.

Reconstruction is hybrid. Action machines and projectile/object lists are
rebuilt only through simulation. After every replayed frame, `ApplySimpleState`
may restore only documented scalar state: timer, weather, player position and
speed, facing, HP, spirit, and card counters. It cannot write actions,
animation state, hitstop, flags, hands, or objects. The validator compares a
canonical FNV-1a-64 hash and a field-by-field diff after each frame.

## Shared memory ABI

ABI version 5 uses 4-byte packing:

- 10884-byte `ControlBlock` with sequenced commands and a seqlock-protected live
  `RawFrameState`.
- 10596-byte fixed-dimensional frame records with 64 object slots per player.
- A 512-record single-producer/single-consumer ring.
- A native 4096-frame short-range checkpoint history.

The producer never performs file I/O. `tools/debug_panel.py` drains the ring
and writes `data/raw/<session_id>/metadata.json`, `frames_000.csv`,
`inputs_000.csv`, and `manifest.json`. A lossless recording requires
`dropped_frames == 0`.

## Safety and invalidation

Battle commands are accepted only in local Practice or replay battle.
Checkpoint arming uses the legitimate local character-select path for Practice
and the ReplayDnD command-line path for replay. Leaving battle or changing
selected characters, stage, start seed, or tracked Practice settings invalidates
the checkpoint. History capacity exhaustion also invalidates it instead of
silently wrapping.

## Replay mode

ReplayDnD owns command-line `.rep` loading. For a replay launch the bridge
automatically freezes the first replay BattleManager state at frame zero. Replay
mode uses the same state capture, dual-input stepping, simple-state patch, and
hash comparison path as Practice; it does not parse or replace ReplayDnD.

```powershell
.\.venv\python.exe tools\sokurl.py replay path\match.rep
.\.venv\python.exe tools\sokurl.py replay path\match.rep --frame 5000
```

The first form starts normal playback. The second fast-simulates from replay
frame zero to the requested frame and leaves the process frozen there.
