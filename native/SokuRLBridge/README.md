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
write game state back into arbitrary memory.

## Simulation boundary

The canonical frame boundary is `SokuLib::VTable_BattleManager.onProcess`, the
same battle-manager process function used by ReplayInputView+ for pause and
frame step. A SokuRL frame increments only after one call to the original
battle process. Rendering, Python polling, and paused process calls do not
increment it.

The existing `KeymapManager::SetInputs` hook at `0x40A45D` remains the logical
input boundary. The hook records the effective P1 and P2 `KeyInput` values and
can replace both values while reconstructing. Normal bridge actions still
control only P1.

## Checkpoint model

Checkpoint restoration and GOTO are currently disabled with
`CHECKPOINT_RESTORE_UNSUPPORTED`. Runtime testing proved that returning
`SCENE_LOADING` directly from an active Practice battle is unsafe and crashes
th123 1.10a. No battle heap or arbitrary process memory is copied or restored,
and the bridge will not claim checkpoint support until a proven Practice reset
entry point is identified.

The implemented but inactive reconstruction validator compares FNV-1a-64 state
hashes after every simulation update and stops on the first mismatch. SokuLib
exposes the match-start seed but not a proven live RNG state.

## Shared memory ABI

The mapping is named `Local\SokuRLBridge`. ABI version 2 uses 4-byte packing:

- 500-byte `ControlBlock` with sequenced commands and a seqlock-protected live
  `RawFrameState`.
- 340-byte fixed-dimensional frame records.
- A 4096-record single-producer/single-consumer ring.
- A native 65536-frame checkpoint history for input reconstruction.

The producer never performs file I/O. `tools/debug_panel.py` drains the ring
and writes `data/raw/<session_id>/metadata.json`, `frames_000.csv`,
`inputs_000.csv`, and `manifest.json`. A lossless recording requires
`dropped_frames == 0`.

## Safety and invalidation

Commands are accepted only in local Practice battle. Leaving Practice or
changing selected characters, stage, start seed, or tracked Practice settings
invalidates the checkpoint. History capacity exhaustion also invalidates it
instead of silently wrapping.
