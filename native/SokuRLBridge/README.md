# SokuRLBridge

## Snapshot-free Fresh Reset

Opt in before process launch with `SOKURL_FRESH_RESET=1`. The existing
`reset_episode(seed)` command then starts a fresh original-engine world in the
same process without a frame-one snapshot or Giuroll. Capability bit zero of
`traceReserved` is exposed as `BridgeClient.fresh_reset_enabled`.

FreshReset.cpp patches only the two character-construction CALL sites in
onArenaStart (0x4818D4/0x4818E7). Cold startup is unchanged. Warm reset runs
original onExit and onArenaStart, including original seed/world initialization,
but reuses the loaded character resources. Native constructor-defined hit/hurt
box counts are cleared before reset; no pointer graph is copied from a snapshot.

Validated scope: unchanged Reimu/Marisa character, side, palette, deck and key
manager configuration; Clear/cards-disabled tests. Unsupported configurations
fail explicitly. ResetTiming and FreshReset cannot own the same hooks together.
The default is OFF. Setting it OFF restores the original scene-reload reset.

Two/four/eight-instance comparisons, native RNG, active-combat and KO resets
passed. Public state observations and subsequent worker input passed. RGB is
optional and can be disabled; pixel parity is NOT VERIFIED and the RL worker
rejects image mode with fresh reset. This does not invalidate state-only reset.
First cold loading remains necessary; external AI DLL internal restart state and
all-character hidden-state equivalence are NOT TESTED.

Reproduce using `tools/validate_fresh_reset.py --workers 2 --frames 700
--cycles 4 --active-reset --immediate-input --terminal-reset --characters 0 1
--output <report.json>`. See the root README for usage and measured results.

## Current CPU ABI (12)

The live frame is 12,436 bytes (previous diagnostic ABI11: 38,380).
Training records retain player resources/combo/invulnerability/skill levels,
box counts and object collision/lifetime fields. They do not contain process
addresses, parent/child pointers, observer generations or box geometry.
Historical ABI7/10/11 binary records remain decodable by tools/bridge_shared.py;
old live DLLs are rejected rather than interpreted as ABI12.

Native function tracing is OFF by default. Explicitly launching with
`SOKURL_NATIVE_TRACE=1` installs its hooks and creates the independent mapping
`Local\SokuRLBridgeTrace_<pid>` (2 MiB). The normal mapping is 7,068,164 bytes
and contains no trace ring. ConfigureTrace enable requests fail explicitly
unless the process opted in. Positive tracing execution has not yet been
revalidated for this layout; default live 10,000-frame parity has passed.

Oracle Trace V1 geometry exporters require historical diagnostic records;
they reject compact live frames rather than fabricating boxes. Re-enable the
archived diagnostic ABI only through an explicit restore/revalidation, not
by assuming NativeTrace opt-in restores removed geometry fields.

Normal/headless/unlimited validation intentionally shows one rendered window
and two black headless windows. Check per-PID command acknowledgements, frame
progress, latest state, ring write/read counters and drops, not window color.
Read-only inspection: `tools/bridge_worker_health.py --output <report.json>`.
CPU tests and controlled snapshots are different from a complete hidden-state
or GPU simulator correctness claim.

Each th123 process publishes a PID-qualified shared-memory mapping:

```text
Local\SokuRLBridge_<pid>
```

This keeps commands and state isolated when MemoryPatch allows multiple game
instances. `MenuConfirm` injects one frame of the game's logical A/confirm
input while the local character-select scene is active; it does not synthesize
an OS key event or write a scene ID.

`SokuRLBridge.dll` is a Win32/x86 SWRSToys module for local Practice/VS Player logical
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
before the first battle-manager simulation update, switches the documented
Practice dummy setting to `DUMMY_STATE_2P_CONTROL`, and freezes there. This
makes both logical input streams controllable without OS input injection. GOTO is
implemented by `tools/frame_validation.py` as a fresh SkipIntro Practice
process followed by recorded P1/P2 logical-input replay. The proven-crashing
active-battle `SCENE_LOADING` route remains disabled.

The opt-in VS launcher uses a separate Title-context bootstrap. After the
original `Title::onProcess` runs, it selects fallback local input ownership,
calls `setBattleMode(BATTLE_MODE_VSPLAYER, BATTLE_SUBMODE_PLAYING2)`, initializes
both profiles and effective decks, and returns `SCENE_LOADING`. It is armed only
by the `SOKURL_VS_BOOTSTRAP` process environment variable from `tools/sokurl.py vs`.

## AI match capture and native replay saving

Run one Reimu/Marisa match with community AI, fixed Clear weather and cards
disabled from the repository root:

```powershell
& .\tools\run_p0_ai_probe_once.ps1
```

Keep the game window focused while the AI plays. Advance the post-match menus
manually and accept the native replay-save prompt. The recorder stops both AI
processes after leaving battle and waits up to 300 seconds for saving. It copies
the sole new/changed, stable nonempty `.rep` to the match directory as `match.rep`,
checks the copied bytes and records SHA-256 provenance in `metadata.json` alongside
`frames.raw.gz`. Missing replay makes the acquisition incomplete, returns a
nonzero exit code and leaves the game window open. Multiple candidate files are
rejected rather than silently assigned to a match.

Native replay saving requires VS **mode 3, submode 1**. SokuLib calls submode 1
`BATTLE_SUBMODE_PLAYING2`; `PLAYING1` is 0 and must not be used for this acquisition
workflow. Original save helper `0x43EBE0` skips saving unless submode at `0x898688`
equals 1. The recorder rejects the wrong mode before starting AI. Runtime save
policy byte `0x8998B3` is temporarily set to 4 (ask), checked and restored during
cleanup. Save policy 4 alone cannot fix an incorrect battle submode.

Do not terminate the game immediately on leaving `SCENE_BATTLE`: that interrupts
the native save flow. Raw Bridge records are observation data, not native replay
files, and successful frame capture alone is not a complete acquisition.

On 2026-10-08 the corrected workflow saved
`th123_jp/replay/261008_2224_reimu_marisa.rep` and archived an identical copy under
`SokuSim/diagnostics/dynamic_probe/p0_ai_batch/20261008T142213.660809Z/match_01/match.rep`.
The capture contains 9,327 raw frames with zero drops; the replay is 34,109 bytes,
SHA-256 `fea8df15c644f501ef045e95d39153417a35817cd8fe6e5971e6bb4664b5393e`.
Native saving and file association are confirmed; playback parity is pending.
Do not assume `.rep` preserves Bridge-only fixed-weather/card-suppression settings:
reapply the capture conditions and verify them when replaying for native tracing.
The earlier ten captures without `.rep` are archived under
`SokuSim/archive/20261008_legacy_captures_without_rep/`.

Same-condition native replay validation is available through
`tools/align_p0_native_replay.py`; see
`SokuSim/diagnostics/dynamic_probe/replay_alignment/README.md` for exact evidence.
Capture `20261008T143448.648089Z` matches the tested battle projection over
12,244 frames and reproduces the frame-3618 air guard. Bridge input observations,
post-match tail and complete hidden state are not covered by that equivalence.
Keep the currently failing hardware FieldWrite watch disabled during validation.

Reconstruction is hybrid. Action machines and projectile/object lists are
rebuilt only through simulation. After every replayed frame, `ApplySimpleState`
may restore only documented scalar state: timer, weather, player position and
speed, facing, HP, spirit, and card counters. It cannot write actions,
animation state, hitstop, flags, hands, or objects. The validator compares a
canonical FNV-1a-64 hash and a field-by-field diff after each frame.

## Shared memory ABI

ABI version 11 uses 4-byte packing. Spirit fields are signed 32-bit ABI values
that preserve the game's signed 16-bit resource semantics, including transient
negative values. Version 8 expands oracle capture with hand costs, selected
card index, combo/resource timers, invulnerability timers, skill levels, raw
player/object hit and hurt boxes, and per-object `(pointer, generation)` oracle
identity. Version 11 additionally records native player source box counts and
the statically documented GameObject collision, lifetime, custom-data pointer,
and parent/child relationship fields:

- 38748-byte `ControlBlock` with sequenced commands and a seqlock-protected live
  `RawFrameState`.
- 38380-byte fixed-dimensional frame records with 64 object slots per player.
- A 512-record single-producer/single-consumer ring.
- A native 4096-frame short-range checkpoint history.
- A separate 16384-record native trace ring with frame/action/function filters,
  explicit dropped-event accounting, and no producer-side file I/O.

The producer never performs file I/O. `tools/debug_panel.py` drains the ring
and writes `data/raw/<session_id>/metadata.json`, `frames_000.csv`,
`inputs_000.csv`, and `manifest.json`. A lossless recording requires
`dropped_frames == 0`.

## Safety and invalidation

Battle commands are accepted only in local Practice, local VS Player, or replay battle.
Checkpoint arming uses the legitimate local character-select path for Practice
and the ReplayDnD command-line path for replay. Leaving battle or changing
selected characters, stage, start seed, or tracked Practice settings invalidates
the checkpoint. History capacity exhaustion also invalidates it instead of
silently wrapping.

## Optional Performance Experiments

`SOKURL_FAST_RESET_PROVIDER=<absolute experiment DLL path>` explicitly enables
native branch checkpoints. `BridgeClient.save_branch_checkpoint()` saves one
paused native frame>=1; `fast_reset()` restores that same world/RNG/frame.
`reset_episode(seed)` still performs complete new-world initialization at frame0.
Seed relabelling is rejected. Images/NativeTrace are unsupported. Do not install
the provider as an ordinary SWRSToys module. Build, safety boundaries and actual
validation: native/FastResetExperiment/README.md and docs/reports/fast_reset/20261012/.
This is not yet a complete RL wrapper/policy-state checkpoint.

`SOKURL_IDLE_WAIT=1` inserts a bounded 1 ms wait only while unlimited local VS
is paused without steps or pending commands. Default OFF: measurements show
less idle CPU but worse single-frame command latency. Do not enable globally
as a throughput optimization.

`SOKURL_PROFILE_STAGES=1` enables independent QPC counters in
`Local\SokuRLBridgePerf_<pid>` (six uint64 values: frequency, frames,
simulation, capture, hash, recording). Default OFF. Read after pause; capture
includes hash. This is diagnostic only and leaves ABI12 unchanged.
See `tools/profile_training_cost.py` and `tools/validate_training_cost.py`;
evidence belongs under `docs/reports/training_cost/`, not the game directory.

## Replay Usage

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
## Optional Command Counters and Combat Events

Set `SOKURL_COMBAT_EVENTS=1` and `SOKURL_NATIVE_INPUT_COUNTERS=1` before
launch for the tested command-input path. Defaults remain OFF. Combat sidecar
ABI8 advertises support/enabled flags; RawFrame ABI12 is unchanged.
The accumulator advances once per controlled simulation frame, not per input hook.
Counter-mode GotoFrame/FastReset/branch checkpoints are explicitly unsupported.
Use `tools/run_command_input.py --help` for the selected-instance command entry.
The runner requires an owned, paused match and exclusive control/event consumption.
