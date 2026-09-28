# SokuRL

A deterministic, faster-than-real-time reinforcement-learning environment
backend for Touhou Hisoutensoku (Touhou 12.3 / `th123`).

SokuRL runs the original game engine as the authoritative simulator. It adds
dual-player logical control, structured per-frame state, deterministic
reconstruction, isolated multi-process workers, and accelerated local VS
simulation without replacing the game's battle logic.

## Highlights

- **28,000+ sim-FPS** on one unlocked worker, about 467x real time
- **150,000+ aggregate sim-FPS** with 8 recording-stable workers
- **10,000-frame deterministic equivalence** across rendered, headless, and
  unlimited execution
- **14,954-frame expert replay capture** with zero dropped records
- Deterministic reconstruction with up to **55 live objects** in one player's
  object list
- Automated local VS Player startup with independent P1/P2 logical control

Current development covers the environment and simulation backend. Policy
learning, population self-play, and online opponent adaptation are the next
stage.

## Architecture

```text
                 Python / RL trainer
                         |
                observations / actions
                         |
             PID-isolated shared memory
              +----------+----------+
              |          |          |
              v          v          v
          th123 #1   th123 #2   th123 #N
              |          |          |
           original   original   original
            battle     battle     battle
            engine     engine     engine
```

Each worker runs the original th123 simulation. SokuRL observes and controls the
game at its logical frame boundary, so collision, animation, hitstop, weather,
cards, projectiles, and character-specific behavior still come from the game.

## Project Status

```text
[x] native simulation backend
[x] per-frame structured state and deterministic hashes
[x] dual-player logical input
[x] deterministic replay, reset, and frame seek
[x] PID-isolated parallel workers
[x] headless battle rendering
[x] faster-than-real-time local VS simulation
[ ] trainer-facing observation and action spaces
[ ] reward design and PPO baseline
[ ] population self-play
[ ] online opponent adaptation
```

## Requirements

- Windows
- A legally obtained Touhou Hisoutensoku 1.10a installation
- `th123.exe` MD5: `DF35D1FBC7B583317ADABE8CD9F53B2E`
- Python 3.11 x64 in the repository-local `.venv`
- MSVC with Win32/x86 support for native bridge builds

The game executable and proprietary game data are **not distributed with this
repository**. SokuRL does not patch `th123.exe`, `th123a.dat`, `th123b.dat`, or
`th123c.dat` on disk.

Verify a local installation with:

```powershell
.\.venv\python.exe scripts\00_check_game.py
```

## Quick Start

```powershell
# Stable local Practice preset
.\.venv\python.exe tools\sokurl.py practice

# Local VS Player through the normal Title -> Loading -> Battle lifecycle
.\.venv\python.exe tools\sokurl.py vs

# Keep the original simulation but skip complex battle rendering
.\.venv\python.exe tools\sokurl.py vs --headless

# Headless local VS without the original 60 FPS wall-clock wait
.\.venv\python.exe tools\sokurl.py vs --headless --unlimited
```

`--unlimited` requires `--headless`. Practice and normal VS retain their
original pacing and rendering behavior.

Manage running workers by PID:

```powershell
.\.venv\python.exe tools\sokurl.py list
.\.venv\python.exe tools\sokurl.py status --pid 1234
.\.venv\python.exe tools\sokurl.py shutdown --pid 1234
```

## What SokuRL Exposes

### Structured battle state

Every actual simulation update produces a `RawFrameState` containing:

- Scene, battle mode/submode, stage, round, timer, weather, and RNG seed
- P1/P2 position, velocity, facing, HP, spirit, cards, action state, animation,
  hitstop, untech, airborne state, flags, and effective logical input
- Up to 64 objects per player with explicit overflow reporting
- Object ownership/type, action and animation state, position, velocity, HP,
  hitstop, hit/hurt boxes, and active state
- A canonical FNV-1a-64 state hash

The shared-memory ABI uses a 512-frame SPSC ring. A consumer can detect any
recording loss through `dropped_frames`; overflow is never hidden.

### Logical actions and frame control

SokuRL injects the game's logical `KeyInput`, not Windows keyboard events. This
removes window focus, keyboard layout, and IME from the control path.

Available controls include:

- Atomic P1/P2 direction and A/B/C/D/card input
- Pause and resume
- `+1F` and `+NF`
- Deterministic `Goto frame N` and freeze
- `-1F` through `Goto(current - 1)`

One SokuRL frame is one call to the original `BattleManager::onProcess`. The
unlimited worker does not batch multiple updates into one main-loop iteration or
substitute a custom timestep.

### Deterministic reconstruction

SokuRL does not copy raw process memory. It reconstructs a target state through
the original simulation:

```text
deterministic frame-zero checkpoint
  -> replay recorded P1/P2 logical inputs
  -> simulate to the target frame
  -> restore documented scalar state when required
  -> freeze and compare the complete state hash
```

The scalar patch is limited to timer/weather, position, velocity, facing, HP,
spirit, and card counters. Actions, animation, hitstop, projectiles, and object
lists must match through re-simulation. A mismatch reports the first divergent
simulation frame and a field/object diff.

### Replay capture and seek

ReplayDnD starts a `.rep` directly. SokuRL freezes replay frame zero and can
play normally or simulate to a requested frame:

```powershell
.\.venv\python.exe tools\sokurl.py replay path\match.rep
.\.venv\python.exe tools\sokurl.py replay path\match.rep --frame 5000
```

### ScenarioRunner v1

ScenarioRunner saves reproducible Practice anchors and runs per-frame opponent
scripts from them:

```powershell
.\.venv\python.exe tools\sokurl.py anchor save graze_test --pid 1234
.\.venv\python.exe tools\sokurl.py anchor load graze_test
.\.venv\python.exe tools\sokurl.py script run scenarios\graze_test.yaml --pid 5678
```

Scripts support waits, numpad-relative directions, direction/button inputs,
explicit raw sequences, and nested repeats:

```yaml
anchor: graze_test
opponent: p1
script:
  - wait: 30
  - raw: [{"input":"5B","frames":3}]
  - wait: 45
  - raw: [{"input":"2","frames":2},{"input":"1","frames":2},{"input":"4C","frames":3}]
  - wait: 60
```

## How VS Workers Start

The stable VS launcher uses the game's Title lifecycle. After Title initializes
its input and profile state, the bridge binds both local players, calls the
game's VS battle-mode routine, and returns through the original Loading path.

It does not write a scene ID or use SkipIntro's incomplete direct VS CSelect
path. The persistent SkipIntro configuration remains the stable Practice preset.

MemoryPatch supplies the existing community multi-instance patch. Each process
then owns an independent mapping:

```text
Local\SokuRLBridge_<pid>
```

## Headless and Unlimited Execution

Headless mode skips only the confirmed complex battle draw/present interval.
The window, D3D device, resources, audio, animation, effects, collision, RNG,
and object updates remain initialized and active.

Unlimited mode changes the VS battle frame wait from blocking to non-blocking.
The original main loop and one-update-per-frame semantics remain intact.

## Validation

The repository includes small, sanitized JSON artifacts for the headline
results:

- [M2-B determinism and throughput](docs/validation/m2b-summary.json)
- [Expert replay capture and seek](docs/validation/replay-summary.json)
- [Practice reconstruction and ScenarioRunner](docs/validation/reconstruction-summary.json)

### Determinism

- Rendered, headless paced, and headless unlimited execution matched for the
  required 3,000-frame trace and a 10,000-frame long run.
- The long-run final hash was `CC6DB833624518B5`.
- All three modes recorded zero divergent frames and zero dropped records.
- Coverage included A/B/C attacks, projectile spawn/movement/despawn, damage,
  hitstop, untech, airborne state, and corner interaction.

### Throughput

These measurements come from one development machine and are not portable
hardware guarantees.

| Workers | Mean sim-FPS / worker | Aggregate sim-FPS | Mean system CPU | Dropped | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 28,059.60 | 28,059.60 | 27.22% | 0 | Stable |
| 4 | 24,063.98 | 96,255.94 | 49.63% | 0 | Stable |
| 8 | 18,832.74 | 150,661.89 | 92.75% | 0 | Stable |
| 16 | 12,935.27 | 206,964.27 | 99.84% | 139 | CPU-saturated |

Eight workers were recording-stable on the test machine. At 16 workers, all
simulations remained alive but two frame rings overflowed under full CPU
contention, so the 16-worker result is a saturation measurement rather than an
accepted lossless configuration.

### Other acceptance results

- Local VS startup: 20/20 successful runs
- Headless startup: 20/20 successful runs
- Practice reconstruction targets: all hashes matched
- Scenario anchors: 20/20 fresh-process loads
- Scenario projectile runs: 3/3 identical traces and final hashes
- Expert replay: 14,954 contiguous frames, zero drops, up to 55 live objects
- Unit tests: 17/17 passed
- Two-process frame stepping: PID-isolated

## Runtime Modules

The tested local setup uses these SWRSToys modules:

```ini
[Module]
WindowResizer=modules/WindowResizer/WindowResizer.dll
SokuRLBridge=modules/SokuRLBridge/SokuRLBridge.dll
SkipIntro=modules/SkipIntro/SkipIntro.dll
MemoryPatch=modules/MemoryPatch/MemoryPatch.dll
ReplayDnD=modules/ReplayDnD/ReplayDnD.dll
```

Community module sources live under `third_party/SokuMods/`; the local game
installation remains outside version control.

## Build the Native Bridge

Open an **x86 Developer Command Prompt for Visual Studio**, then run:

```cmd
cmake -S native\SokuRLBridge -B native\SokuRLBridge\build -A Win32
cmake --build native\SokuRLBridge\build --config Release --target SokuRLBridge
```

The output is:

```text
native\SokuRLBridge\build\Release\SokuRLBridge.dll
```

Do not build the bridge as x64.

## Reproduce the Validations

```powershell
.\.venv\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\python.exe tools\frame_validation.py
.\.venv\python.exe tools\replay_validation.py
.\.venv\python.exe tools\scenario_validation.py --loads 20 --runs 3
.\.venv\python.exe tools\vs_stress_validation.py --headless --runs 20 --frames 300
.\.venv\python.exe tools\headless_validation.py --unlimited --frames 10000
.\.venv\python.exe tools\unlimited_benchmark.py --mode unlimited --workers 8 --duration 5
```

Full local run logs are written to `logs/validation/` and excluded from Git.
The curated summaries under `docs/validation/` are versioned.

## Known Limits

- Only th123 1.10a with the documented executable hash is supported.
- Headless workers still create a window and initialize D3D, resources, and
  audio. SokuRL is not a standalone reimplementation of the game.
- Eight workers were lossless on the validation machine; 16 workers saturated
  CPU and overflowed two frame rings.
- Practice currently accepts P2 movement input but filters P2 B/C attacks. The
  accepted projectile ScenarioRunner script controls `opponent: p1`.
- `24C` remains an explicit raw input sequence rather than a named macro.
- Checkpoints use deterministic short-range reconstruction, not raw savestates.
- Trainer-facing spaces, rewards, PPO, self-play, and adaptation are not yet
  implemented.

## Repository Layout

```text
native/SokuRLBridge/     Win32 bridge and shared-memory ABI
tools/sokurl.py          launcher and process lifecycle CLI
tools/bridge_shared.py   Python ABI and command client
tools/scenario_runner.py anchors and scripted actions
tools/*_validation.py    deterministic and runtime validation harnesses
docs/validation/         publishable machine-readable summaries
scenarios/               deterministic script examples
third_party/SokuMods/    community source/build dependency
th123_jp/                local game runtime, excluded from Git
```

## Development Install

```powershell
.\.venv\python.exe -m pip install -e .
```
