# SokuRL

中文文档：[安装与验收](docs/installation.md) · [架构与能力边界](docs/architecture.md) · [状态决策树基线](docs/baselines.md) · [社区规则策略](docs/community-ai.md) · [双人博弈与胜率评估](docs/strategy-evaluation.md) · [后续开发计划](docs/development-plan.md)。

实测报告：[Linux 并行采样与 RL 施工决策](docs/linux-performance.md)。

当前原生模块拒绝 `GotoFrame`，不能把下文历史上的帧导航与场景重建说明当作已完成的任意状态恢复接口。ABI 7 新增 `ResetEpisode`，通过游戏内部场景流程重建对局，保留游戏进程；完整轨迹验证正在进行。公共接口为 PettingZoo 双人环境，默认提供经过可见性和精度过滤的状态，也支持真实图像。已接入 TorchRL、BenchMARL IPPO、OpenSpiel NFSP 和 PSRO；最终策略胜率和联网人机对战尚未验收。入口与分层见[双人环境文档](docs/multi-agent-env.md)。

AI 工作进程默认静音，配置为 `runtime.mute_audio=true`。模块只在该游戏进程内把音乐和音效的音量设为零，不修改系统总音量、原始游戏文件或保存的游戏音量配置。

训练可选[学习包装层](docs/learning-wrappers.md)：血量势函数奖励、公开相对位置、己方按键历史和 90 种按键组合；完整 576 动作仍可选。固定规则对手训练支持 PPO 和带 LSTM 记忆的 RecurrentPPO，双人训练支持 IPPO、NFSP 和 PSRO。接口检查通过不代表已经达到最终胜率目标。

使用标准 Windows 虚拟环境时，Python 路径为 `.venv\Scripts\python.exe`；
下文的 `.venv\python.exe` 是原开发环境的路径。请使用实际存在的解释器路径。
Python 依赖安装不包含游戏本体、SWRSToys 模块或原生桥接 DLL。

SokuRL is a Windows control and state-extraction layer for Touhou Hisoutensoku
(`th123`) 1.10a. It currently provides deterministic local Practice and VS Player
automation, per-simulation-frame battle state, logical input control, replay
seeking, reproducible scenario anchors, multi-instance isolation, and an
experimentally validated faster-than-real-time VS worker.

The public RL interface is a two-player PettingZoo environment. Training adapters
use BenchMARL IPPO and OpenSpiel NFSP and PSRO. Policy quality and network play
still need validation. See the current Chinese RL guide linked above.

## Supported Runtime

- Game: Touhou Hisoutensoku 1.10a
- Executable: `th123_jp/th123.exe`
- Required MD5: `DF35D1FBC7B583317ADABE8CD9F53B2E`
- Game architecture: Win32/x86
- Python environment: repository-local Python 3.11 x64 in `.venv`
- Native compiler: MSVC Win32/x86
- Bridge ABI: version 7

The 64-bit Python process controls the 32-bit game out of process. All pointers
read from th123 memory are therefore represented explicitly as 32-bit values.
Every DLL loaded by th123 must be built for x86.

Verify the installed game before running anything:

```powershell
.\.venv\python.exe scripts\00_check_game.py
```

SokuRL does not patch `th123.exe`, `th123a.dat`, `th123b.dat`, or `th123c.dat`
on disk.

## Runtime Stack

The current game installation loads these SWRSToys modules:

```ini
[Module]
WindowResizer=modules/WindowResizer/WindowResizer.dll
SokuRLBridge=modules/SokuRLBridge/SokuRLBridge.dll
SkipIntro=modules/SkipIntro/SkipIntro.dll
MemoryPatch=modules/MemoryPatch/MemoryPatch.dll
ReplayDnD=modules/ReplayDnD/ReplayDnD.dll
```

- **SokuRLBridge** exposes state and logical controls through shared memory.
- **SkipIntro** owns the stable default Practice preset and initial selections.
- **MemoryPatch** enables the existing community `AllowMultiInstance` patch.
- **ReplayDnD** loads a `.rep` directly from the th123 command line.
- **WindowResizer** configures the game window without changing simulation.

The persistent SkipIntro preset remains Practice (`type=8`). The VS launcher
temporarily starts at Title under a named mutex, passes process-local launch
parameters to the bridge, and restores the exact original INI bytes.

## Quick Start

All Python commands use the repository-local environment:

```powershell
.\.venv\python.exe tools\sokurl.py practice
.\.venv\python.exe tools\sokurl.py vs
.\.venv\python.exe tools\sokurl.py vs --headless
.\.venv\python.exe tools\sokurl.py vs --headless --unlimited
```

| Command | Behavior |
| --- | --- |
| `practice` | Starts the configured local Practice preset and sends acknowledged in-game confirm inputs until battle is ready. |
| `vs` | Starts local VS Player through the legitimate Title, Loading, and Battle lifecycle. |
| `vs --headless` | Keeps all game initialization and simulation but skips the confirmed complex battle-render interval. |
| `vs --headless --unlimited` | Also removes the VS battle wall-clock wait while preserving one original update per main-loop iteration. |

`--unlimited` is rejected unless `--headless` is also present. Practice and
normal VS never enable the unlimited pacing hook.

Process management is PID-aware:

```powershell
.\.venv\python.exe tools\sokurl.py list
.\.venv\python.exe tools\sokurl.py status --pid 1234
.\.venv\python.exe tools\sokurl.py shutdown --pid 1234
```

Shutdown requests a normal window close first and terminates only after a
timeout.

## Local VS Player Bootstrap

The supported VS path does not use SkipIntro `type=3` and does not write a scene
ID. After the original `Title::onProcess` has initialized Title-owned state,
SokuRLBridge:

1. Selects fallback local input ownership.
2. Calls the game's `setBattleMode(VSPLAYER, PLAYING1)` routine.
3. Initializes both profiles, key managers, palettes, and effective decks.
4. Sets the configured stage and music.
5. Returns `SCENE_LOADING` from Title so the original game creates the battle.

The old direct `type=3` CSelect experiment is not supported. It skipped P2
device/profile ownership and produced repeatable invalid-parameter crashes.

## PID-Isolated Multi-Instance Control

Each th123 process publishes a separate mapping:

```text
Local\SokuRLBridge_<pid>
```

Commands, state, frame counters, ring buffers, and checkpoints are bound to that
PID. Stepping or pausing one worker does not advance another worker.
SokuRL relies on MemoryPatch's existing `AllowMultiInstance` implementation and
does not patch the single-instance check itself.

## Per-Frame Battle State

The canonical simulation boundary is `BattleManager::onProcess`. A SokuRL frame
increments only after one call to the original battle update. Rendering,
wall-clock time, Python polling, and paused main-loop calls do not increment the
simulation frame.

Each `RawFrameState` contains:

- Global scene, battle mode/submode, stage, round, timer, weather, and RNG seed.
- P1/P2 character ID, position, velocity, facing, HP, spirit, card gauge/count,
  hand IDs, action/sequence/subsequence, animation frame, elapsed action time,
  hitstop, untech, airborne state, frame/attack flags, and effective logical
  input.
- Up to 64 objects per player with explicit overflow reporting.
- Object owner/list identity, runtime type identity, action and animation state,
  position, velocity, direction, HP, hitstop, hit/hurt-box counts, character
  index, and active state.
- A canonical FNV-1a-64 deterministic state hash.

ABI v6 uses a 512-frame single-producer/single-consumer ring and a 4096-frame
native reconstruction history. A lossless recording requires
`dropped_frames == 0`; overflow is reported and is never silently ignored.

## Logical Input, Pause, and Frame Navigation

The bridge operates on the game's logical `KeyInput`, not Windows keyboard
events. The bridge and Python reconstruction tools together support:

- Atomic P1/P2 logical input for exactly one simulation frame.
- Horizontal/vertical directions and A/B/C/D/card inputs.
- Pause and resume.
- `+1F` and `+NF` stepping.
- `Goto frame N` through fresh-process checkpoint reconstruction, followed by freeze.
- `-1F` as `Goto(current - 1)`.

Input is applied at the game's `KeymapManager::SetInputs` boundary before the
BattleManager consumes it. This avoids window focus, keyboard layout, and IME
dependencies.

## Checkpoints and Deterministic Reconstruction

SokuRL does not implement a raw process-memory savestate. Frame loading uses:

```text
deterministic frame-zero checkpoint
  -> replay recorded P1/P2 logical inputs
  -> run the original game simulation to the target
  -> restore only documented simple scalar fields when required
  -> freeze and validate the complete state hash
```

The documented scalar patch is limited to timer/weather, player position and
speed, facing, HP, spirit, and card counters. Actions, animation, hitstop,
flags, hands, projectiles, and object lists have no direct write path and must
be reconstructed by the original simulation.

Hash mismatches stop reconstruction and report the first divergent simulation
frame plus a field-by-field and object-by-object diff.

## Replay Playback and Seeking

ReplayDnD owns `.rep` loading. SokuRLBridge freezes the first replay battle state
at frame zero and then uses the same state, input, stepping, and validation
machinery as Practice.

```powershell
.\.venv\python.exe tools\sokurl.py replay path\match.rep
.\.venv\python.exe tools\sokurl.py replay path\match.rep --frame 5000
```

The first command starts normal replay playback. The second simulates from
replay frame zero to frame 5000 and leaves the game frozen there.

## ScenarioRunner v1

ScenarioRunner saves reproducible Practice anchors and runs deterministic
per-frame scripts from them.

```powershell
.\.venv\python.exe tools\sokurl.py anchor save graze_test --pid 1234
.\.venv\python.exe tools\sokurl.py anchor load graze_test
.\.venv\python.exe tools\sokurl.py script run scenarios\graze_test.yaml --pid 5678
```

Anchor documents under `anchors/` use `SokuRLAnchor/v1`. They contain the
checkpoint identity, target frame/hash, and the dual-input/simple-state setup
history needed to reconstruct the target in a fresh process. Loading leaves the
new process frozen only after the reconstructed hash matches.

Scripts support `wait`, numpad-relative directions, direction plus A/B/C/D,
explicit per-frame `raw` sequences, and nested `repeat` blocks.

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

`24C` remains an explicit raw sequence because no ambiguous move macro is
promoted without runtime evidence.

## Headless Rendering and Unlimited Pacing

M2-A installs a battle-only render dispatcher at `0x00407FAE`. In headless
scene 5 it skips `0x00407FB4..0x00408047` and rejoins at `0x00408048`.
Initialization, HWND/D3D, textures, resources, audio, animation/effects,
collision, objects, RNG, and loading remain active.

M2-B redirects only the `WaitForSingleObject` operand at `0x00419689` inside the
frame-wait function at `0x004195E0`. In unlimited local VS battle the original
event is polled with timeout 0 instead of `INFINITE`. It does not call multiple
BattleManager updates, invent a timestep, or change hitstop/input/animation
cadence.

## Acceptance Results

All figures below were measured on the current development machine and are not
portable hardware guarantees.

### Core Bridge and Reconstruction

- Practice reached `PRACTICE_READY` after seven acknowledged logical confirms.
- Projectile Practice trace: 1043 frames, 21 P1 objects, no overflow.
- Reconstruction targets 472, 479, 916, and 917 matched exactly.
- Repeated forward/backward seeks matched, including `-1F` then `+1F` identity.
- Two simultaneous processes retained distinct mappings and independent frame
  counters.

Local report: `logs/validation/frame-validation-20260928-100935.json`

### Replay Validation

- Recorded 14,954 contiguous replay states with zero dropped frames.
- Observed up to 39/55 P1/P2 objects plus extensive airborne and hitstop state.
- Reconstruction targets 1558 and 5190 matched every stable field and object.

Local report: `logs/validation/replay-validation-20260928-104655.json`

### ScenarioRunner

- Anchor reconstruction passed 20/20 fresh-process loads.
- Projectile script passed 3/3 load/run/reset cycles with identical traces and
  final hashes.
- Accepted script observed `5B`, explicit raw `24C`, and up to 15 objects.

Local report: `logs/validation/scenario-validation-20260928-152941.json`

### VS Player and M2-A

- Title-context VS startup passed 20/20 runs with at least 300 battle frames.
- Both players performed logical attacks, created projectiles, and dealt
  persistent VS damage.
- Headless startup passed 20/20 runs.
- Normal versus headless matched all 3000 per-frame hashes, including projectile
  spawn/movement/despawn, damage, hitstop, untech, airborne state, and corner
  interaction.
- Both comparison processes reported zero dropped frames.

Local reports: `logs/validation/vs-stress-20260928-203316.json` and
`logs/validation/m2a-validation-20260928-2113.json`

### M2-B Determinism and Performance

- Normal rendered, paced headless, and unlimited headless matched for 3000
  required frames and a 10,000-frame long run.
- The 10,000-frame final hash was `CC6DB833624518B5`; no divergence or dropped
  record occurred in any mode.
- Single-instance rates were 60.03 rendered, 60.00 paced headless, and 28,059.60
  unlimited simulation frames per second.

| Workers | Mean sim-FPS per worker | Aggregate sim-FPS | Mean system CPU | Dropped records | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 28,059.60 | 28,059.60 | 27.22% | 0 | Stable |
| 4 | 24,063.98 | 96,255.94 | 49.63% | 0 | Stable |
| 8 | 18,832.74 | 150,661.89 | 92.75% | 0 | Stable |
| 16 | 12,935.27 | 206,964.27 | 99.84% | 139 | Saturated; not recording-stable |

The accepted full-recording worker count on this machine is eight. At 16
workers all simulations remained alive, but two 512-slot frame rings overflowed
under full CPU contention and one process missed graceful shutdown timeout.
Per-process Direct3D GPU utilization was not available, so no GPU number is
claimed.

Local report: `logs/validation/m2b-validation-20260928-2150.json`

### Regression and File Safety

- Existing unit tests: 17/17 passed.
- Default Practice behavior passed after M2-A and M2-B.
- Normal VS remained paced at approximately 60 simulation FPS.
- Deployed bridge is Win32/x86.
- Current deployed bridge SHA-256:
  `BDF68AF07B9780805A16BCFD3CAD20C089E96F3C7C867B6524802055C6EC5ACA`.
- `th123.exe` MD5 remains `DF35D1FBC7B583317ADABE8CD9F53B2E`.
- Protected `.dat` hashes remained unchanged.

## Validation Commands

```powershell
.\.venv\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\python.exe tools\frame_validation.py
.\.venv\python.exe tools\replay_validation.py
.\.venv\python.exe tools\scenario_validation.py --loads 20 --runs 3
.\.venv\python.exe tools\vs_stress_validation.py --runs 20 --frames 300
.\.venv\python.exe tools\vs_stress_validation.py --headless --runs 20 --frames 300
.\.venv\python.exe tools\headless_validation.py --unlimited --frames 10000
.\.venv\python.exe tools\unlimited_benchmark.py --mode unlimited --workers 8 --duration 5
```

Runtime JSON reports are written under `logs/validation/`.

## Native Build

Initialize the Visual Studio x86 environment and build Release:

```cmd
call "T:\VisualStudio\Common7\Tools\VsDevCmd.bat" -arch=x86 -host_arch=x64
cmake -S native\SokuRLBridge -B native\SokuRLBridge\build -A Win32
cmake --build native\SokuRLBridge\build --config Release --target SokuRLBridge
```

The output is `native\SokuRLBridge\build\Release\SokuRLBridge.dll`. Do not build
the bridge as x64.

## Known Limits

- Only th123 1.10a with the documented executable hash is supported.
- The accelerated worker still creates a window and initializes D3D, resources,
  and audio. It is render-skipping, not a standalone simulator.
- The 16-worker benchmark exceeded lossless frame-consumer capacity on the test
  machine; eight workers were stable.
- In the Practice dummy path, P2 movement is controllable but P2 B/C attacks are
  filtered even when effective input is captured. The accepted ScenarioRunner
  projectile scenario therefore controls `opponent: p1`.
- `24C` is exposed as a raw sequence rather than a guaranteed named macro.
- Checkpoint history is short range and deterministic; it is not a raw memory
  savestate.
- PPO, training replay datasets, and policy training are not implemented.

## Repository Layout

```text
native/SokuRLBridge/     injected Win32 bridge and shared-memory ABI
tools/sokurl.py          launcher and process lifecycle CLI
tools/bridge_shared.py   Python ABI and command client
tools/frame_validation.py
tools/replay_validation.py
tools/scenario_runner.py
tools/scenario_validation.py
tools/headless_validation.py
tools/unlimited_benchmark.py
scenarios/               deterministic script examples
anchors/                 generated anchor documents
logs/validation/         machine-readable acceptance reports
third_party/SokuMods/    community source/build dependency
th123_jp/                local game runtime; protected original files
```

Detailed local structures and milestone history are maintained in
`STRUCTURES.md`, `PROGRESS.md`, and `HISTORY.md`. These files and runtime
validation JSON are currently excluded from Git by the repository ignore rules.

## Development Install

```powershell
.\.venv\python.exe -m pip install -e .
```
