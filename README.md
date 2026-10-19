# SokuRL Platform

CPU-based control of the original Touhou Hisoutensoku 1.10a engine. Platform
provides native Bridge state/events, frame input and stepping, multi-instance
process control, reset, replay/image facilities and reusable worker transport.
It is not an independent game simulator or GPU implementation.

This is the Platform README. The optional learning module is distributed on the
RL branch as a complete rl/ folder placed beside platform/. It has its own
rl/README.md and is not part of the Platform branch's source payload.

## Requirements

- Windows and a legally supplied TH123 1.10a installation.
- Python3.11 x64; the existing local environment uses .venv/python.exe.
- MSVC, a Windows SDK and CMake for native builds; injected DLLs are Win32/x86.
- SokuMods/SokuLib and the pinned dependencies in platform/config/dependencies.lock.json.

Game files, third-party packages, compiled DLLs, private configuration and local
evidence are external resources, not included in this repository's source release.
Original th123.exe MD5: DF35D1FBC7B583317ADABE8CD9F53B2E.
Do not overwrite the original EXE or DAT files to install the Bridge.

## Python Setup

Prepare a Python3.11 x64 virtual environment first. A standard Windows venv may
use .venv/Scripts/python.exe; substitute its actual interpreter below. Do not
recreate an existing working environment merely to match the example path.

```powershell
.\.venv\python.exe -m pip install -e "./platform[dev]"
```

Platform's package is sokurl_platform, not platform; it does not depend on
PyTorch, Gymnasium, SB3 or the RL package. Source-only tools also work through
platform/tools entries. No change to a global editable registration is needed.
Clean-machine installation has not been independently certified by the local
migration tests; dependency versions and native resources must still be prepared.

## Native Build And Resources

Review platform/config/dependencies.lock.json and the existing bootstrap/verify
scripts before preparing third-party sources. The bootstrap can clone/check
submodules and apply a pinned patch: do not run it against modified assets blindly.

```powershell
.\platform\scripts\build_bridge.cmd
.\platform\native\BridgeTools\build_host.cmd
.\.venv\python.exe -B -m pytest platform/tests -q -p no:cacheprovider
```

Build outputs: platform/build/bridge/Release/SokuRLBridge.dll (x86), and the
optional platform/native/BridgeTools/build/HostBench.dll (x64 Python-side helper).
Building is separate from deployment. MSVC/CMake paths can be supplied through
the existing build script's environment overrides.

Default runtime resources are th123_jp/ and third_party/ in the workspace root.
SOKURL_GAME_DIR can select another prepared game installation. Match the game,
loader and module versions; simply compiling a DLL does not install a working
runtime. Use platform/tools/BridgeDeployment.psm1 for controlled temporary Bridge
deployment: backup, hash check and finally restoration. It refuses existing game
processes and evidence collisions; verify recovery after interruption.

## Public Interfaces

- sokurl_platform.bridge_shared: Bridge state, frame/control interface and ABI types.
- sokurl_platform.command_input: shared input programs and command execution.
- sokurl_platform.native_batch.NativeGameBatch: raw multi-instance state/step/reset.
- sokurl_platform.batch.EngineBatch: shared scheduler with caller-supplied adapters.
- sokurl_platform.transport.WorkerConnection: generic worker transport.
- Native readers, combat events, replay codecs, images and resource locators live
  in the same package. Legacy tools entries delegate rather than copy their bodies.

A raw batch caller must explicitly configure the required runtime switches.
In particular, fresh-reset validation requires SOKURL_FRESH_RESET=1 at game
startup as well as Clear/cards-disabled configuration; a Python flag cannot
retroactively enable a DLL capability. No normal simulation falls back to an
approximate rewritten engine.

## Verification And Limits

An opt-in original CPU launch is available through Platform's existing
`sokurl._launch_vs_group_from_title(..., native_cpu=True)`. It selects native
VS COM with the externally controlled player on the left and original CPU on
the right. Controlled steps never overwrite the CPU player's inputs. The
default remains VS Player. Original REP recording remains disabled for CPU.
Snapshot-free CPU resource-reuse reset has a separate opt-in:
`SOKURL_NATIVE_CPU_FRESH_RESET=1` together with `SOKURL_FRESH_RESET=1`,
headless non-image observations, cards disabled, Marisa-left/Reimu-right.
The Bridge and launcher reject unsupported combinations; rebuild the Win32
Bridge before using this capability. It reuses the existing native arena
exit/start path, not a frame-zero snapshot. Two-instance tests cover alternating
seeds, RNG and 1024-frame new-process parity, neighbor isolation, and KO reset.
These are bounded validation results, not all-character or unbounded-run proof.
With cards disabled, this mode supplies empty decks for both players, rather
than assuming physical key suppression constrains the CPU. CPU difficulty is
inherited from the game configuration; no highest-difficulty claim is made.
RL's dedicated native CPU worker exposes the scoped reset. See `rl/README.md`
for the takeover configuration; ordinary VS Player defaults are unchanged.

Native REP recording is an explicit, separate mode. EngineBatch recording adapters
launch with `SOKURL_ORIGINAL_REPLAY_RECORDING=1`; ABI12 capability mask `0x2` confirms
the protection is active. Until the original input record deque becomes nonempty,
controlled requests remain neutral. The engine, not early direct player-key
overwrites, then performs input latching and release-edge calculation. Ordinary
training/control defaults remain unchanged. Use the matching card/weather setup
for playback; file export alone is not a fidelity verdict. RL checkpoint review
orchestration and its reversible deployment entry are documented in `rl/README.md`.

Offline tests cover layouts, codecs, command programs, interfaces and dependency
boundaries. Native verification requires an installed game and reversible DLL
deployment; no game is started by the offline command above.

Known tested scope includes state-only same-seed reset, partial new-seed reset,
same-PID reuse and four-instance isolation. These bounded cases do not prove every
character, weather, replay or hidden-state path. Fresh-reset RGB parity and
official recording across fresh reset are not verified. NativeGameBatch's raw
adapter explicitly rejects recording; learning-specific adapters are separate.

When reporting results, record the Platform commit, dirty state, Bridge ABI,
game/resources and input/configuration. Do not equate a successful build or short
PPO smoke test with gameplay coverage or policy improvement.
