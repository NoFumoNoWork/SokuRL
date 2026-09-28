# SokuRL

Python scaffolding for a Touhou Hisoutensoku reinforcement-learning environment.

The project code lives directly in this repository. The local game installation stays isolated in `th123_jp/`, community tools belong in `third_party/`, and extracted game data belongs in `extracted/`.

All project Python commands use the repository-local `.venv`:

```powershell
.\.venv\Scripts\python.exe <command>
```

## Current status

The Win32 `SokuRLBridge` provides PID-isolated shared memory, per-simulation-frame
state capture, logical P1/P2 input stepping, pause, and deterministic hybrid
checkpoint reconstruction. ScenarioRunner v1 adds reproducible anchors and
per-frame scripted actions. PPO, rewards, and training are intentionally out of
scope for this milestone.

Practice remains the default SkipIntro preset. Local VS Player is available
through the normal Title/Loading lifecycle without editing the persistent preset:

```powershell
.\.venv\python.exe tools\sokurl.py practice
.\.venv\python.exe tools\sokurl.py vs
```

## ScenarioRunner v1

Anchors are JSON documents under `anchors/`. They store the checkpoint identity,
target frame and hash, and the setup input/scalar history needed to reconstruct
the target without a raw memory savestate.

```powershell
.\.venv\python.exe tools\sokurl.py anchor save graze_test --pid 1234
.\.venv\python.exe tools\sokurl.py anchor load graze_test
.\.venv\python.exe tools\sokurl.py script run scenarios\graze_test.yaml --pid 5678
```

`anchor load` starts a fresh Practice process and leaves it frozen at the
validated anchor. With `--pid`, it gracefully closes that existing instance
before creating the replacement. Scripts select `opponent: p1` or `opponent: p2`
and support `wait`, numpad-relative direction/button tokens such as `5B`, explicit
per-frame `raw` sequences, and nested `repeat` blocks. See
`scenarios/graze_test.yaml` for the accepted format.

## Verify the game executable

```powershell
.\.venv\Scripts\python.exe scripts\00_check_game.py
```

The supported executable is Touhou Hisoutensoku 1.10a with MD5 `DF35D1FBC7B583317ADABE8CD9F53B2E`.

`th123.exe` is a 32-bit x86 process. The current `.venv` interpreter is 64-bit, which is valid for out-of-process Windows memory access, but target addresses and pointers must always use 32-bit/4-byte semantics. Any DLL or other code loaded into the game process must be built for x86.

## Development install

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```
