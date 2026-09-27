# SokuRL

Python scaffolding for a Touhou Hisoutensoku reinforcement-learning environment.

The project code lives directly in this repository. The local game installation stays isolated in `th123_jp/`, community tools belong in `third_party/`, and extracted game data belongs in `extracted/`.

All project Python commands use the repository-local `.venv`:

```powershell
.\.venv\Scripts\python.exe <command>
```

## Current status

Only the initial project boundaries and module layout are established. Memory addresses, input injection, state decoding, environment behavior, and training code are not implemented yet.

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
