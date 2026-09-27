# AGENTS.md

## Project

This repository is `SokuRL`, an RL environment for Touhou Hisoutensoku (th123).

Project root:

```text
D:\github\SokuRL
```

Unless explicitly requested otherwise, run project commands from this directory.

---

## Python Environment

This project uses the local Python environment:

```text
D:\github\SokuRL\.venv
```

Always use this environment for Python commands.

Preferred commands:

```powershell
.\.venv\python.exe <script>
.\.venv\python.exe -m pip <args>
```

Examples:

```powershell
.\.venv\python.exe scripts\00_check_game.py
.\.venv\python.exe -m pip list
```

Do not use:

- system Python
- the Conda `base` environment
- another global Python installation
- bare `pip` when `python -m pip` can be used instead

The environment currently uses Python 3.11 x64.

Note that the Python process is 64-bit while the target game is 32-bit. This is acceptable for external process control and `ReadProcessMemory`, but target-process pointer fields must be treated as 32-bit values when reconstructing game data structures.

For example, do not assume that a pointer inside th123 memory has the same width as Python's `ctypes.c_void_p`. Use an explicit 32-bit representation such as `ctypes.c_uint32` when appropriate.

---

## Touhou Hisoutensoku

The target executable is:

```text
D:\github\SokuRL\th123_jp\th123.exe
```

The supported game version is Touhou Hisoutensoku 1.10a.

Expected MD5:

```text
DF35D1FBC7B583317ADABE8CD9F53B2E
```

`th123.exe` is a 32-bit Windows executable.

Any native code, DLL, SokuMods module, or injected component intended to run inside th123 must therefore be built for:

```text
Win32 / x86
```

Never build th123 modules as x64.

Protected original game files include at least:

```text
th123.exe
th123a.dat
th123b.dat
th123c.dat
```

Do not patch, replace, delete, rename, or overwrite these files unless explicitly requested.

Adding separate mod files such as `d3d9.dll`, `SWRSToys.ini`, or files under `modules\` is allowed only when the task explicitly requires it.

---

## Visual C++ / MSVC Environment

Visual Studio Build Tools / Visual Studio is installed at:

```text
T:\VisualStudio
```

The Visual Studio developer environment initializer is:

```text
T:\VisualStudio\Common7\Tools\VsDevCmd.bat
```

For th123-related native builds, initialize it for x86:

```cmd
call "T:\VisualStudio\Common7\Tools\VsDevCmd.bat" -arch=x86 -host_arch=x64
```

After initialization, `cl` and `cmake` are available.

Verified MSVC target:

```text
x86
```

Do not manually add MSVC compiler, SDK, INCLUDE, LIB, or LIBPATH directories to permanent system environment variables. Prefer `VsDevCmd.bat`.

---

## CMake

Visual Studio's CMake executable is located at:

```text
T:\VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
```

Verified version:

```text
cmake 4.3.1-msvc1
```

A normal PowerShell session may not have `cmake` in `PATH`.

Either initialize the Visual Studio developer environment first or invoke CMake by its absolute path.

For SokuMods / th123 native code, always configure for Win32:

```cmd
cmake -S . -B build -A Win32
```

Typical build:

```cmd
cmake --build build --config Release
```

Typical build and install:

```cmd
cmake --build build --config Release --target install
```

If the current default MSVC toolset causes compatibility problems, a VS 2022 v143 toolset is also installed and may be tried explicitly:

```cmd
cmake -S . -B build -A Win32 -T v143
```

Do not switch toolsets unless there is an actual build issue.

Do not use MinGW or Cygwin for SokuMods.

---

## SokuMods

The preferred source/build location for SokuMods is:

```text
D:\github\SokuMods
```

Do not place SokuMods build artifacts inside the SokuRL Python environment.

Keep CMake build artifacts under the SokuMods workspace, for example:

```text
D:\github\SokuMods\build
```

The intended SWRSToys / WindowResizer build target is Win32/x86.

Expected installed SWRSToys files include:

```text
SWRSToys\d3d9.dll
SWRSToys\SWRSToys.ini
SWRSToys\modules\WindowResizer\WindowResizer.dll
SWRSToys\modules\WindowResizer\WindowResizer.ini
```

Before copying generated native files into the game directory, show exactly which files will be copied and what existing files, if any, would be overwritten.

---

## Network and Proxy

The host machine runs a local Xray proxy.

Available local proxy endpoints:

```text
SOCKS5:
127.0.0.1:10808

HTTP:
127.0.0.1:10809
```

Both ports have been verified to work.

Prefer the HTTP proxy for Git/HTTPS commands:

```text
http://127.0.0.1:10809
```

The host Git configuration currently has no persistent `http.proxy` or `https.proxy` setting.

Do not add a permanent Git proxy configuration unless explicitly requested.

For one-off Git operations, prefer command-scoped proxy configuration:

```powershell
git -c http.proxy=http://127.0.0.1:10809 `
    -c https.proxy=http://127.0.0.1:10809 `
    <git command>
```

Example:

```powershell
git -c http.proxy=http://127.0.0.1:10809 `
    -c https.proxy=http://127.0.0.1:10809 `
    ls-remote https://github.com/SokuDev/SokuMods.git HEAD
```

For curl:

```powershell
curl.exe --proxy http://127.0.0.1:10809 https://github.com
```

or via SOCKS:

```powershell
curl.exe --proxy socks5h://127.0.0.1:10808 https://github.com
```

### Important: Codex sandbox proxy variables

The Codex restricted execution environment may inject isolation variables such as:

```text
ALL_PROXY=http://127.0.0.1:9
GIT_HTTP_PROXY=http://127.0.0.1:9
GIT_HTTPS_PROXY=http://127.0.0.1:9
HTTP_PROXY=http://127.0.0.1:9
HTTPS_PROXY=http://127.0.0.1:9
```

Port `127.0.0.1:9` is not the user's real proxy.

If a network command unexpectedly tries to connect to `127.0.0.1:9`, do not conclude that GitHub or the host proxy is unavailable.

For commands where the execution environment permits access to the host proxy, explicitly override the proxy for that command with:

```text
http://127.0.0.1:10809
```

or:

```text
socks5h://127.0.0.1:10808
```

Do not silently modify the user's global proxy configuration.

---

## Git

Before assuming Git/network failure, test with:

```powershell
git --version
```

and:

```powershell
git -c http.proxy=http://127.0.0.1:10809 `
    -c https.proxy=http://127.0.0.1:10809 `
    ls-remote https://github.com/SokuDev/SokuMods.git HEAD
```

This command has previously succeeded when using the HTTP proxy explicitly.

For repositories with submodules, use recursive cloning:

```powershell
git clone --recursive <repository>
```

or, for an existing checkout:

```powershell
git submodule update --init --recursive
```

---

## Development Rules

Prefer small, independently verifiable milestones.

For the RL environment, the intended progression is approximately:

1. Verify game executable/version/process.
2. Verify external input control.
3. Verify reading one known game-state value.
4. Build a live state reader.
5. Build the Gymnasium environment.
6. Add frame synchronization.
7. Begin RL experiments.

Do not prematurely combine input injection, memory reading, frame synchronization, reset logic, and RL training into one large implementation.

When debugging Windows process interaction, preserve the distinction between:

- executable on disk
- running process
- PID
- process handle
- target memory address
- module-relative address
- 32-bit target pointers

Prefer explicit diagnostics over speculative fixes.

---

## Safety Around the Game Installation

The game installation is treated as an external runtime dependency, not as disposable build output.

Before modifying anything under:

```text
D:\github\SokuRL\th123_jp
```

check whether the change is:

- adding a new mod/configuration file, or
- modifying an original game file.

Original game files should remain untouched unless the user explicitly authorizes modification.

Do not regenerate, replace, or patch `th123.exe` merely to make tooling easier.