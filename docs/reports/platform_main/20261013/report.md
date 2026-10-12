# Platform-only Main Update

Updated: 2026-10-13 by Codex. Base: b40607c.

## Scope

Only CPU Platform code is copied from the RL working tree: Win32 Bridge,
ABI12 command/state client, launcher, recording/replay/frame/scenario tools,
optional x64 host hash helper, process health and reset/cost validation.
`profile_training_cost.py` measures host/Bridge costs; it does not train a model.

No `src/` or `config/` changes, trainer, learning node, policy, checkpoint,
model, GPU simulator, original EXE/DAT or binary snapshot is included. Existing
main base files are retained. The RL working tree is not switched or merged.
Optional native trace hooks remain OFF; no new tracing correctness claim.
The external experimental Giuroll branch adapter is not distributed here.

## Evidence

`copy_manifest.json` records42 source/evidence copies with old/new SHA256.
README/AGENTS/ignore changes and the capability test are additional deliberate
main-only edits. `publication_manifest.json` binds staged files to their hashes.
Four small historical PASS JSONs are curated here; the full reset experiment,
failed attempts and RL worker evidence remain in the original local report tree.
They are not regenerated or claimed to be part of this main checkout.

Actual isolated-worktree results:

- Win32 Bridge configure/build: PASS (existing compiler/upstream warnings).
- Optional x64 HostBench build: PASS.
- pytest before helper build:30PASS/4SKIPPED; final:38PASS/0SKIPPED.
- `reset_recheck.json`:2instances,700frames,4cycles, native RNG/active-input/
  KO and unselected-instance isolation PASS using the previous deployed DLL.
- `rebuilt_dll_reset.json`:2instances,200frames,3cycles, native RNG/input/
  unselected-instance isolation PASS using the newly built main DLL.
- `default_recheck.json`: full10000frame new-DLL normal/headless/unlimited
  PASS, zero drops, elapsed178.915s. The existing historical PASS was not
  substituted for this run.

New main DLL SHA256:
0CBBA65B36D97C7EE7D54E36353168028A5D0E0F7B9210356B6469625D5F0430.
Previously deployed DLL SHA256:
DC81A1526B2DA64D134AD033AC20C744D965BBA85E74CB7C5291F9A8661BD54D.
Protected th123.exe MD5 remains DF35D1FBC7B583317ADABE8CD9F53B2E.

## Reproduce / Roll Back

```powershell
.\.venv\python.exe -B -m pytest tests -q
cmake -S native/SokuRLBridge -B native/SokuRLBridge/build -A Win32
cmake --build native/SokuRLBridge/build --config Release
cmd /c native\BridgeTools\build_host.cmd
.\.venv\python.exe -B tools/headless_validation.py --frames 10000 --unlimited
```

These tests require the documented legal game installation and native modules.
Local ignored junctions to an existing game/dependency installation were used;
they are not Git assets or a substitute for dependency bootstrap instructions.
This environment used the existing RL `.venv` interpreter, not a new global
installation. The publication does not need Torch, Hydra or training packages.

Rollback a published commit with `git revert <publication-commit>` on main,
not a force push or reset of the dirty RL worktree. Local DLL rollback is the
exact `Bridge.before.dll` backup in the original working-tree report folder;
stop owned games before restoring a mod. Do not alter the original EXE/DAT.
