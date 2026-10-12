# Snapshot-free CPU Fresh Reset

Updated: 2026-10-13 by Codex.

## Result

PASS for the tested fixed Reimu/Marisa, Clear, cards-disabled state-observation
path. No first-frame global snapshot, process restart or Giuroll is required.
Initial cold loading is not eliminated. Enable `SOKURL_FRESH_RESET=1` before
launch, check `BridgeClient.fresh_reset_enabled`, call `reset_episode(seed)`.
RL worker integration additionally supports `runtime.fresh_reset=true`.

## Mechanism and Evidence Boundary

Original onArenaStart0x481810 calls seed initialization0x409890, constructs both
characters at0x4818D4/0x4818E7 (0x46EAD0), then world reset0x479520. Native character
reset callbacks clear objects/resources and invoke common reset. FreshReset
retains loaded characters, executes original onExit followed by onArenaStart,
skipping only those two construction calls on warm resets. Constructor-defined
hit/hurtbox counts are zeroed first. ABI12 and original battle update remain
unchanged. Per-process settings and thread-local reuse guards isolate instances.

Original timing showed roughly2.7s resets, dominated by the two character
constructions. Constructor RNG fingerprints did not change on the observed
fixed-character paths. Exact exported-state and complete MT624/index hashes are
compared to independent original cold-start references. This is not proof of
every hidden byte or every character constructor branch.

## Actual Runs

- `parity_v2.json`: 2workers,8resets,1000frames; mean42.774ms.
- `parity_four_long.json`: 4workers,32resets,3000frames; mean48.376ms.
- `parity_active_ai.json`: 4workers,32resets,2000frames; mean47.566ms;
  active objects, damage and immediate input included.
- `parity_eight_reversed.json`: 8workers,64resets,1000frames; mean64.173ms;
  opposite character ordering and unselected-instance isolation PASS.
- `final_rng_terminal.json`: 2workers,8resets,700frames; mean43.973ms;
  native RNG and KO at frame4918 followed by reset PASS.
- `reset_burst64.json`: 4workers,256resets; PASS. Mean102.757ms while another
  native regression ran concurrently; not a controlled throughput comparison.
  Private bytes remained bounded over this short test, not a long PPO guarantee.
- `worker_state/result.json` and `worker_state_final/result.json`: public state
  worker reset, observations and subsequent AI inputs PASS.
- `default_parity.json`: normal/headless/unlimited10000frames, zero drops, PASS;
  measured179.736s. This tests exported state, not whole-memory identity.

## Failures Preserved

`parity_trial.json` initially failed on frame0 hurtbox counts. Fixed using known
native constructor initialization, not relaxed comparisons. Scene pacing and
texture cache tests failed to improve reset cost and were removed; source of
the rejected new cache experiment is retained under `rejected_texture_cache/`.

`worker_image/result.json` found optional RGB disagreement at frame0 despite
equal battle hashes. RGB is not required for the accepted state-only task and
can be disabled. Image-mode parity is NOT VERIFIED; the worker rejects image
mode with fresh reset rather than silently returning inconsistent pixels.

## Reproduce and Roll Back

```powershell
.\.venv\python.exe -B tools/validate_fresh_reset.py --workers 2 --frames 700 --cycles 4 --active-reset --immediate-input --terminal-reset --characters 0 1 --output docs/reports/fresh_reset/recheck.json
.\.venv\python.exe -B tools/headless_validation.py --frames 10000 --unlimited
```

Fresh reset defaults OFF. Disable its environment switch to use the original
scene-reload path. The pre-change DLL is retained in `before/SokuRLBridge.dll`;
restore only after stopping owned games and with the matching Python ABI.
Do not revert the entire dirty repository. Protected EXE MD5 remains
DF35D1FBC7B583317ADABE8CD9F53B2E. Validated final DLL SHA256:
DC81A1526B2DA64D134AD033AC20C744D965BBA85E74CB7C5291F9A8661BD54D.

## Not Tested

All20characters, configuration changes, external God-AI Lua/DLL restart state,
pixel parity, whole hidden-state equivalence and long training. Existing
same-world branch checkpoints are a separate opt-in diagnostic facility.
