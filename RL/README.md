# RL Module

Training, evaluation, policies and RL-facing adapters live here. Platform native
code and tools remain in the parent repository; dependency direction is RL to
Platform only. Existing `soku_rl.*` model class names are preserved.

From the repository root, use the project interpreter:

```powershell
D:\github\SokuRL\.venv\python.exe -B RL/run.py train --config-name train_platform_initial
D:\github\SokuRL\.venv\python.exe -B RL/run.py validate_platform_integration --help
```

Use `RL/run.py` instead of executing tools with an uncontrolled editable package
installation. `SOKURL_PLATFORM_ROOT` optionally selects the Platform checkout.
New outputs belong in `RL/runs`; historical logs remain in their original paths.
Migration evidence and rollback manifests are under `docs/reports/rl_isolation/`.

Optional public-token prototype (default MLP unchanged):

```powershell
D:\github\SokuRL\.venv\python.exe -B RL/run.py train --config-name train_platform_tokens
```

Learning experiment index:PROGRESS_RL.md. Optional local development rules:
AGENTS_RL.md (private, ignored, not distributed).
The token prototype reads the existing public state/history only; no inferred
object velocities or hidden engine state. See ../docs/reports/token_features/.

## Platform Reset

State-only training must explicitly use `runtime.fresh_reset=true`,
`runtime.cards_disabled=true`, `runtime.fixed_weather=clear` and
`runtime.record_replays=false`. The worker enables the Platform's
`SOKURL_FRESH_RESET=1` capability and rejects a Bridge without support.
Existing instances call `BridgeClient.reset_episode(seed)`; this is native
resource-reuse new-game reset, not branch-checkpoint `fast_reset()` and not
the fork's old title/scene-reload path. Character-configuration changes may
still replace the affected instance. RGB and official recording are excluded.

```powershell
D:\github\SokuRL\.venv\python.exe -B RL/run.py validate_platform_integration --workers 4 --frames 160 --output RL/runs/reset-validation.json
```

The validator checks same-seed native/public trajectories, partial new-seed
isolation and unchanged PIDs. Latest evidence and original God AI stage/BGM
reaction conditions: `../docs/reports/token_training/20261013-continuation/`.
God AI is not interchangeable with the independent `community_combo` baseline.
Its one-frame privileged contract is not yet integrated with public-token
three-frame decisions. Muting audio does not disable script BGM conditions.
