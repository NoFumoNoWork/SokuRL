# SokuRL Learning Module

Policy observations/history, rewards, opponents, training and evaluation over
the original-engine SokuRL Platform. RL uses sokurl_platform; it does not contain
another game engine, private memory reader or copy of the native scheduler.

## Install Beside Platform

Obtain the complete rl/ folder from the RL branch and place it into an already
prepared SokuRL workspace:

```text
SokuRL/
  platform/
  rl/
  .venv/          prepared Python3.11 x64 environment
  th123_jp/      external TH1231.10a installation and loader/modules
  third_party/   external native dependencies
```

Copying rl/ alone is not sufficient on a new machine. Prepare Platform's Python
dependencies, compatible native Bridge/game resources and MSVC tools first; see
the Platform README at the workspace root. No models, game files or run data are
included in the RL source branch.

With the existing workspace interpreter (a standard venv may instead use
.venv/Scripts/python.exe):

```powershell
.\.venv\python.exe -m pip install -e "./platform"
.\.venv\python.exe -m pip install -e "./rl[ppo,dev]"
.\.venv\python.exe -B rl/run.py train --help
```

The launcher selects rl/src, rl/tools, platform/src and platform/tools explicitly
and passes their paths to workers. SOKURL_PLATFORM_ROOT can select a prepared
Platform source directory; SOKURL_GAME_DIR selects game resources. Formal new
runs do not require the historical Integration worktree or its saved-model alias.

## Configure And Validate

Configurations are in rl/config; outputs/checkpoints/telemetry are in ignored
rl/runs/<unique-run-id>. Choose an explicit experiment profile: the generic
train configuration is not interchangeable with the Combat V2 PPO profile.

```powershell
.\.venv\python.exe -B -m pytest platform/tests rl/tests -q -p no:cacheprovider
.\rl\tools\run_combat_v2.ps1 -Output rl/runs/my-check -PlatformIntegrationTest
```

The wrapper requires the previously built platform/build/bridge/Release DLL and
a prepared runtime. It refuses existing game processes/output directories,
temporarily deploys that DLL, performs native preflight and restores the original
DLL with a hash check. Without -Timesteps it does not train. Do not interrupt a
deployment and assume restoration succeeded; inspect the recorded deployment state.

## PPO Example And Continuation

Run only within the authorized training/resource budget, using new output names:

```powershell
.\rl\tools\run_combat_v2.ps1 -Output rl/runs/my-ppo -Timesteps 2048 -ConfigName train_combat_v2_credit
.\rl\tools\run_combat_v2.ps1 -Output rl/runs/my-ppo-next -Timesteps 1024 -ConfigName train_combat_v2_credit -ContinueFrom rl/runs/my-ppo/training
```

These are bounded examples, not a recommended convergence budget. Profiles specify
opponents, decision timing, masks, observation/reward and PPO parameters; changing
a profile may break a checkpoint contract. Do not silently change action masks,
timing vocabulary or observation schema on continuation. Preserve historical best
models; the final checkpoint is not automatically the best policy.

Combat V2 uses public-token Transformer features with native animation proxies,
not pixel-only perception. Its configuration distinguishes policy inputs from
privileged diagnostics and reward events. Card actions are masked in this profile;
native damage/Combo/CH facts come from Platform, while RL owns reward accounting.

## Reproducibility And Limits

Record both RL and Platform commit SHAs, their dirty states, actual config/seed,
opponents, roles, observation/action/reward contract, Bridge ABI and external data.
A root HEAD alone does not identify both modules when they coexist locally.

Offline tests and short native/PPO/reset/save/reload tests establish engineering
capability, not stronger play. Compare policies using paired seeds, both seats and
appropriate independent evaluation. Current historical tests have not established
a general heldout improvement. Unknown technical segment attribution remains
conservative and must not erase actual native damage or Combo.

Fresh-reset RGB parity and official replay recording across fresh reset remain
outside verified scope. Old checkpoints may still contain absolute historical
paths; handle those explicitly rather than making a legacy junction a silent
dependency. Private Agent instructions, progress/history, reports, models, traces
and run outputs are local-only, not part of the RL Git payload.
