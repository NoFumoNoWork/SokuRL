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

## Original CPU Opponent Probe

The original CPU integration can be checked without PPO or a parallel game
environment. Build the compatible Platform Bridge, then run from SokuRL:

```powershell
.\.venv\python.exe -m pip install "lupa==2.6"
.\rl\tools\run_native_cpu_benchmark.ps1 -Output rl/runs/native-cpu-my-test -Games 10
```

This temporarily deploys/restores the Bridge and runs full original Marisa Lua
against native Reimu CPU on the right, with Clear, empty decks and neutral
stage/music0. It uses the declared god-script package in `third_party/th123_ai`,
not a rewritten pressure policy or imported historical training runtime. No
policy is trained. To evaluate a trusted, saved animation/1F checkpoint instead:

```powershell
.\rl\tools\run_native_cpu_benchmark.ps1 -Output rl/runs/native-cpu-checkpoint-test -Games 10 `
  -TrainingDirectory rl/runs/combat-v2-adaptive-20261015/campaign/train-4 `
  -Checkpoint final.zip -Device cuda:0
```

The checkpoint path reuses the saved observation/history/action contract and
seeded categorical inference, rejects incompatible contracts, and caps each game
at its saved horizon. Timeouts are separate outcomes, not HP-based wins.
Results and compressed raw-frame evidence stay under the
new output directory. Games are single-KO episodes, not best-of-three matches.
CPU difficulty is inherited from the original game configuration; this command
does not change it. CPU-left and fast-reset reuse
are unverified. This entry does not enable native CPU in ordinary PPO configs.
Use `-Games 1 -MaxFrames 1200 -Smoke` for a neutral-left autonomy check, not a
god-policy strength evaluation. Saved frame evidence is not a native REP file.

## Record And Watch Checkpoint Battles

Requires the updated Platform Bridge with native recording-boundary protection.
A complete Marisa/Reimu, Clear/cards-disabled checkpoint-vs-pressure case passed
5,642-frame battle-projection parity, including hitstop and projectiles. This is
a bounded validation, not proof for all checkpoints, characters or conditions.
Earlier failed recordings remain rejected; they are not retroactively repaired.

The PowerShell entry temporarily deploys the built Bridge and restores the prior
DLL. It never trains. Use a new output directory; add `-Watch` to play the saved
game immediately, or open its catalog later in another protected session:

```powershell
.\rl\tools\review_checkpoint.ps1 -TrainingDirectory rl/runs/my-training -Output rl/runs/my-match -Watch
.\rl\tools\review_checkpoint.ps1 -Catalog rl/runs/my-match/battles/catalog.json -Output rl/runs/my-match-view
```

Select `-Checkpoint checkpoints/my-model.zip`, `-Opponent pressure`, `-Seat right`
or `both`, and `-Seed` as needed. The checkpoint must remain under its training
directory with the matching `config.yaml`; it need not be the final checkpoint
or come from a completed training run. `-DryRun` validates arguments without
deploying or launching the game. Build the compatible Bridge first.

Use a prepared, compatible Platform Bridge session (the same reversible deployment
requirements as evaluation). This records new evaluation games, not past training
episodes. By default, load the checkpoint on the left and play one game:

```powershell
.\.venv\python.exe -B rl/run.py record_battles --training-directory rl/runs/my-training --checkpoint final.zip --output rl/runs/my-review --seeds 918042743 --opponents pressure
.\.venv\python.exe -B rl/run.py watch_battle --catalog rl/runs/my-review/catalog.json --index 0
```

Both commands support `--dry-run`; recording supports `--seat right` or `both`,
`--num-envs 1..4` and `--device`. The recorder reuses `benchmark_br` and Platform's
original REP export; it performs inference only and never updates the checkpoint.
It explicitly disables fresh reset **only in this dedicated evaluation** because
original recording across resource-reuse resets is not verified. Normal training
configuration, input timing, rewards and model weights remain unchanged.

**Recording-mode input boundary:** startup requests stay neutral until the native
recorder starts, with no fixed warmup frame count. Native player input latching
computes release edges rather than overwriting the previous player key state.
This intentionally differs from legacy training injection, which remains unchanged.
The action NPZ contains requested commands; pre-ready requests are not executed.
Do not treat review-vs-training differences as checkpoint learning improvement.

REP files and hash-bound sidecars live in the run's `replays/` directory.
`catalog.json` associates each file with its seed, checkpoint, matchup, outcome,
reward totals and available combat statistics. Incomplete or ambiguous associations
fail rather than guessing. A saved REP is **not** a playback parity verdict.
The viewer uses native replay inputs, restores recorded Clear/cards-disabled
conditions, renders at normal speed, and stops at the recording horizon; it does
not inject policy inputs. Vanilla playback without these conditions is unverified.

No reward or attention bytes are appended to the original REP format. Per-frame
attention capture and in-game diagnostic overlays are not implemented here;
adding native drawing/Hook code requires separate Platform approval. Attention
weights would be diagnostic weights, not proof of causal importance. Community
frame-advantage counters also need validation before being labelled exact.

## Human Replay BC

Human replay supervision has its own provenance and acceptance files; do not pass
it off as rule-AI demonstrations or invent human rewards/returns. The exporter
uses the existing native replay reader, remaining-input counts, frame-end events
and renderer-state observations. Labels are original logical keys paired with
the preceding observation, not inferred accepted moves or postprocessed keys.
Unknown histories and unavailable/masked actions retain their time rows but do
not receive cross-entropy supervision. Original weather, decks and inputs are
never rewritten to manufacture compatible examples. The current exporter keeps
only compatible first-round prefixes and stops at a round boundary.

`export_human_replays`, `validate_human_export`, and `train_human_bc` are available
through `rl/run.py`; use `--help` for their explicit paths. They require the local
audited split, source checkpoint and existing compatible Platform capture build,
which are external resources, not bundled training data. The convenience
`export_human_replays.ps1` uses reversible Bridge deployment. Training requires
hash-bound prefix acceptance for separate train/validation manifests, loads exact
source weights with a fresh optimizer, and performs actor-only fitting for at
most ten epochs. It does not supervise the Critic or change the 90-action contract.

The bounded `evaluate_human_bc.ps1 -Candidate <training-directory> -Output <new-run>`
compares source and BC-best checkpoints against pressure on eight paired seeds
and both seats. Each two-seat batch starts a fresh Worker because the recording
configuration's post-KO scene reload failed in the earlier multi-batch test.
REP catalogs are stored per seed. NLL/accuracy gains are not evidence of stronger
play; compare real native damage and outcomes, and preserve failed runs.

### Multiscale Human BC Pilot

`train_multiscale_bc --data <accepted-export-run> --source <train-4-directory>
--output <NEW-rl/runs-directory>` runs a bounded A/B/C comparison. All groups use
the same eligible anchors, source Actor weights, optimizer settings and epoch
budget. A keeps the original short-history actor; B adds recent states at
20/15/10/5/0F, compact distant states at60/45/30F and60 consecutive own logical
inputs. C uses the same inputs plus horizon-indexed future0..4 action heads.
D is the renamed weighted-C method: `L0 + lambda/4 * (L1+L2+L3+L4)`,with
`--future-weight 1` by default;each horizon uses its own masked CE mean. An
empty future horizon contributes zero without redistributing its coefficient.
C retains the original pooled five-head mean;D changes loss weighting,not
architecture or runtime control. Historical weighted-C files keep their old tag
and are linked by provenance rather than overwritten or renamed.
Use `--variants D --initial-artifact <typed-best.json>` to continue full BC
actor/context/future weights with a fresh optimizer. `--train-collection` reads
hash-bound references to separately accepted datasets without copying shards.
Bounded larger runs support at most20epochs and explicit `--max-anchor-visits`
up to2000000;defaults remain10epochs/1000000. Weight snapshots are saved every
five epochs,not resumable optimizer moments. `-Groups D` selects D-only native
evaluation. All raw replay capture and label timing remain Platform-backed.
Missing future labels are individually masked; history rows are never spliced.
Observed displacement is not native velocity. No extra Hook or opponent input
is used. Distant summaries omit old resources and full historical object lists.

The typed `best.json` manifest binds `best-actor.zip` and `best-context.pt`.
For B/C, the ZIP alone is NOT the full policy. Load the manifest through
`human_multiscale`; legacy PPO loading remains unchanged. Deployment observes
every frame, maintains separate episode history and executes only current head0,
never five open-loop frames. The context residual is disabled during the first
60 history steps. PPO continuation with this context is not implemented; do not
silently treat the sidecar as an unchanged PPO checkpoint.

Use `evaluate_multiscale_bc.ps1 -TrainingRoot <run> -Output <NEW-run>` for the
paired four-world,two-seat,pressure pilot with native REP recording. It uses
fresh two-slot Workers per seed and reversible existing Bridge deployment.
Raw data, models and reports stay under local `rl/runs/`, outside Git.

### Offline Action And Movement Chains

`extract_human_chains --collection <accepted-training-collection.json>
--validation <validation-manifest.json> --validation-acceptance <acceptance.json>
--output <NEW-rl/runs-directory>` reads existing accepted shards, original state
sidecars and decoded native events. It does not launch TH123 or train a model.
The output retains all input-accepted operations,including no-contact attempts,
source-root-linked contacts,observed Combo sequences,and movement/release runs.
Multiple hits from one root remain one operation. Unknown roots remain unknown.
Native-root joins can include a movement operation rather than an attack.
Behavior chains retain flight,jump,movement/release and intervening operations,
including steps without a hit. The P0 attack-range diagnostic only concerns
damage-source interpretation,not behavior-chain eligibility. Root association
alone cannot certify an independent attack orlegal cancel route.
Absolute input programs reuse Platform `compile_input_frames`;they are evidence
for known contiguous input runs,not invented command names or executable whole
Combo macros. Movement runs are submitted-input proxies,not confirmed jumps.

Each `decisions.npz` has one pre-state per operation/run,not one repeated loss
per held-input frame. Native action IDs and movement targets are a NEW offline
label contract,not the existing90 primitive actions. Same-frame multiple
acceptances are masked. Future outcomes remain in `chains.json`,not features.
Acceptance-time history may already contain the command: these labels alone do
not prove learning initial intention or establish a deployable high-level actor.

`weather_before` is separate metadata from an existing native field;the1544
PPO observation remains unchanged. Non-Typhoon weather eligibility is only a
candidate audit: no previously rejected rows are silently admitted,and cards,
skill/history constraints still apply. A weather-conditioned policy and expanded
replay export need separate validation before non-Clear training is claimed.

### Event-Driven Hierarchical BC

This independent path does not replace PPO,D checkpoints,rewards orthe primitive
90-action transport. `hierarchical_bc.yaml` defines a current-frame spatial
Transformer initialized from the source policy,a3-layer128-wide4-head event
Transformer over16past public events,and the existing60F own-input CNN. State128
plus input64 is projected to128,then concatenated with event128 and fused256->128.
The heads predict a single command,termination hazard,and3future command events
autoregressively through a GRU decoder. Only the first selected command executes.
The loss is actionCE+0.5*terminationBCE+0.5*sequenceCE;elapsed control duration
conditions termination. No supervised Critic orPPO optimizer continuation.

Build data with `build_hierarchical_data --chains <accepted-chain-run>
--motion third_party/th123_ai/source/motion.txt --output <NEW-runs-data>`. It uses
existing raw logical input edges joined to community action names and native
acceptance. Unknown/ambiguous command onsets,skills andcharged variants stay
masked;acceptance time is not substituted for onset. Unknown operations cut
future targets instead of joining two known actions across them. Native damage,
Combo,hidden flags andopponent keys do not enter the public event history.

`train_hierarchical_bc --data <data/manifest.json> --output <NEW-run>
--updates 2 --device cuda:0` performs a bounded correctness update and typed
save/load test. Default updates=0;the current entry caps at16updates and does
not authorize a long run. Candidate masks are vocabulary masks,NOT a complete
engine legal-action table. The new checkpoint JSON/PT cannot load as PPO/D.

`policy/hierarchical_bc.py:HierarchicalEpisode` connects model decisions to
Platform `CommandPlayer` and the existing90-action inverse map. It supports
replanning,interruption,feedback,per-episode memory and a120F watchdog;feedback
opportunities alone do not force switching. Forced states are not policy choices.
The caller must supply an explicit verified engine legal set. Existing Bridge
feedback proves acceptance after a command,not the complete legal set before it.
Consequently current native deployment is BLOCKED at that contract,not silently
run with all commands marked legal. Controller tests use explicit synthetic
legal sets and do not constitute TH123 closed-loop verification. Weather is not
added silently;the current event-data implementation retains existing prefixes.

The strict default remains available. An explicit `masked_native_feedback`
controller mode instead uses a restricted candidate mask and leaves final action
acceptance to the EXE. Such a mask is NOT labelled an engine legal set. Unknown
states and forced reaction/hitstop feedback block new choices;candidate windows
reuse existing V3 control gates. Every attempt separately records submission,
observed native acceptance,and interruption/end. `NOT_OBSERVED` is not proof of
rejection,especially for movement that may not emit matching acceptance feedback.

`./rl/tools/smoke_hierarchical_native.ps1 -Output rl/runs/<NEW-run>` is a bounded
two-instance,left/right native wiring check with reversible existing Bridge
deployment. Its independent `hierarchical_native_smoke.yaml` restricts candidates
to previously tested basic movement/jump/5A/5B. It reuses the existing command
test's600Fprewarmup,checks the known5Bfixture,then runs256Fmodel control and64F
after a new-seed fresh reset. Histories/controllers are separate per slot/reset.
This is not training,a complete battle,a performance comparison orall56command
legality verification. A120Fprewarmupattempt failed and remains evidence;no
native startup threshold ornew game mechanic was inferred from it.

The bounded hierarchical supervised continuation uses a separate explicit profile:

```powershell
.\.venv\python.exe -B rl/run.py train_hierarchical_bc --data rl/runs/hierarchical-bc-implementation-20261017/data-final/manifest.json --training-config rl/config/hierarchical_training.yaml --output rl/runs/my-hierarchical-training --device cuda
```

This profile requires the local hash-bound event dataset and typed hierarchical
initialization checkpoint. It preserves the existing model and masked loss,
shuffles each training partition once per epoch, and caps execution at ten epochs
and 5,000 updates. Full validation composite loss selects an immutable epoch
checkpoint; `best.json` indexes it. Initial, final and every epoch retain model
weights and separate Adam state. Starting from the earlier correctness checkpoint
uses fresh Adam because its optimizer moments were not saved. This is not a D or
PPO resume, and offline improvement alone does not establish stronger play.
Use the native smoke wrapper's `-Checkpoint <typed-json-path>` parameter to check
the selected model through the existing bounded candidate-only control loop.

The optional `hierarchical_balanced_training.yaml` profile changes only event
sampling: known attack and movement/flight choices receive equal draw quotas,
while every non-choice row remains once per cycle. Rare attack rows may repeat;
their full histories and unknown-label masks are unchanged. A cycle is therefore
not a unique pass over all action choices. Validation stays at natural frequencies.
Inspect exact attack accuracy, false attack predictions, stop recall and both
teacher-forced/free-running sequence metrics, not just the composite loss. This
profile changes training priors; it is not evidence that the resulting policy
chooses attacks appropriately or outperforms the uniform baseline.

For expanded-data continuation, `hierarchical_expanded_training.yaml` names the
accepted expanded dataset and prior selected hierarchical model. Its
`resume_training_report` must bind both that exact checkpoint and its Adam
sidecar by SHA-256. A mismatched, missing or modified sidecar fails rather than
falling back to fresh Adam; restored learning rate must match the profile. The
new dataset starts a new seeded sampling schedule, not an exact continuation of
the previous minibatch/RNG position. PPO optimizers and frozen D models are not
used or modified. Data capture/acceptance must finish before this training entry.

With that accepted local dataset and its required prior model/report available:

```powershell
.\.venv\python.exe -B rl/run.py train_hierarchical_bc --data rl/runs/hierarchical-bc-expanded-20261017/data/manifest.json --training-config rl/config/hierarchical_expanded_training.yaml --output rl/runs/my-expanded-continuation --device cuda
```

Use a new output name. The profile limits the run to six sampling cycles and
5,000 new updates, preserves the fixed validation partition and restores only the
bound hierarchical Adam state. These data/models are external local resources,
not files supplied by the RL branch.

## Factorized Event BC (Correctness Profile)

The independent `hierarchical_bc_v2_factorized` format keeps the spatial encoder,
3-layer/128-wide/4-head Event Transformer, last16 events and own60-frame input
CNN. Choice first selects one of six command categories (locomotion, flight/dash,
melee, shot, skill, charged shot), then a member of that category. The unchanged
56-command catalog maps to Platform `compile_commands`/`CommandPlayer`, not native
action IDs. Backward movement is not automatically labelled a successful guard.

Category masks are derived from member masks; conditional probabilities normalize
only over allowed members. Empty categories cannot be selected. Offline masks
remain vocabulary candidates, NOT a recovered complete engine legality oracle.
Current masks do not claim to know future-event legality.

The four overlapping supervision roles are separate from command categories:
movement/neutral, attack/skill, cancel/link, defense/reaction. Initial sampling
weights are30/30/25/15%; exhausted or absent roles redistribute across available
roles. Samples retain all valid task targets even when drawn for one role.
The fixed profile caps a row at2 draws, a REP at256 and an identified attack chain
at32 per schedule. An impossible budget reports BLOCKED_CAPACITY, not an unlimited
replacement fallback. Observed succession can supervise a link, not prove cancel.

The loss is `choice + 0.3*termination + 0.15*future`; each term divides by its own
valid-label count. Choice is category CE plus within-category CE (joint NLL).
Train-only action frequencies provide square-root inverse-frequency weights,
bounded1..4, for current/future CE. Unknown labels contribute no CE. Future3
predictions remain autoregressive auxiliary outputs; live execution submits only
the current command and can interrupt/replan at the existing public opportunity.

Use NEW outputs, with the accepted local v1 dataset and its referenced original
exports available:

```powershell
.\.venv\python.exe -B rl/run.py build_semantic_event_data --data rl/runs/hierarchical-bc-expanded-20261017/data/manifest.json --output rl/runs/my-semantic-data
.\.venv\python.exe -B rl/run.py check_factorized_bc --data rl/runs/my-semantic-data/manifest.json --output rl/runs/my-factorized-check
.\rl\tools\smoke_hierarchical_native.ps1 -Output rl/runs/my-factorized-native -Checkpoint rl/runs/my-factorized-check/checkpoint.json
```

`hierarchical_factorized.yaml` permits only four correctness updates, not formal
training or model selection. The explicit warm start copies v1 shared parameters,
initializes new heads, and uses fresh Adam. Typed persistence rejects incompatible
formats; v1/D/PPO checkpoints are preserved. No automatic PPO policy conversion
or Critic compatibility is claimed. Live execution still uses the original90
primitive interface and1F environment clock, with a separate per-instance memory.

Derivative shards preserve source state/input/targets; past HIT/BLOCK feedback is
added only after its existing public native event is available. WHIFF/CANCEL/
RECOVERY encoding is supported only for independently confirmed timestamped
feedback; current sources do not supply those proofs, so they stay UNKNOWN.
Do not substitute prefix-end no-contact or future combo outcomes. Native acceptance
and raw operation onset remain separate in alignment evidence. Unknown skill and
charge labels stay masked; the new output heads do not invent their supervision.

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
