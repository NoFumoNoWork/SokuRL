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

### Bounded Factorized Training

The separate `hierarchical_factorized_training.yaml` profile authorizes no search
or automatic expansion. It uses fresh Adam3e-5, batch64, ten sampling cycles of
4096draws (40960total,640updates), the fixed1/.3/.15loss and2/256/32repeat caps.
A sampling cycle is NOT a unique pass through all41075source rows. Its initial
model is the typed pre-correctness v2 checkpoint, not the four-update smoke model.
These local data/model resources must exist before running:

```powershell
.\.venv\python.exe -B rl/run.py train_factorized_bc --config rl/config/hierarchical_factorized_training.yaml --output rl/runs/my-factorized-training --device cuda
```

All checkpoints and Adam sidecars remain immutable. Best selection includes the
initial model and uses full natural validation with train-only class weights and
the1/.3/.15objective. Accuracy/histograms use actual category-first decisions,
not the possibly different flat joint-probability argmax. Free-running future
metrics likewise use the hierarchical decoder. Record category/attack/movement,
stop and autoregressive metrics, not overall accuracy alone. No sealed test or
win-rate claim is implied; absent skills/charge and scarce defense labels remain.

### Post-Opening Projectile-Context Selection

Build an independent derivative from accepted semantic v2 shards:

```powershell
.\.venv\python.exe -B rl/run.py build_post_opening_events --data rl/runs/hierarchical-factorized-20261017/data-v2/manifest.json --output rl/runs/my-post-opening-data --min-frame 300
.\.venv\python.exe -B rl/run.py check_factorized_bc --data rl/runs/my-post-opening-data/manifest.json --output rl/runs/my-post-opening-check
```

The v3 dataset preserves every original row, target and history array. Only
training eligibility changes: original pre-action observation frames below300
cannot be sampled or affect class weights. All reliable attack labels after the
cutoff remain eligible, whether they hit, are blocked or have no observed contact.
Continuous direction/D/neutral-release labels may train with current visible
object context even without an own attack. Existing observed links, defense
context and attack stop/continue supervision are also retained. Empty-screen
unlinked movement is excluded. Selection does not create new labels or turn
passive events into autonomous choices.

Object context uses the existing16 public visible-object tokens, both owners;
these do not prove attackability, danger or graze properties. Neutral/release
counts are reported separately from direction/D. Opening states still supply
causal history to later samples; raw REP files are never cut or rewritten.
The factorized loaders accept explicitly typed v2/v3 datasets. Full natural
validation remains unchanged, including opening rows; report this separately
from the filtered training distribution. This command performs at most four
correctness updates, not a new formal training run or evidence of stronger play.

`config/hierarchical_factorized_post_opening.yaml` is a prepared training profile
that changes only the data path relative to the prior bounded profile. It does
not authorize starting another training run; the old profile/data remain intact.

To expand a fully used training pool, `plan_human_expansion` accepts `--audit`,
`--index`, `--output` and `--max-new` (1..200). It appends only new local
Reimu/Marisa REP after byte/hash/character-seat checks. All old validation/test/
excluded assignments remain exact, and heldout player/SET/hash overlap is rejected.
This static source audit is NOT native alignment acceptance. The existing
`export_human_replays.ps1` accepts `-Audit` and `-Plan` for the new audit and subset;
its default audit and native capture semantics are unchanged. New exports must
pass `validate_human_export` before joining a hash-bound HumanCollection and the
existing chain -> hierarchical -> semantic -> post-opening derivative pipeline.
Keep fixed validation and sealed test untouched; do not turn a source audit into
fabricated capture acceptance or claim full-match replay parity from prefixes.

## PPO Example And Continuation

### Event BC Vs Direct PPO

`tools/run_event_duel.ps1 -Output rl/runs/my-duel -Checkpoint PATH_TO_TYPED_BC_JSON`
compares the frozen event BC with the existing `train-4/final.zip` PPO (override
`-PpoDirectory` explicitly). Both retain their trained Marisa character in a
mirror matchup; four fixed world seeds, both seats, eight games, Clear/no cards,
7200F horizon and two independent native instances. It reuses the existing
event controller, seeded stochastic PPO episode, primitive interface and guarded
Bridge deployment. Model weights are never trained. Full compiled command
candidates are not claimed to be exact engine legality. KO wins and timeout HP
leads are reported separately, with damage/contact summaries and submitted
primitive inputs; these NPZ inputs are not native REP files. Mirrors are outside
the original training matchup, BC inherited PPO feature weights, and this small
screen is not a compute-matched training-method comparison or significance test.

Factorized event BC continuation is separate from PPO: the prepared
`config/hierarchical_factorized_continuation.yaml` binds the prior selected
checkpoint and completed training report. It verifies the checkpoint/Adam SHA,
same data, loss/sampler configuration, class weights and optimizer layout. Adam
moments/steps are restored, not recreated. Sampler RNG is restored from the bound
checkpoint report; older reports reproduce and verify saved draws to recover its
position, failing on mismatch. Fresh-Adam defaults remain unchanged. Use a new
output directory and an explicitly authorized additional budget; local update
counts and cumulative updates are distinct. Old checkpoints are never overwritten.

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
## Restricted God-AI Kickstarting

This is an opt-in auxiliary objective on the existing 1F, 90-action PPO, not a
new policy network. Student samples alone control the environment. Expert CE is
averaged over valid labels and added to the same PPO optimizer step; reward,
GAE and value loss remain unchanged. `lambda_initial` decays linearly to zero
over `decay_transitions` collected transitions (sampled at each rollout start),
with its clock saved in models. The joint-loss adapter is pinned to SB3 2.9.x.
Old `online_teacher` is a separate post-PPO optimizer and is not this method.

Under the original screen-only contract Marisa/Reimu scripts are **BLOCKED for
training**. Their `x` is a world coordinate; original student animation positions
are screen coordinates. Missing coordinates, boxes, timers, enemy keys
and skill/card state are never filled with guessed values. No new native Hook or
state extraction is included. The entire teacher view is delayed (default 6F),
neutral home/BGM context is explicit, RNG/coroutines are episode-local, and Lua
has no file/process access. A continuation whose input prefix was not executed
by the student is masked, not replaced by the student's own previous action.

`VALID_INPUT` proves vocabulary/card-mask compatibility only, **not** that the
engine accepted a move or opened a cancel window. Advice is never submitted, so
teacher move acceptance cannot be measured by pretending advice was executed.
Unknowns remain masked; positive-weight rollouts with zero labels stop before
any optimizer update. Full legal-decision/teacher adaptation and native rollout
validation are still required before formal training.

From the prepared SokuRL root (install the existing `god` optional dependency):

```powershell
.\.venv\python.exe -B rl/run.py audit_god_teacher --output rl/runs/NEW-expert-audit --check-student
```

The audit uses existing external Lua sources and an explicitly **synthetic**
student observation compatibility probe, not gameplay coverage. It records
source hashes/lines and missing dependencies. `train_combat_kickstarting.yaml`
is a bounded correctness profile, not a ready teacher-training recipe. Retain
original PPO/BC checkpoints; use `kind: weights` to start a new objective with a
fresh optimizer, or identical kickstarting contracts for checkpoint continuation.

### Shared World Coordinates

`episode.animation_world_coordinates: true` is an explicit opt-in extension.
It appends world x/y for the same visible fighters and selected object slots to
each animation frame, without changing existing screen tokens or exposing hidden
objects. Positions come from the already captured Platform `RenderSnapshot`;
there are no new native reads or Hooks. With 16 slots the frame width is 221
(old 185), and eight frames plus eight submitted-input rows occupy 1,832 floats
(old 1,544). Raw world coordinates remain float32; the existing feature extractor
normalizes its new coordinate inputs internally by 1280.

Old models cannot silently load this observation shape. Create an independent
artifact with old parameters preserved and new feature columns zeroed:

```powershell
.\.venv\python.exe -B rl/run.py prepare_world_student `
  --checkpoint rl/runs/combat-v2-adaptive-20261015/campaign/train-4/final.zip `
  --config rl/runs/combat-v2-adaptive-20261015/campaign/train-4/config.yaml `
  --output rl/runs/NEW-world-student
.\.venv\python.exe -B rl/run.py audit_god_teacher --world-coordinates `
  --student-directory rl/runs/NEW-world-student --check-student `
  --output rl/runs/NEW-world-audit
```

The migration verifies old parameter columns, zero new columns, CPU action/value
outputs and reload; it creates fresh Adam state, not optimizer continuation.
Output comparisons use rtol=1e-6/atol=1e-7; do not generalize fixture equality to
native trajectory equivalence. `train_combat_kickstarting_world.yaml` requires
`SOKURL_WORLD_STUDENT` explicitly; it does not silently substitute an old model.

In legacy input mode the original scripts pass world-coordinate reads but block
on native vertical velocity (`my_yspeed`). Position finite differences are not
silently substituted for that field. Spirit/maxSpirit, full object API, cards,
skills, collision flags/boxes and special timers also need explicit treatment.
Many already exist in Platform, but that does not authorize exposing them to the
student. Joint PPO supervision is offline-tested; original-teacher coverage,
legal-decision proof and native rollout validation are still outstanding. Audit
JSON separates these gaps from genuinely missing adapters and tools.
### Cancellable Intent Queries

The kickstarting profiles now request `teacher.mode: intent`. They deliberately
stop at PPO attachment until the decision-boundary and primitive-label adapters
exist; do not start a native rollout expecting working expert supervision yet.

`QueryTeacher.spawn(seed).query(observation, opportunity=..., boundary_id=...)`
supports a reviewed Lua adapter exposing `teacher_decide()`. The adapter may call
`command('236B')` or `propose_intent('236B')`. This captures a proposal, not keys:
the decision coroutine is immediately discarded and never resumed. Lua globals,
closure upvalues and actor-local RNG persist for that episode. The source file is
loaded once, not every frame. Generic reviewed adapters retain pre-proposal
memory writes; the pinned original-source adapter below is stricter and rolls
back decision writes. Outcome-dependent code after a proposal never executes.

`advice.intent` and `advice.intent_id` identify a proposal. Reject it with
`actor.intents.reject(advice.intent_id)`; the next trusted boundary can propose
the same command afresh, without losing tactical memory or disabling the episode.
Boundaries use the existing `Opportunity` catalog with
`PROVIDED_ENGINE_LEGAL_SET`, `can_respond`, a non-forced state and a boundary ID
equal to the current query-frame index (starting at zero). A candidate mask,
repeated/stale boundary or matching button is not legal-opportunity proof.
Between boundaries the query returns a masked wait, not a macro continuation.

Proposals have no 90-action target, no game-control side effects and no native
acceptance claim. Unsupported commands and masked intents are not neutral labels.
Raw key/wait/background-thread operations mask the current decision segment,
without blocking the episode. Source/dependency failures remain explicit BLOCKED.
An unadapted main-only script still reports `BLOCKED_INTENT_ADAPTER`.

`teacher.source_adapter: th123ai_20110925` now selects a byte-pinned adapter for
the external 2011 `01_marisa_main.ai` / `00_reimu_main.ai`, `def_clc.lua` and
`function_command.txt`. It keeps original source line IDs and conditions, splits
the reviewed outer loops into initialize/observe/decide, and rejects other source
versions. Source assets are not edited or copied into the formal RL package.
Initialization runs once; the original calculation body runs once per delayed
observation, even without a decision boundary. This perception-before-decision
order is an intentional adaptation, NOT original scheduler equivalence.

In the legacy source experiment, decision globals/table writes are rolled back after every query, keeping
the committed perception memory and table aliases. This prevents unexecuted
plans from committing charge/combo flags. It also drops some legitimate original
strategic writes: those require individual review before retention. The bounded
transaction is not a general closure-upvalue rollback mechanism. Source analysis
found no `local` declarations in these pinned main/perception files.

The reviewed `lvr236`, `lvr214`, `lvr623` and `lvr22` helpers emit intents instead
of input coroutines. Both orientation branches are preserved: for example the
original lvr236 uses opposite-facing input when `is_dir_front` is false, so that
branch maps to 214, not blindly to 236. A mirrored 623 becomes 421, which remains
unrepresentable in the existing command catalog. Held variants, raw-key movement
and other unaudited operations are masked; no motion timing equivalence is claimed.

With the shared-world contract both original source probes currently block in
`def_clc.lua:670` on `enemy_obj_n`, before any decision. Full object counts/data,
box geometry and other native fields are not silently synthesized from capped
visible tokens. Thus the original entry split is implemented but **not yet a
usable original-AI teacher**. Run the existing synthetic audit for precise source
hashes, split boundaries, contract blockers and optional model initialization:

```powershell
.\.venv\python.exe -B rl/run.py audit_god_teacher --world-coordinates `
  --student-directory rl/runs/teacher-world-20261018/student --check-student `
  --output rl/runs/NEW-source-intent-audit
```

The directory argument must identify an existing explicitly migrated artifact.
Weight checks do not attach the teacher or bypass the formal PPO readiness guard.

Legacy `mode: input` (also the old omitted-mode contract) remains for existing
checkpoints/offline tests only. Its unexecuted-prefix guard still masks until
reset; it is not the new intent protocol. No formal teacher-training readiness or
native decision-boundary validation is claimed by the intent unit tests.

### Restricted Teacher Phase 1

The perception-only staged profile is `train_combat_kickstarting_visible.yaml`, based on
the unchanged 1,832-float world observation, 1F decisions and 90 primitive actions.
It is **perception-only**, not a teacher-training recipe. It returns
`PHASE1_PERCEPTION_READY` without a target action; legacy complete-list calculation,
main action execution and global decision rollback are not invoked. Phase 2 must
classify/synchronize memory and replace the legacy rollback experiment, followed
by atomic-label wiring and native side-channel rollout validation.

`policy/god/perception.py` maps existing fighter frameCount, animation subphase,
pose frame, quantized spirit fractions and screen coordinates. Frame count is not
pose ID, and spirit fraction cannot recover current/max absolute spirit. Query
and delayed-observation indices are explicit caller clocks, not native battle_time.
Clear/cards-disabled context binds declared runtime config only; native enforcement
is not proven by a config file. Default-skill/card APIs remain unavailable rather
than filled with assumed levels or fabricated hand contents.

`get_visible_obj_data(owner,index)` exposes a read-only selected-object record:
world/screen position, facing, character, action, sequence, animation counters,
relative owner and slot. Slot is not a stable cross-frame ID. HP, native address,
sprite image ID, exact frameCount, native speed and collision geometry are absent;
requesting them fails explicitly. The native `get_obj_data` and `enemy_obj_n` are
not aliased to this interface. Selected own/enemy counts have distinct names.
Below 16 occupied slots the visible selection is complete; at capacity truncation
is UNKNOWN, not proven. Lua exposes boolean proof/known flags plus a status enum,
so an "UNKNOWN" string cannot accidentally pass a truthiness test. Complete native
list coverage is always unknown because hidden entities are filtered.

The optional branch auditor uses an isolated [Lua AST parser](https://github.com/boolangery/py-lua-parser),
not the training interpreter: its ANTLR 4.13 dependency conflicts with Hydra's
ANTLR 4.9. Prepare only the isolated target, keeping all caches/temp inside SokuRL:

```powershell
New-Item -ItemType Directory rl/.tools/lua-parser/tmp -Force
$env:TEMP = "$PWD/rl/.tools/lua-parser/tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$PWD/rl/.tools/lua-parser/cache"
.\.venv\python.exe -B -m pip install --target rl/.tools/lua-parser/site `
  --only-binary=:all: -r rl/tools/requirements_teacher_ast.txt
.\.venv\python.exe -B rl/run.py audit_god_teacher --world-coordinates `
  --visible-perception --branch-parser-site rl/.tools/lua-parser/site `
  --output rl/runs/NEW-teacher-phase1-audit
```

Use a fresh target/output, not an in-place package upgrade. The auditor launches
a separate `-I -S` process and does not import the parser into PPO. Registry rows
bind source hashes, CP932-to-original-byte ranges, transitive helper dependencies,
lexical bindings, earlier/enclosing conditions and loop dependencies. Comments
are not references. `DEPENDENCY_READY_ONLY` is not branch execution support:
guard probes evaluate only pure conditions; branch bodies and labels stay disabled.
Unknown earlier conditions are not treated as false to reach a later `else`.
`branches.json`, `guard_probes.json` and `report.json` provide the lists and reasons;
they are local audit evidence, not implicit runtime dependencies or Git assets.

### Restricted Teacher Passive Memory

`train_combat_kickstarting_memory.yaml` adds `observed_source_v1` memory to the
same 1,832-float contract. It is still **not trainable teacher guidance**. The
profile returns `PHASE2_OBSERVED_MEMORY_READY` without an action label; the PPO
intent-attachment guard remains BLOCKED. Original main, complete-list calculation,
and global decision rollback are not run.

The adapter executes byte-pinned, original `def_clc.lua` slices for action/frame
history, same-action frame rewinds, enemy ground/air counters, the AI's
`after_guard_count` heuristic and facing-relative wall-distance heuristics. Each
has source lines, source/code hashes and independent runtime validity. The latter
two are **AI heuristics**, not recovered native stun timers or verified stage
geometry. The observer ticks once per consecutive delayed student observation;
this is not a native battle_time alias or proof of original scheduler equivalence.

Pre-snapshot history, unknown counter durations and missing dependencies are not
initialized to zero or the original script's defaults. Read-only sparse history
tables fail on unknown slots rather than letting nil select an arbitrary branch.
Observation gaps invalidate continuity; groups recover when actual observations
establish their dependencies. The episode VM is preserved. Rejecting a proposal
does not advance an unexecuted coroutine or rewrite observed history. A source
error or instruction-limit failure stops the memory adapter until Reset rather
than silently continuing a partially executed update.

Tactical writes, B/C hold progress, object identity/prediction, native timers and
other privileged-dependent memory are not enabled. Their source write sites are
listed by role and trust, with unreviewed variables explicitly UNKNOWN. Merely
classifying a variable does not make its initialization or every writer trusted.
Local synthetic lifecycle tests do not demonstrate native teacher coverage.

```powershell
.\.venv\python.exe -B rl/run.py audit_god_teacher --world-coordinates `
  --visible-perception --observed-memory `
  --branch-parser-site rl/.tools/lua-parser/site `
  --output rl/runs/NEW-teacher-memory-audit
```

The additional `memory_inventory.json` records source-specific assignment sites,
direct dependencies and reviewed role/trust status. Reports remain local under
the run directory. No teacher actions are submitted, no weights are updated, and
this command does not replace Phase 3 atomic-label or Phase 4 native validation.

### Shared Current-State Contract (Offline V1)

The explicitly authorized `shared_current_native_v1` expands BOTH learner and
teacher information. It preserves the1832public prefix and appends one current
lossless native context, using the existing decoder and encoding. Both consume
the same tensor: exact resources, flags, skills, boxes and complete bounded object
lists are included. Enemy private key counters and future deck order are redacted;
their teacher APIs reject access rather than reporting padding as observed data.
Own already-consumed key counters remain available. No extra reaction delay is
imposed; existing Lua waits/delays remain part of its strategy implementation.

`SharedNativeFeatures` retains the spatial/temporal Transformer and adds the
existing address-invariant native feature encoder with zero initial output.
Explicit weight transfer preserves Actor/Critic parameters and static card mask;
the optimizer is fresh. Old models/configurations are not silently reinterpreted.

```powershell
.\.venv\python.exe -B rl/run.py migrate_shared_teacher `
  --checkpoint rl/runs/teacher-world-20261018/student/model.zip `
  --output rl/runs/NEW-shared-input-student
```

This is OFFLINE weight expansion, **not** a resumable training configuration.
The output records `training_ready=false`. No capture or environment stepping
occurs. The complete existing layout is592658float32 values (2370632bytes) per
observation; dense rollouts are not an acceptable final storage plan. Reuse the
existing sparse-storage/transport facilities before scaling, without truncating
objects or hiding overflow.

`SharedTeacherPerception` reads only the common tensor. A controlled diagnostic
`SharedNativeSourceActor` reuses the original GodActor scheduler and original
Reimu/Marisa sources. It neither sends game inputs nor produces PPO labels.
If the previous proposal was not executed it stops the stale coroutine. This
demonstrates the control-alignment boundary; it is NOT a completed local recovery
teacher and must not replace the ongoing intent/replanning design.

`policy/god/continuous.py` now supplies a separate, **offline-validated**
`ContinuousSourceTeacher`. It runs the pinned original `calculation` body once
per supplied consecutive observation, independently of whether the student follows
advice. Original `main` decision writes live in a disposable plan scope; observed
history tables are read-only to decisions. Rejection discards only the affected
coroutine and input queue, never the episode VM, observation history or actor RNG.
States are `READY`, `ACTIVE`, `INVALIDATED`, `WAITING`. Fresh plans require a
caller-proven legal boundary, an explicit 90-action mask, and actual preceding
input evidence. The existing `LearningInterface` supplies all action conversion;
there is no second 90-action vocabulary or game-input writer.

`ExecutionReceipt` distinguishes input consumption from separately attributed
native acceptance, including the accepted plan identity. Matching keys alone do
not prove an attack started. Missing receipts, poll gaps, facing changes,
interrupts, hitstop and unknown dependencies mask advice rather than inventing
progress. A caller-proven cancel/replan boundary may replace an active wait.
Source helper/call-site provenance and per-frame reconciliation are retained.
Consumed inputs now explicitly distinguish press, hold and release. A held attack
key does not require a fresh acceptance event on every frame: original wait/hold
helpers advance only through confirmed consumed input frames, while action
acceptance remains `NOT_OBSERVED` unless independently reported. This also permits
release inputs without pretending that a charge completed in the native engine.
`ExecutionReceipt.accepted_buttons` identifies the independently attributed attack
button; matching plan id alone cannot commit a tactical variable. Contradictory
button evidence is rejected.
The initial action must be supplied explicitly. Reset creates an isolated actor.

This is a conservative source adaptation, **not original policy equivalence**:
observation-before-decision changes scheduling. `tactical_state.py` binds the five
reviewed Reimu `ex_combo` write sites to the corresponding next or preceding
accepted attack. Unconfirmed writes are discarded on rejection or unaccepted
release; confirmed memory survives later plan rejection. These are Lua variable
write sites, not a per-game-action handler table. `crush_combo` has only the source
observation reset, and the main `flag_kishitu=2` card-use write is not enabled in
the cards-disabled task. Calculation-owned weather memory remains independent.
Unreviewed writes stay plan-local. No native charge timing or full policy
equivalence is asserted. A partially failed calculation requires Reset because its memory
cannot be trusted. Original-source synthetic recovery tests and fixture motion
tests do not establish native coverage. No production PPO attachment is enabled.
`audit_continuous_teacher` reuses a source-hash-bound existing Lua AST audit:

```powershell
.\.venv\python.exe -B rl/run.py audit_continuous_teacher `
  --audit rl/runs/EXISTING-AUDIT/branches.json `
  --package third_party/th123_ai/package/th123ai/script `
  --output rl/runs/NEW-continuous-teacher/lua-state-audit.json
```

The default Worker still uses its existing observations. The separately approved
opt-in `episode.shared_teacher_context: true` now pairs the existing animation
capture and `SharedCurrentReader` at the same paused frame. It requires world
coordinates,16public object slots,1F/zero latency and the unchanged1832 learning
prefix. Only the current sanitized native context is appended,not eight copies.
The shared reader never reads future deck order;its zero padding is redaction,
not an observed20-card deck. The normal complete Reader is unchanged.

`policy/god/native_feedback.py` requires matching frame-end keys AND an observed
native input-handling scope. It retains native scope/order IDs separately from
its verified receipt sequence;neither frame IDs nor event order are represented
as EXE/REP poll counters. Action acceptance reuses Platform CommandPlayer rules,
rejects ambiguous attack-button evidence and records actual attack roots.
Parser reentry authorizes re-evaluating original main,NOT every high-level move.
Native feedback and the interpreter never write teacher inputs to the game.

A bounded two-instance smoke is available after preparing the explicit migrated
checkpoint shown above and the existing x86 Bridge build:

```powershell
.\rl\tools\smoke_teacher_native.ps1 -Output rl/runs/NEW-native-teacher
# Optional bounded coverage profile;default remains Marisa128+64F/instance.
.\rl\tools\smoke_teacher_native.ps1 -Output rl/runs/NEW-reimu-teacher `
  -Character reimu -Frames 1024 -ResetFrames 64
```

It runs Marisa PPO on both seats against the unchanged Reimu pressure opponent,
128then64frames/instance with same-process Reset,without training. It preserves
game/DLL identity and model parameters. Native control verification observed
384queries,280candidate labels and269recoveries;no known stale prefix labels in
the recorded window. Separate student-action diagnostics found5native acceptance
receipts,NOT5adopted teacher attacks. Most suggestions were neutral/movement.
The extended1024+64F profiles were also checked separately for both original
teachers and both seats,4352queries total. Marisa1913/2176candidate labels and
Reimu1924/2176;no observed stale rejected prefixes. Original lvr66 was reached
by both teachers;Reimu lvr623 was reached3times,but the student rejected its
prefix each time. These are proposed first inputs,NOT3accepted skills. Complete
skill/charge native acceptance remains unverified. Separate student acceptance
diagnostics23/33 are not teacher adoption(0/0). The same fixed checkpoint was
used for wiring tests in both roles,not a Reimu policy-strength evaluation.
Per-actor readiness markers are intentionally not promoted by this bounded
smoke. Do not enable formal expert training on this evidence alone.

### Scoped Original Decisions

`train_combat_kickstarting_scoped.yaml` adds the opt-in `source_scoped_v1`
decision environment. It keeps the episode Lua VM, definitions, RNG, static AI
parameters and observation-confirmed flags, but separates each query's scratch
variables and speculative plan writes. No global snapshot/rollback is used.
An intent rejection discards the plan, not observed history. A later supplied
trusted boundary reevaluates the actual delayed student state. Missing fields,
unreviewed writes and unsupported key/wait operations mask the current segment;
unexpected source errors and resource-limit failures require Reset.

The four original scripts and original API are SHA-pinned. Only complete helper
definition regions are loaded from `def_clc.lua`; its unused object/history
initializers and full calculation thread are not executed. The original
`base_states()` and static defaults supply AI parameters, not verified native
move timing. `rapid` and library tables are read-only during decisions. Unused
source hold flags are constant zero in this exact package, **not** a statement
about student-held buttons. Reimu's observed flag updates at source53..63 are
separate from execution-dependent Combo plans; unexecuted plans never establish
those flags for the next query.

Three reviewed pure guard reorderings check necessary action/height conditions
before missing velocity or read-only frame flags. They preserve Boolean results
for complete numeric inputs; no flag or velocity value is substituted. When an
explicit Clear/cards-disabled task is bound, weather21/display21/counter0 are
declared task premises derived from existing Platform enforcement, **not new
native reads or proof of deployment correctness**. A generated, SHA-bound
`card_only_gates.json` identifies card-only conditions by conservative transitive
active-input effects in the original AST. It is a formal runtime registry, not
a run report; runtime does not import the AST parser. Mixed attacks, dynamic
keys, unknown/recursive calls and active non-card guard inputs cannot prove
card-only behavior and are not pruned. Current source has84 Marisa/72 Reimu
eligible conditions. This is deliberate restricted-policy pruning, not original
policy equivalence. Hand/skill/object APIs stay unavailable, rather than being
filled with false values. Unsupported `else` branches are not rewritten.

`pure_guard_order.json` additionally binds seven original Boolean conditions to
their exact source byte spans. The isolated `teacher_guard_order.py` generator
proves a narrow scalar type contract and moves known necessary conjuncts ahead
of unavailable numeric reads only inside pure runs. RNG, unknown calls, table
indexing and unproved expressions are barriers. It does not supply a skill,
velocity or gravity value, and a potentially relevant unknown still masks the
query. Complete-input Boolean/RNG parity, effect barriers, local shadowing,
source binding and later source-line preservation have offline tests. This is
not native behavior validation or general Lua expression optimization.

Do not connect the current Bridge `skillLevels` to `get_skill_lv`: the existing
community reader uses `+0x6C4` (`skillMap`), while Bridge captures `skillLevels`
at `+0x6A4`. Both are absent from the unchanged student tensor. Disabling cards
alone does not establish every current skill level or reset loadout. The audit
records this distinction and the still-unavailable dependencies with source
hashes; no native reader or observation change is introduced.

The restricted special-state API can return the original `-1` only when the
public character ID and existing Platform applicability metadata prove that a
character-specific field does not apply. For example, the original Tenshi-only
special10 API does not read a timer for Reimu/Marisa. Applicable, common and
unreviewed fields still mask the query; this is not a zero-filled special-state
array and does not expose privileged native values or enemy inputs.

```powershell
.\.venv\python.exe -B rl/run.py audit_god_teacher --world-coordinates `
  --visible-perception --observed-memory --scoped-decisions `
  --branch-parser-site rl/.tools/lua-parser/site `
  --output rl/runs/NEW-scoped-teacher-audit
```

This audit supplies an explicitly synthetic opportunity mask to exercise the
original decision entry without a game. It is **not native legality evidence**.
The report distinguishes state infrastructure, actual source proposals, missing
dependencies, source call stacks and unimplemented primitive/native/training
gates. A local mask is not a teacher label. Generated hashes, source regions,
task gates and original source identities are recorded separately. The original
PPO attachment remains BLOCKED: no formal training before actual-prefix
90-action supervision and native continuous-query validation are complete.

`test_formal_card_gate_registry_reproduces_from_pinned_original_source` regenerates
the classification and checks exact source/API/generator hashes and all entries.
The AST tool's optional `--gate-registry` writes only a new JSON in the teacher
source module and refuses overwrite; preserve an existing registry before an
explicit regeneration. Ordinary audits output the same candidate list under
`card_only_task_gates` without modifying formal source. Do not edit condition
offsets or hashes to bypass a mismatch. Snapshot/compiled/runtime data are not
inputs to this registry, and original scripts are never modified on disk.
Only the generator's text identity normalizes CRLF to LF for Git-checkout
portability; original Lua/API bytes and condition hashes remain strictly exact.
