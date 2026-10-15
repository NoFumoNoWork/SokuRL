# Human Replay Command Calibration

2026-10-15 01:06 by Codex.

## Actual Result

Two non-spell attack sequences now reproduce in independent live VS with the shared command
interpreter and Platform frame-level input. The fixture runner loads no .rep file
and never copies an original next-frame state. Only known native RNG regions are
aligned once at frame0 before stepping. Both cases use Clear and cards disabled.

1. 20260911_124925: Reimu jA306 -> j8A309; source action start823, contacts832/860.
   Two hits,1159damage. Independent VS runs0->920, ordered contacts and final combat
   projection match. Portable fixture rerun PASS8.469s;0firstCH in this combo.
2. 20260830_124811: Marisa300->320->321->400->411->418, or
   AAA -> 5B -> 6C -> 66C using SokuLib enum names. Start264, contacts270->516;
   13 damaging contacts,3775 total damage across multiple combos, NOT one13-hit
   true combo. The native count is0->6 over270..334 (AAA/B,1747damage), then
   resets6->1 at365 and reaches6 at380 (6C,668damage), then resets6->1 again
   at516 (66C,1360damage/CH). Independent VS runs0->576; contact order, damage-time Rate,
   spirit/reaction/combo fields and final combat projection match.
   Portable fixture rerun PASS6.593s; the1CH is the first hit of the final separate
   exchange, not an erroneous CH on a continuation of the first combo.

Reimu/Marisa input queues and both seats are driven by the same CommandPlayer.
The exact human timing is retained, not shortened into guessed cancel windows.
Case1 has83/117 RLE stages for the two seats; case2 has41/75. These are generated
input runs, not character-specific handlers or reconstructed move frame tables.

## Evidence and Validation

- Existing lossless replay parser reused read-only. Entire original poll streams
  roundtrip through shared Keys/CommandPlayer into byte-identical generated .rep.
  Each stream has an incomplete last input pair; it is retained, not padded.
- Native reader validation on the generated inputs:2500frames per replay,
  every RawFrame field/hash identical to native source playback.
- Neutral-input generated files diverge at62/80F, proving the source input matters.
- Independent live VS with no replay reader: every frame through920/576 compared
  against original observed combat projection. Native direction/A/B/C/D counters
  also matched. Weather forecast separately checked after RNG alignment; no difference.
- Neutral live VS diverged at62/80F. It did not secretly follow the original replay.
- Portable cases were then tested separately without even passing a replay path.
  Ordered contact outcomes and final combat projection/input counters match.
- Platform tests54PASS0.35s; RL compatibility155PASS/2SKIP/2subtests20.44s;
  diff checkPASS. No new native source semantics or build
  required: the existing verified ABI8 Bridge binary was temporarily deployed.
- Game EXE MD5 remains DF35D1FBC7B583317ADABE8CD9F53B2E. Original installed DLL
  restored to SHA256 0CBBA65B36D97C7EE7D54E36353168028A5D0E0F7B9210356B6469625D5F0430.
  All owned game processes shut down. The calibration run itself did not commit/push.

## Why Initial RNG Was Necessary

Same replay seed did not produce the same initial RNG cursor in normal live VS
with its configured deck versus native playback. The624-word MT tables matched,
but their cursor hashes differed. Forecast diverged at183F; random object movement
later diverged at263/331F despite identical actual input counters.

Known addresses are reused from existing tools/validate_fresh_reset.py:
0x89B660,624*4bytes;0x896B20,4bytes. The source frame0 snapshot is SHA-bound to its
replay, restored once to the owned paused live process, and read back byte-for-byte.
2500bytes are written once. No positions,HP,actions,projectile states or future
RNG state are applied. This is explicit initial-state setup, not a general reset
feature or proof of complete hidden-state equivalence.

## Preserved Failures

- Native Replay mode input injection was invalid: original reader continued
  consuming inputs; durations sometimes advanced twice. An all-neutral override
  could still follow the first replay. These results remain in20261015-first.
- Changing only replay submode2->live submode1 did not replace the replay-created
  input managers; injected controls became zero.20261015-live-source records FAIL.
- The old parser's per-round field labelled stage=3 disagrees with actual native
  stage14. Fresh live VS uses verified RawFrame.stageId14, not that legacy label.
  No claim is made about the correct meaning of the parser's mislabeled byte.
- Raw duration counters cannot be submitted as logical StepWithInputs: Bridge
  validation rejects axes outside[-1,1] and nonboolean buttons. One early attempt
  timed out at63F after rejection; later tools explicitly report this route BLOCKED.
- Fresh VS without RNG alignment fails even after excluding card transport fields.
  Excluding forecast alone did not hide random-projectile divergence; the failed
  comparison is retained in20261015-independent-vs-combat.

## Acceptance Boundary

Combat projection excludes stateHash,segmentId,replay-vs-live submode, transported
RawFrame input fields and cardGauge/cardCount/handIds/handCosts/selectedCardIndex.
The remaining player/object fields,actual weather,forecast,RNG seed,physics,HP,
spirit,hitstop,untech,combo and timing fields are compared. Actual native input
counters are checked through the sidecar, not inferred from RawFrame transport.
Contact comparison includes order/frame,source action/sequence,HP/spirit,Rate,
combo/Limit,reaction actions and first-hit CH. Process pointers are not compared.

Case1 removes5card-input samples from the prefix; no spell/system action was
observed there, and the resulting noncard combat fields actually matched.
Case2 removes none. This does not prove card-input removal is safe elsewhere.

Not claimed: whole-match independent live parity; all weather/skills/cards; all
hidden state; a one-line motion macro with default tap timing reproducing exact
human timings; full startup/active/recovery classification. Start these fixtures
atframe0; do not start at823/264 without restoring the complete corresponding state.

## Reproduction

From D:/github/SokuRL-main-platform, with its existing Win32 Bridge build and no
other th123 process, the wrapper backs up/deploys/restores the DLL automatically:

```powershell
& .\tools\validate_replay_commands.ps1 `
  -Cases @('docs/reports/replay_commands/20261015-delivery/portable_cases/replay_1/case.json', `
           'docs/reports/replay_commands/20261015-delivery/portable_cases/replay_2/case.json') `
  -Output docs/reports/replay_commands/<new-run>
```

Fixture sources: portable_cases/replay_1/case.json and replay_2/case.json here.
Fresh independent acceptance reports: ../20261015-portable-acceptance/case_1.json,
case_2.json. Native captures: ../20261015-first/run/. Full5000-frame source/generated
reader parity: ../20261015-compiled-reader/run/. Independent every-frame comparison:
../20261015-independent-vs-initial-rng/run/. All failed experiments remain local.
The fixtures contain programs,initial RNG and acceptance fields; the standalone
runner does not need the original recordings or the original large capture files.

## Published Bundle

The subsequent user-authorized main publication includes this report and manifest,
the two portable cases, portable-acceptance/case_1.json and case_2.json, the compiled
reader summary, and the independent-VS summary and positive command-player reports.
The referenced large raw captures, DLL backups, original REP files and intermediate
failure directories remain local; the failure findings above are preserved here.
No unrelated command_platform reports or obsolete nonportable delivery/cases are
part of the publication. JSON evidence is kept byte-exact by local Git attributes.
The manifest describes the calibration working tree at generation time, including
its then-uncommitted status. Source hashes refer to those working-tree bytes;
source line endings may differ on another Git checkout. Fixture/acceptance hashes
remain byte-exact and independently bind the real test results.

Publication correction: the original candidate collector split only on entry
combo_count_before==0. It missed resets performed inside the resolver (6->1),
therefore the preserved fixture's historical combo/hits13 metadata describes a
contact window, not proof of one true combo. The ordered raw contact evidence and
execution PASS are unchanged. The collector now also splits on count decreases/
resets, with a focused regression. No collision, damage, CH or game rule was changed.
Current native counter segments are indexed in combo_segments.json. Publication
regression after this analysis correction:55Platform tests passed in0.39s.
