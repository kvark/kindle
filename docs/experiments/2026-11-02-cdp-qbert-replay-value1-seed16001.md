# Qbert replay-value scale1: fresh-seed confirmation

Completed:200k actions/799,817 frames/18,715 updates in24m45.754s. Greedy
14,913.542 andsampled14,745.833 eachclear24/24 first pyramids butmiss15,000.
The fixed recipe passes greedy onone of two learners. All912 pins andoriginal
services pass;terminal reader completes02:50:48.666 UTC November2. No stage
repeats;CPU preparation failures/repairs remain. No extension,retry,third learner,
tuning,seed substitution orautomatic successor.
[Completed result](../results/2026-11-02-cdp-qbert-replay-value1-seed16001.md).

Selected November 2 after the [development comparison](../results/2026-11-02-cdp-qbert-replay-value1.md)
passed the greedy gate on reused seed14009: 16,525 mean return and 24/24 first
pyramids in 25m36.935s. Sampled return was 14,882.292, below the 15,000 gate.
The recipe was selected using that learner; seed16001 is the independent
confirmation. The five-game goal remains unfinished.

## Fixed question and decision

Does the unchanged replay-value1 recipe pass the original mean15,000 and
90% first-pyramid gate on a fresh learner and evaluation cohort? Greedy is
primary. Report sampled results separately, without selecting an evaluation
mode, checkpoint or episode subset after seeing results.

Only learner and evaluation seeds change: 14009→16001 and
4000014009→4000016001. A pass establishes a development pass plus one fresh
confirmation; a failure leaves one of two passing learners. Neither establishes
broad seed reliability. This comparison does not isolate why the loss change
helped on14009. The earlier replay-value0.3 confirmation failure remains recorded.

## Recipe and cost

Keep12M64/Auto/N8/B16/T16/fullBPTT and microbatch16/H5/R24; base, encoder and
effective behavior rates4e-5, dynamics1e-4, replay-value scale1.0, greedy
collection, sampled posterior, actorunimix0, intrinsic/extrinsic1, replay100k
and warmup1000. Replay-value representation gradients stay enabled; actor/critic
world gradients stay disabled. There are16,334,353 trainable parameters.

Keep expert100, unit action weights, expert world gradients and all-state
corrective labels;20% continuation-guided episodes with cap1920. Measure actual
overrides separately. Online guided returns are not frozen competence.
Keep full18 actions, sticky.25, repeat4, noops0 and native pooled RGB with one
GPU resize. Eight lanes use the same fixed seed offsets.

Native447fe50/SHA35ad0db, runnere304af89 and teacherfa261c53 remain byte-identical.
Runtime: `/x/Temp/kindle-qbert-color-masks-20261101/runtime`. No native build or
production change is needed. Start fresh: no reused weights, replay, belief,
game state, RNG state or experience.

There is no additional bounded learner. This exact recipe has already completed
both the8,192-action check and a200k learner. Its measured1,536.934591s full
training time plus10% projects **1,690.628050s (28m10.628s)**, below the unchanged
30-minute training cap. Startup, recovery and frozen checks still apply.

## Budgets and evidence

- Train200,000 aggregate actions /18,715 updates within1,800s. First update376;
  final residual credit.75.
- Retain the100,000-action /9,340-update checkpoint before final overwrite,
  including all four files, metadata identity and357 finite tensors.
- Four frozen arms: final/actual-initial × sampled/greedy. Each uses the first
  three natural episodes per lane,24 total;600k actions and300s per arm.
  Guidance off, zero updates, no truncations and357 invariant tensors.
- Replay the complete guided training pixel/action/label trajectory and all
  frozen action/reward/frame/episode trajectories; retain videos. Ordinary
  frozen logs lack live pixel hashes, so direct live-image equality is untested.
- Training replay cap1,200s, total audit1,800s, main service5,400s,
  postprocessing/progress6,000s, recovery1,800s and collection6,300s.

Require the exact configuration and source identities,18,715 finite reports,
changed CNN/reward/actor/critic tensors, aligned imitation reports, host guards,
curves, coverage, exposure and original service identities. Own initial tensors
must match the qualified357-tensor schema and differ from development
initialization. Cross-seed initial values and semantic traces need not match;
each learner is assessed against its own actual-initial controls.

Report scores, first-pyramid counts, both gate outcomes, training time,
initial-subtracted return/minute, late imitation loss and actual guidance.
Keep CPU co-host costs visible without subtracting them from measured time.
No retry, extension, cap increase, seed substitution, tuning, checkpoint
selection or automatic successor follows either outcome.

## Qualification and execution

CPU qualification passes in9.724s (9.833s wrapper): five actual commands,
38 config cases, three actual launch-preflight cases,19 header cases,
12 recovery-body checks, four observer-parser cases, six retention fixtures
and two integration checks, five teacher and eight collection-mode cases.
Both initialization readers accept retained different-seed tensors and reject
identical tensors (four cases). All21 scripts compile,52 paths resolve,815
qualified bounded pins and859 development pins remain unchanged. No native
agent or emulator is constructed.

The original CPU qualification failed on a duplicated package prefix in three
generated controller paths. Original sources and failure receipt remain under
`failed-cpu-preflight-source` and `cpu-preflight-execution.json`; the separate
path repair passes. This occurred before selection or GPU work. No completed
training, replay or collection stage is repeated.

The first declaration attempt then stops while hashing a historical full-run
`review.json`; that trial uses `learning-review.json`. No declaration or GPU
stage exists at failure. A separate CPU finisher uses the correct existing
review, verifies the already-written queue/manifest and completes the declaration
without repeating those writes. The original selection receipt and source,
repair script and prior CPU failures are pinned. The fixed protocol snapshot
retains the version copied before this documentation-only failure note. The
first finisher then fails before selection because its controller import path
is absent. Its original source and receipt remain pinned; adding that import
path completes selection in11.586s. No GPU stage is repeated. The subsequent
actual launch preflight passes all912 pins, then starts the single learner and
its original observers. No production or qualified controller source changes
after successful CPU qualification.

Evidence: `/x/Temp/kindle-qbert-replay-value1-seed16001-20261102/full`.
Trial: `qbert-12m-64-b16-h5-r24-dynamics1e4-replayvalue1-expert100-200k-seed16001`.
Main: `kindle-qbert-replay-value1-seed16001-200k-20261102`, with postprocess,
progress, recovery and collection observers. Recheck live host before launch;
do not overlap another GPU job, restart this learner or repeat terminal stages.

## Result-reader preparation while training runs

The separate `review_full_terminal.py preflight` passes at02:20:42.892 UTC
in2.485s (2.510s wrapper). It verifies all912 active pins and reproduces the
development learner's four fixed-cohort scores using the existing score reader.
The fresh-seed decision uses its own greedy gate, regardless of whether its
score exceeds the development mean. OriginalPID882846/invocationd78987af stays
unchanged before and after; no native module or emulator is imported.

ReaderSHA054976eb, score-helperSHAafe4018f and identity-helperSHA131bee46 are
retained in `terminal-reader-preflight.json`. After all original services are
terminal, run `review_full_terminal.py terminal` once through `run_cpu.py` to
verify completion files, journals, recovery and comparison before reporting the
confirmation outcome. This reader does not repeat an experiment or collector.
The separate `review_recovery.py` was prepared for the100k milestone; its
subsequent successful readout is recorded below. CPU co-host cost remains uncorrected; the original learner continues.

## Retained100k recovery

The original recovery observer succeeds **02:30:04.880 UTC November2**, with
100,000 actions/9,340 updates and357 finite tensors. Its audit covers a complete
log prefix of100,560 actions/9,392 finite reports. Four files totaling206,067,159
bytes are retained before final overwrite. Replay, live belief, game state and
RNG are not included, so this is not an uninterrupted restart state.

Separate read-only confirmation passes **02:30:39.931** in2.524s (2.542s wrapper).
All912 declaration pins, retained-file hashes, metadata/configuration and the
original recovery invocation/journal payload pass. The original learner identity
is unchanged before and after. Recovery is terminal success; training and the
remaining observers continue. No original audit, copy, checkpoint or GPU stage
is repeated. CPU co-host costs remain uncorrected.

MetadataSHA356bd3b7; checkpoint-reviewSHAded4637d; retentionSHA158aa03c;
confirmationSHAddffac5c. Frozen task performance remains unmeasured at this
milestone; the original200k budget and four frozen assessments remain fixed.
