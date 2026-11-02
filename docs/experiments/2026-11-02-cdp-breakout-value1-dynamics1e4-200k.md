# Breakout: fresh200k lower-dynamics/replay-value1 comparison

Select one fresh200k comparison after the [bounded check](../results/2026-11-02-cdp-breakout-value1-dynamics1e4.md)
and both original services close successfully. Its48.489s/8,192-action result
projects21m42.198s conservatively, below the unchanged30-minute training cap.
All382 pins,357 initial/frozen tensors, exact cadence/pre-update traces, replay,
guards and terminal reader pass. No bounded experience is reused.

The conditional hypothesis and [fixed recipe](2026-11-02-cdp-breakout-value1-dynamics1e4.md)
remain unchanged: only dynamics4e-4→1e-4 from the replay-value1 control.
Keep12M64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16, base/encoder4e-5,
behavior8e-5, replay-value1/world gradients, sampled collection/posteriors,
actor unimix.01, replay100k, action-effects1, expert100/unit weights/world
gradients, corrective labels and20% episode guidance/cap1920. Native447fe50,
runnerSHAea2bf8c6 and teacher stay fixed at the qualified check bytes.

Use fresh seed1009/evaluation4000001009 and original lane offsets; these are
reused development seeds, not independent confirmation. Start with no retained
weights, replay, belief, games, RNG or experience. Parameter count16,334,353;
full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64 resize remain fixed.

## Budgets and comparison

- Train200,000 aggregate actions /12,477 updates within1,800s; first update376,
  residual.5. Retain100k/6,227 recovery and all four checkpoint files before
  fixed final200k. No intermediate score or checkpoint selection.
- Final/actual-initial × sampled/greedy frozen arms: first three natural
  episodes per eight lanes,24 per arm;600k actions/300s caps, guidance/labels
  off, zero updates/truncations and357 invariant tensors each.
- Compare against sampled70.750/greedy143.667, own initials1.417/2.208,
  training21m27.910s. Sampled remains primary; report greedy separately and
  initial-subtracted score per training minute. The original90%-two-wall/864
  gate remains fixed. A higher mean alone is not task success or reliability.
- Full training pixel/action/teacher-label replay cap600s/process630s; each
  frozen outcome/frame/reset replay/video500s. Ordinary frozen logs lack live
  pixel hashes; do not claim unavailable direct image equality.
- Audit1,200s; main4,500s; postprocess/progress5,100s; recovery3,000s;
  terminal collector5,400s. No cap changes or repeated GPU stages.

Require357 finite matching initial tensors against control and bounded check,
47 matching pre-update transitions, paired actual-initial semantic traces,
12,477 finite aligned learner reports and changed CNN/reward/actor/value weights.
Validate actual commands, launch preflight, startup/recovery paths, dynamic
declaration inputs and source/native identities on CPU before launch. Retain
original observers, all replays, host guards, curves, exposure and comparison.
Report actual guidance fraction and final50k imitation fit separately from
frozen competence. Preserve CPU co-host durations without time correction.

No retry, extension, cap increase, rate sweep, confirmation, seed substitution
or automatic successor is selected. Preserve any original failure and inspect
saved evidence before a separate CPU-only repair. Earlier mixed dynamics and
failed teacher results remain; no isolated architectural cause is assumed.

Root `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102/full`, trial
`breakout-12m-64-b16-r16-value1-dynamics1e4-expert100-200k-seed1009`.
Main `kindle-breakout-value1-dynamics1e4-200k-20261102`, with original postprocess,
progress, recovery and collection observers. Check live state before launching;
do not overlap an active GPU allocation or restart a terminal stage.

## CPU qualification

The full harness passes06:43:22.361 UTC November2 in2.260s (2.365s wrapper):
18 scripts,800 declaration-path entries,44 literal paths,five actual commands,
32 config cases,seven actual launch-preflight cases,16 startup-header cases,
four observer-parser cases,11 recovery-body cases,six retention fixtures,two
collector-retention cases andfour metadata cases. All382 bounded/338 control
pins and357 tensor shapes remain unchanged. No native agent or environment
is constructed. Source preparation succeeds in.028s; no full preparation stage
fails or repeats. The earlier bounded source-generator failure remains retained.

## Original launch and startup

The first launch is **06:44:30.656 UTC November2/sourcec691416**,with422 pins
and declarationSHAc14ab341372b82c6c5b971cdc790826bdb2c6764f7346de9a63b59cb0321b859.
Main907551/47b00c7eccad47d6a116a2792b0a6f8b andall original observers are active:
postprocess907554/542a7c1be918475389fc39e292eca17d,progress907557/
e35cac0f9d5e46e5b4d67abe3448ffb5,recovery907686/ffcc2c76fdfe43389e50814923f093b5,
collection907689/e4b2e9d35bbc47b89a8feb3137c3942b. Do not restart or overlap them.

Startup passes **06:44:55.788** (1.489s wrapper):422 pins,357 finite initial
tensors matching both bounded andretained control,47 matching pre-update
transitions,first six updates376/392/408/424/440/456 and200 finite supervised
reports through3,576 actions. Fresh counters,configuration andsource identities
match. No frozen result orbenefit isavailable.

The separate terminal reader qualifies **06:45:41.914** in2.383s (2.404s wrapper),
reproducing allfour control scores andrejecting four corrupted score cases. All
422 pins andoriginal main identity remain unchanged; no native or emulator
module isimported. Record both CPU co-host costs without correcting training
time. Reader preparation succeeds in.016s; no successful stage repeats.

Qualified readers live in `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102`:
`review_recovery.py` (SHAcb71ef3c) runs once afterthe original100k retention and
recovery service succeed,while main remains active. `review_full_terminal.py
terminal` (SHAa8c0632e) runs once afterallfive services succeed andoriginal
completion exists. Use `run_cpu.py recovery` and `run_cpu.py full-terminal`
respectively for unique receipts. The terminal-reader preflight isalready
complete; do not rerun it, startup, completed checks orcollectors.

## Verified 100k recovery

The original recovery observer succeeds **06:55:37.842 UTC November 2**.
The checkpoint contains **100,000 actions /6,227 updates**, with 357 finite
tensors and all four files retained (206,067,163 bytes). Its audited complete
log prefix contains 6,391 finite reports through 102,632 actions.

The separate recovery reader passes once at **06:55:54.958**, in 1.291s
(1.311s wrapper). All 422 pins, retained hashes, original observer journal and
main PID907551/invocation47b00c7eccad47d6a116a2792b0a6f8b pass. The original
learner remains active. Recovery stores weights and metadata; replay, belief,
game and RNG state are omitted. Record CPU co-host time without correction.

Recovery confirmation is complete; do not repeat its reader or observer.
Only the terminal reader remains pending, after all five original services
succeed and completion exists. Final frozen gameplay remains unmeasured.

| Recovery artifact | SHA256 |
| --- | --- |
| Original checkpoint review |`24c9da83b7fea696bf64ae859d18178bba3fd4cc38597e04825a64dc2b4cf4b5`|
| Retention |`244ff20072ddd064ce7b084a0e775a8797f03572d581d9e5359460961081a1c1`|
| Separate confirmation |`a79ff50d3b899ba3483601c7338c19307bff5322fa7f7f3e2ba7d8e0aaae6f33`|

## Qualified saved-log coverage follow-up

Before final outcomes, a separate CPU reader qualifies **06:59:46.515 UTC
November 2/source03d7df9**, in 2.462s (2.482s wrapper). Four cases validate the
actual configuration and reject wrong dynamics rate, behavior rate and action
mode. The parent's five passing score/cap/frame boundary fixtures are retained
without repetition. All 48 diagnostic pins, 422 active inputs and the original
learner identity remain unchanged. Source preparation takes .017s; no failure,
native or emulator import, GPU work, replay or learning occurs.

After original completion and the terminal summary pass, read only this new
learner's training and four frozen logs once. Reuse the completed scale1 control
analysis from `kindle-breakout-score-coverage-20261102/review.json`,
`trials.candidate`. Keep pre-action score ranges 0–215,216–431,432–647,648–863,
864+, actual overrides/native actions and all completed/partial episodes.
Frozen summaries retain the original first three natural episodes per lane.
These score fractions do not identify wall boundaries or isolate the cause of
an outcome. The sampled-primary metric and original task gate stay unchanged.

Root `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102` contains
`review_score_coverage.py` (SHA18eca0beec0c6869e201274a3e55f2cacee988bd9f1b65c9482ebb4c4ece2c9e)
and `score-coverage-declaration.json` (SHA174ee79f0c51ec71adb01073556f3f46e701cb6ab99d30dd72c673ce587b9d83).
Preflight is complete; do not repeat it. Run terminal mode once via
`run_cpu.py score-coverage-terminal` after `full-terminal-summary.json` passes.
Output `score-coverage-review.json`. Preserve CPU co-host cost without training
time correction; no old analysis, environment replay or successful stage repeat.
No new learner, guidance change or successor is selected by this diagnostic.
