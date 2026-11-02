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
