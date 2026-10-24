# Pong: test reliable play with more experience

The full goal remains DreamerV3 quality on Boxing, Pong, Freeway, Breakout and
Qbert with CDP, retaining the matched RGB budget comparison. This is one
bounded experience-budget study, not a replacement for that five-game goal.

The retained12M500k Pong seeds1009/2017/3019 score7.21/12.96/17.04, mean12.40,
with21/24/24 wins. Only3019 passes the full historical intermediate gate.
[All controls and videos](../results/2026-10-08-cdp-pong-capacity.md) remain.
Qbert's completed2M study shows useful forecasts/critics and gains from more
experience, but does not prove Pong will improve. The
[bounded compiler check](../results/2026-10-11-cdp-measured-build.md) is closed;
ordinary updates remain about108ms. Do not make optimization another gate.

## Fixed allocation

- Three fresh learners, seeds1009/2017/3019 in order, each2,000,000 actual
  actions/499,939 updates. Total6M actions, approximately24M emulator frames,
  and1,499,817 updates. No restore of completed500k lifetimes.
- Unchanged12M centered CDP/action-effects coefficient1: N8/B8/T16/H15/R32,
  microbatch8/replay100000, cosine500, encoder6e-6/dynamics4e-4/base4e-5,
  warmup1000, AGC.3, ac_grads=false;16.33M actual parameters including
  exploration. No capacity, loss, precision, entropy, exploration or BPTT change.
- Native pixels with one GPU RGB64 resize, full18/sticky.25/repeat4,
  no reset no-ops and100000-frame artificial cutoff. No action hint, RAM input,
  detector, externally shaped reward or video pretraining.
- Qualified Meganeura2c42b448/Blade56f0565/native0f26ee63, upstream rechecked
  October11. Measured construction is not enabled. Retain all12M500k and1M
  controls with their original pins; the backend differs, so no exact
  same-binary budget-ablation claim or historical repin.

Audit each learner's finite metrics/checkpoints, complete stream/replay/update
accounting, exact recipe,499,939 updates and zero final debt. Save actual
zero-experience weights before acting. Checkpoints omit live replay/RNG/belief;
an interruption is not an equivalent mid-life resume. Retain all failed work.

## Frozen evaluation and decision

After each learner, before the next, evaluate final and actual-initial models:
sampled policy, eight streams, reused development seed base4,000,000,000+seed,
first3 **natural** episodes per stream,600k actions/30min maximum per worker.
Select `cohort(..., natural_only=True)` explicitly. Require complete cohorts,
zero updates and exact346 saved tensors/model. Keep excess episodes, cutoffs
and unfinished tails, but never select cutoffs into the natural cohort.

CPU replay verifies all actions/rewards/resets/frames and writes whole
stream-zero videos. An independent CPU reviewer reselects natural cohorts
and checks wins from their replay outcomes. Rehearse that reviewer on the
retained three500k seeds before launch, with zero new GPU work or actions.
Automate all stages and the final three-seed budget/curve review.

Report all frozen scores/wins/actual initials, paired gains over retained500k,
score-versus-actions/time curves and learner-seed bootstrap uncertainty.
Streams/episodes are not independent learner replicates. Keep all online
reports, including the500k prefix; no favorable-checkpoint selection.
The historical gate remains>=90% wins and mean>=15 **per learner seed**.
The declared long-run reference remains20.44546396994667, with published
protocol/model/averaging differences disclosed. Neither winning most matches
nor passing the intermediate gate establishes the full five-game goal.

If weak seeds remain weak, review their learning evidence before any follow-up.
No automatic budget extension, other-game allocation, architecture sweep or
compiler retry. This study cannot establish a matched RGB compute-saving claim.

## Operation

Expected47–48h including evaluation;20h maximum per learner,30min per frozen
worker, one72h persistent systemd user service with Restart=no,
KillMode=control-group and zero swap. Serialize guarded GPU workers on
RTX5080/driver580.178.04; require>=2GiB sampled Vulkan estimated headroom.
Budget-minus-usage is not physical free memory or GPU utilization.
Record allocation warnings; API/numerical/hard faults/deadlines stop for review.
No blind retries, separate NVML polling, reset or recovery operation.

GPU workers have no CPU quota. Preparation/replay/review use one CPU,2GiB,
zero swap. Preserve source/native/ROM/job identities; no rebuild during the
allocation. Inspect training about every30min or at a stage boundary.

Artifacts: `runs/cdp-pong-2m-20261011.lb5DyI6h`. Stop after the three reviewed
train/evaluation pairs and final independent budget summary.
