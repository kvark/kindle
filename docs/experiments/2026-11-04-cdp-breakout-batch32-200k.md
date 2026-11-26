# Fresh 200k Breakout comparison with larger effective batches

Select one full learner after the [B32/T16 bounded check](../results/2026-11-04-cdp-breakout-batch32.md)
passes execution and its fixed cost screen:44.142s versus47.690s,7.44% lower,
padded200k19m45.465s. Later warmup gives3.07% fewer sampled positions; learning
time per position falls3.11%. Sustained cost and frozen gameplay remain unmeasured.
The purpose is useful frozen learning per minute, with the original864/90%-two-wall
gate unchanged. This separately selected comparison is not a batch sweep.

Use the exact qualified native447fe50/SHA35ad0db, runnercf4ea238, teacher and
B32/T16 configuration. Relative to retained143.667 greedy/70.750 sampled,
only batch_size16→32 and world_microbatch_size16→32 change. T16/fullBPTT,
centeredCDP12M64/Auto/N8/H15/R16/context1, base/encoder4e-5, dynamics4e-4,
behavior8e-5, horizon333/lambda.95, policy/value/replay-value1, representation.1,
dynamicsKL/free-nats/continuation1, CDPfuture500, disagreement1, expert100/unit
weights/world gradients, all-state corrective labels,20% episode guidance/
cap1920, sampled collection/posteriors, actor-unimix.01 and replay100k stay fixed.
Full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64 resize stay fixed.
No new build, optional reward, crop or loss. Batch diversity, centered-target
statistics, cadence and update-based warmup change together; no isolated cause.

Fresh1009,200,000 aggregate actions/6,231 updates,30-minute training cap.
First update632, then every32 actions, residual credit.25. Preserve100k/3,106
recovery with357 finite tensors. No weights, replay, beliefs, RNG, game state
or experience transfer from prior runs. Both initial and frozen tensors must
match the exact schema; all357 actual-initial values must match the retained
and bounded initial checkpoints. Shared pre-update semantics through376 actions
must match retained. Finite reports, weighted totals and480-label bounds required.

Run final and actual-initial sampled/greedy assessments onseed4000001009,
firstthree natural episodes per eight lanes=24 each, with guidance/labels off,
zero updates/truncations and357 unchanged tensors per arm. Keep the existing
25,000-action per-lane episode limit and300s per-arm cap. Require actual-initial
semantic traces to match retained. Training and every frozen arm receive the
existing host guard. Preserve failure evidence; no cap changes or completed
stage repeats.

Audit full training pixel/action/label/override replay, complete frozen recorded-
action/outcome/frame/boundary replay and videos, curves, exposure, recovery and
terminal identity. Ordinary frozen logs lack live pixel hashes; do not claim
equality to unavailable images. Report actual overrides separately from labels;
online guided returns are not unaided competence.

Greedy is primary. Favor only higher raw greedy mean and own-initial-subtracted
score per training minute, with the fixed paired eight-lane10,000-bootstrap95%
gain/min interval entirely above zero. Report sampled and raw paired intervals,
all per-lane returns, actual initial controls and unchanged task success. Seed/
cohort reuse is development evidence, not independent reliability. Keep actual
CPU co-host time without correction. No checkpoint selection, extension, second
learner, batch sweep or automatic successor follows either result.

Qualify all five actual commands and their audit expectations before launch.
Use original main/recovery/terminal observers, no concurrent GPU learner.
Training1800s, each frozen300s, training replay900s, each frozen replay500s,
main7200s, collector7500s; retain existing subprocess failure handling. An
observation timeout never authorizes a restart. Independent closure reads saved
outputs/journals only. Check live state immediately before launch.

Root`/x/Temp/kindle-breakout-batch32-20261104/full`; service
`kindle-breakout-batch32-200k-20261104`. Preparation is pending at selection.
The five-game goal remains unfinished.
