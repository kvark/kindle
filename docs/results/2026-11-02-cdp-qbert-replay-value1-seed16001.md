# Qbert replay-value1 confirmation misses the mean-return gate

Fresh seed16001 scores **14,913.542 greedy** and **14,745.833 sampled**, each
with **24/24 first pyramids**, after **24m45.754s** of training. Both means are
below the fixed15,000 requirement. Greedy is86.458 points short; this remains
a failed task gate. The experiment itself completes with all audits passing.

The unchanged recipe therefore passes primary greedy on **one of two learners**:
development14009 passes and fresh16001 fails. The earlier development gain from
replay-value0.3→1 remains valid on14009; this confirmation does not establish
two-learner success or broad reliability. No extension, retry, seed substitution,
third learner, tuning or automatic successor is selected.

| Fixed recipe / learner | Greedy mean | Sampled mean | First pyramids, both modes | Training time | Greedy gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Development14009 |16,525.000|14,882.292|24/24|25m36.935s|Pass|
| Fresh16001 |14,913.542|14,745.833|24/24|24m45.754s|Fail|
| Own initial14009 |147.917|205.208|0/24|—|Fail|
| Own initial16001 |86.458|148.958|0/24|—|Fail|

Both sampled means fail15,000. Every reported score selects the first three
natural episodes per eight lanes. No truncated episodes occur. Raw end-of-run
means can include extra episodes and are not used for this decision.

## Fixed recipe and measured work

The [protocol](../experiments/2026-11-02-cdp-qbert-replay-value1-seed16001.md)
changes only learner/evaluation seeds:14009→16001 and4000014009→4000016001.
Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT and microbatch16/H5/R24;
base, encoder and effective behavior rates4e-5, dynamics1e-4, replay-value1,
greedy collection, sampled posterior, actorunimix0, intrinsic/extrinsic1,
replay100k and warmup1000. Expert100, unit action weights, expert world gradients,
all-state corrective labels and20% continuation-guided episodes/cap1920 remain.
Full18/sticky.25/repeat4/noops0 and one GPU resize from native pooled RGB stay fixed.

Native447fe50/SHA35ad0db, runnere304af89 and teacherfa261c53 are byte-identical
between learners. Runtime remains
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`. No weights, replay, beliefs,
games, RNG state or experience are reused. No additional bounded learner runs:
the unchanged recipe already completed its check and a full development learner.

Fresh training completes **200,000 actions /799,817 frames /18,715 updates** in
1,485.754463s, under the1,800s cap. The first update is376 and final residual
credit.75. Learning accounts for1,206.914s; acting185.427s, environment61.140s,
observation13.043s, reset.546s and checkpoint1.836s. These are measured wall
timers, not exclusive GPU costs. Different trajectories and CPU co-host costs
preclude an isolated runtime-speed claim.

| Descriptive quantity | Development14009 | Fresh16001 |
| --- | ---: | ---: |
| Initial-subtracted greedy return/training minute |639.341|598.770|
| Initial-subtracted sampled return/training minute |572.975|589.473|
| Actual overridden actions |56,526 (28.263%)|48,177 (24.0885%)|
| Final50k replay expert CE |.059228|.068976|
| Guided online last50 mean |15,799.5|15,187.0|

These are different learner and evaluation cohorts, not a paired intervention.
Guided online return and replay fitting do not establish frozen competence or
explain the cross-seed difference. The existing replay-value term already uses
real rewards and boundaries with imagined-return bootstraps; this confirmation
changes neither its targets nor its coefficient.

## Complete evidence and original closure

All **912 pins**, the100k/9,340 recovery,357 finite initial and invariant frozen
tensors, full training pixel/action/label replay, four frozen outcome replays
and videos, host guards, curves, coverage, exposure and comparison pass.
All18,715 learner reports are finite. Initial tensors match the qualified schema;
48 of357 differ from development, and all47 pre-update transition records differ.
Cross-seed initial equality is not required; own actual-initial controls are used.

The four frozen arms total **140,360 actions /561,276 frames /104.749772s**,
with zero updates and zero truncations. The ordinary frozen logs lack live
pixel hashes, so no unavailable live-image equality is claimed. The final
comparison plot is visually checked; its curves are guided online returns and
its caption separately reports frozen means and training cost.

MainPID882846/invocation`d78987afbb3344c3b65b84d3652ceb79` launches
02:17:27.588 UTC November2/sourcebff6eb4. Original terminal times are:

- Main: **02:49:22.758**.
- Postprocessing: **02:49:37.668**.
- Collection: **02:50:08.343**.
- Recovery: **02:30:04.880**; its four retained files total206,067,159 bytes.

All five original services succeed. The progress observer has no separate
resource-consumption timestamp; its original identity/journal and successful
terminal state are verified. No original training, frozen, replay, audit or
collector stage fails or repeats. Separate read-only recovery confirmation
passes02:30:39.931 in2.524s. The terminal reader runs once and passes
**02:50:48.666** in5.166s (5.184s wrapper), preserving all original completion
files and verifying the fixed primary decision: one passing learner of two.

The earlier CPU preparation failures remain: a duplicated generated path,
a historical full-review filename mismatch and a missing finisher import path.
Separate repairs precede the single GPU launch. Their original sources and
receipts are pinned; they are not hidden or relabeled as original successes.
Startup, result-reader preparation, recovery confirmation and the separate
6.366s Breakout consistency diagnostic co-hosted on CPU. No wall-time correction
is applied.

Root `/x/Temp/kindle-qbert-replay-value1-seed16001-20261102` is closed. Do not
repeat this learner, observers, collector or terminal reader. No Kindle GPU
allocation remains at closure; check live state before separately selected work.
The five-game goal remains unfinished, including Breakout's zero two-wall result.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`384851cbe88f3e64c4ac8a829e7c1afb71ba5bed9241e11b4143ea96af1f19fb`|
| Learning review |`d4462371f6a1271e3447f601926e09a7a7e2c50693274f107b90e4b007e050ae`|
| Original completion |`587f591ed93ea3fe569475ea27d6d538fee46f0b409c4118984e32d389c138c9`|
| Terminal summary |`8d825bdf3ad8ea4755bcf07b45fcf25e40a48588933a57d8d8ef49eafbe325ea`|
