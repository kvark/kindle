# Current training cost after shorter imagination and the faster Qbert teacher

Learning accounts for about **81% of Qbert training time and88% of Breakout**
in the completed current recipes. Emulator stepping takes only4–5%. Within
Qbert's H5 updates, world training is now the largest recorded phase at44%,
followed by posterior preparation at24% and imagination at20%. Breakout's H15
updates still spend40% on imagination. This updates the older cost diagnosis;
it does not identify an implementation defect or a beneficial training change.

The analysis reads every saved update in four completed200k-action runs:
[Qbert10007](2026-11-01-cdp-qbert-rgb64.md),
[Qbert12011](2026-11-01-cdp-qbert-rgb64-seed12011.md),
[Breakout encoder4e-5](2026-11-01-cdp-breakout-encoder4e5.md) and
[Breakout dynamics1e-4](2026-11-01-cdp-breakout-dynamics1e4.md).
All use12M/RGB64/native447fe50. Qbert usesH5/R24 and the qualified faster
teacher; Breakout usesH15/R16. These different games, trajectories, settings
and CPU co-host histories do not provide an isolated cross-game speed estimate.

## Recorded wall-time accounting

The first three percentages below use total training wall time. The final
column is mean reported learner time, including its CPU and synchronization
work, per update; it is not GPU kernel time.

| Completed learner | Training time | Learning | Acting/guidance | Emulator | Updates | Mean update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qbert10007 |25m09.797s|81.31%|12.29%|4.04%|18,715|64.956ms|
| Qbert12011 |25m27.366s|81.14%|12.13%|3.99%|18,715|65.468ms|
| Breakout encoder4e-5 |21m27.701s|88.72%|4.91%|4.89%|12,477|90.496ms|
| Breakout dynamics1e-4 |21m48.095s|87.57%|4.82%|4.81%|12,477|90.745ms|

Acting includes native action selection, guidance generation and feedback.
Observation, reset, checkpoint and unattributed wrapper time account for the
remaining wall time. All original timings remain unchanged. In particular,
checkpoint timing differs between runs and must not be attributed to a
learning-rate effect.

The next table instead uses the sum of reported learner-update times:

| Completed learner | World training | Posterior preparation | Imagination/targets | Replay sampling | Behavior training |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qbert10007 |44.53%|23.68%|20.02%|7.03%|3.99%|
| Qbert12011 |44.21%|23.73%|19.88%|7.47%|3.97%|
| Breakout encoder4e-5 |31.66%|16.20%|39.84%|5.94%|5.81%|
| Breakout dynamics1e-4 |31.56%|16.02%|39.98%|6.03%|5.86%|

Replay refresh, synchronization and small inter-stage residuals complete these
totals. World training includes the learned encoder, recurrent model and
associated objectives; posterior preparation and imagination also use the world
model. These labels do not partition the architecture into independent modules.
GPU utilization and exclusive kernel costs are unmeasured.

Qbert's two runs spend541.302/541.621s in world training,287.831/290.692s in
posterior preparation and243.332/243.620s in imagination. Shorter imagination
therefore no longer dominates its update cost. The earlier H3 comparison was
negative, so this accounting does not select another horizon reduction. For
Breakout, the failed H5 comparisons likewise prevent treating the largest cost
as work that can simply be removed. Improving frozen gameplay per update
remains the priority; no backend optimization or new learner is selected.

## Verification and preserved CPU repair

All62,384 sequential learner timing records are finite and nonnegative. Their headers,
final counters and wrapper timers exactly match the original learning reviews;
all39 files referenced by the four original completion receipts remain unchanged.
The complete800k-action transition counts and declared update totals reconcile.
Twenty-one analysis inputs and all606 inputs of the active Qbert dynamics-rate
learner pass before and after, preserving originalPID857991/invocation
1e44edc5d824407b8a21cd62657f6ee2.

The first CPU attempt stops on its first timing row because it incorrectly
requires phase timers to sum exactly to the separately measured total timer.
The retained native447fe50 source confirms that the total also includes
inter-stage accounting. A separate repaired script preserves that residual
explicitly: totals are.014380/.015219/.009884/.009972s across the four runs.
Every per-update residual is nonnegative within the declared numerical tolerance.
Original failed source, declaration, stderr and1.894s wrapper receipt remain;
no completed analysis or training stage is repeated.

The corrected CPU analysis completes **22:42:05 UTC November1/sourcea369f61**
in4.289s (4.308s wrapper). It uses one low-priority CPU core and only the standard
library: no native/emulator import, model construction, GPU work or update.
Retain both attempts' co-host time without correcting the active learner's wall
time. Its original recovery and final assessments remain pending at this review.

Evidence `/x/Temp/kindle-current-training-cost-20261101` contains both attempts,
exact timing-source snapshots, repair provenance and complete phase values.
The successful analysis is complete; do not repeat it. The five-game goal remains
unfinished. No allocation, loss, teacher, model or successor is selected here.

| Artifact | SHA256 |
| --- | --- |
| Repaired declaration |`d986ac17f205fa516b45d8924e63522deba0aeb2dfa108bef1c5456fbdf921bc`|
| Completed review |`7880b17166fea1ca268f665858c5d360f025e40a6131ae3fad6f4f30440bc9b3`|
| CPU timer repair |`2a054094fc8d90dc7204fd2de23cb9281f40c639d2a952de7c8974b767c15c5c`|
