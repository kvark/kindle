# Reward logits: tiny tails dominate decoded scalar errors

The goal **prove the world model training on Breakout** remains unproven.
The [frozen head-only diagnosis](../experiments/2026-11-09-breakout-denoising-event-logits.md)
reconstructs every native scalar reward and positive probability from all1,600
logit rows. All12 cached ordinary event arrays match bit-for-bit and135 frozen
tensors stay exact. No learning, new RGB generation or production change.

The decoder accurately averages its categorical distribution; very small mass
on enormous scalar values dominates the error. This is not evidence of an
arithmetic implementation bug. It occurs even with actual next images, so
generated-image distribution shift is not its sole cause.

| Input task | Rows | Raw reward RMSE | RMS error from central contribution | Tail contribution RMS | Mean mass outside±32 |
|---|---|---|---|---|---|
|ordinary|512|65.604367|0.000208|65.604311|0.000021335|
|forecast|864|1592.516259|0.031191|1592.546944|0.000090655|
|actual|224|1803.231163|0.057521|1803.232860|0.000105700|

The raw mean is exactly central plus signed tail contribution, partitioned at
absolute scalar32. Squared error additionally includes their cross term; all
three terms are retained in review.json. Central contribution is unnormalized
probability times in-range support, not a new decoder or trained model. None of
these diagnostic readouts counts as a learning improvement or gate pass.

Ordinary zero-reward rows put99.9897% mean probability on zero, with mean CE
.000102941; zero is the modal scalar on all512 rows. Yet the residual outside32
has mean mass.0000213354 and dominates65.604 RMSE. One-step forecast rows include
four draws of one positive event: positive probability is about.9867 and central
contribution about.9703 for true reward1, but native raw rewards are about5565–5567
because tail contribution is about5564–5566. Actual inputs also expose a separate
recognition failure: arrival4/lane4 has positive probability.2758 for reward1;
arrival5/lane1 reaches.9866. Two examples cannot establish reliability.

Actual mode supplies real history and real current/next RGB at arrivals4..31,
including both positive events. It is a privileged classifier diagnostic, not
a causal world forecast. Forecast mode uses real past plus cached one-step RGB
at5..31, four draws; ordinary mode uses cached generated past/current RGB at
16..31. No reward/ending label enters native inference. Actual history is pushed
after each forecast-mode head query; ordinary history recursively uses generated
images. There are no terminal targets, and the ordinary slice is zero-reward
only. Outcome-selected reused clips do not prove natural calibration.

Source143e640b adds one ignored native collector; production sources are exact
tofecc251b. Only the retained sigma_data.05 age2048 checkpoint is queried.
22 focused CPU checks and strict release test-target Clippy pass; five GPU tests
stay ignored until selection.200 native head batches take0.684s including restore/
export, on a shared host. One terminal successful service and one sealed nine-file
host guard pass. Invocationb24317642b5d4b72a6992acc1ba24161. All39 declaration pins
and the separately retained CPU pins verify. No derivative/backend qualification
is repeated because no graph or learning code changes.

Raw255 logits per row, continuation logits, exact scalar support, decoded values,
all1,600 individual rows, six positive-case rows and the full MSE decomposition
remain. The own build cache is released after packaging; standalone executable,
source and frozen checkpoint evidence remain. All earlier jobs/failures/repairs
stay closed. This diagnostic selects no learner/extension/collection/automatic
successor. Full physics/events/natural calibration, changed-brick memory, late/
tunnel/second-wall/above-top, untouched families and fresh confirmation remain
required. Actor/critic, CDP comparison and five-game work stay parked. No fitting
score completes the goal. Root
`/mnt/data/kindle-breakout-denoising-event-logits-20261109`.

Closure verifies75 pins; own719,963,816-byte apparent build cache is absent.
