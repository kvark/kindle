# Convolution backward coverage diagnostic

The [shared tiled experiment](../results/2026-10-13-tiled-gap.md) improves dense
arithmetic but leaves RGB world training at30.15ms. Isolate the missing native
BF16 convolution-backward path on the prior unchanged-kernel control, not on
top of that tile change. This is diagnostic implementation, not adoption.

Use Meganeura056562b with already-qualified Bladefe5f5b5/Nagaa4b4183. Extend the
shared checked, type-aware cooperative loop to convolution input/weight
derivatives. Explicit BF16 forward convolutions in the opt-in reduced-gradient
path use BF16 incoming derivatives, FP32 accumulation, FP32 master gradients and
unchanged optimizer/reductions. Default FP32 differentiation stays unchanged.
Keep the existing convolution split-selection rule and16-position partition
boundaries; mask the last stage at each boundary. No geometry search.

Validate shader generation on CPU, then full independent F64 derivatives using
explicitly BF16-rounded inputs, weights and incoming derivatives. Cover partial
channels, rectangular kernels, padding, stride, long reduction partitions and
small1e-12 gradients. Retain existing strict-backward checks and their gates.
Numerical failures stop for source review, not tolerance adjustment.

After those checks, one ordinary RGB12M/B8 matched control/candidate pair uses
16 warmup +256 measured updates, the same actual initial tensors, inputs and
optimizer. Audit finite reports/states, first forward outputs, graph changes,
partition counts and headroom. Final stochastic trajectories need not be exact:
this intervention changes arithmetic. It does not establish learning quality.
CDP mixed-world arithmetic is not silently changed to BF16, and its timing is
not compared with RGB/JAX. No new Atari actions or training-default change.

Artifacts: `runs/conv-backward-gap-20261013.8N61hY2O`. Preparation/analysis:
one CPU,2GiB, zero swap. GPU checks: serialized guarded systemd service, bounded
120s per numerical process and300s per full-update process, no CPU quota/NVML
polling, expected RTX5080, ≥2GiB sampled Vulkan budget headroom. Retain allocation
warnings; other faults stop the affected job. No driver recovery or blind retry.

Measure the complete update, including new casts and partial reductions. A small
or negative gain is evidence that precision coverage alone does not supply a
good convolution algorithm. Do not infer parity by adding isolated savings.

Preparation retains the first CPU build failure: the shader-fixture test's
exhaustive entry match needed the new gradient variant. Corrected before any
GPU work; no numerical gate changed. Original patch/log remain in the artifacts.
The second CPU build is deliberately stopped before GPU binaries are published
to correct that fixture's gradient binding names. Both source snapshots remain.
A subsequent preparation-only stop makes oracle intermediates non-aliased so
the test can inspect saved partials after the complete step. No GPU work ran.

## Reviewed numerical failure and bounded correction

The first GPU check stops after the32 shorter cases on an unsplit65,542-term
weight reduction: −2.458471 versus F64−2.4589882, exceeding the unchanged
`1e-5 * scale + 2e-4 * abs(reference)` pointwise gate. No timing or remaining
qualification runs start. The failed source/binary/guard remain; no driver fault
or recovery is observed. Rechecked upstream main2214412 is unchanged;31026833
fixes partition boundaries, not long unsplit accumulation.

The corrected typed-gradient lowering imposes a numerical floor of one K
partition per4,096 positions even when optional occupancy splitting is disabled,
using the same16-position boundaries and FP32 final reduction. Partial storage
is explicitly bounded; insufficient scratch is rejected before dispatch. This
is a changed accumulation algorithm, not a tolerance change or removal of the
failed case. Ordinary FP32 paths remain untouched. Repeat the original40 cases
and17 independent partial checks with new binaries/output receipts, then execute
only the still-unstarted controls if qualification passes. The nominally
"unsplit" long fixture now tests mandatory bounded accumulation; it is not
claimed as successful long-unsplit execution. Compare actual complete-plan
partition choices before claiming they stayed fixed in the learner fixture.

Both still-unstarted timing arms enable the existing NVTX host annotations to
export complete plans before warmup. No profiler or GPU timestamps are enabled;
the annotations are identical in both arms, and their reported eight-update
range is not a claim that a capture was taken. The CPU audit compares actual
convolution descriptors/partition counts as well as the other session summaries.
The bounded build wrapper's mistakenly rewritten target-directory name is
corrected before GPU work; its short preparation-only stop remains in the journal.
