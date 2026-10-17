# First-maximum selection as a general row reduction

Complete: [results and retained failures](../results/2026-10-14-row-selection.md).
About9% fewer groups yields only1% lower complete-update time; no extension.

The [target attribution](../results/2026-10-14-target-boundary.md) places
compiler dependency depth ahead of host targets. RGB imagination's839 groups
match its839-node longest logical chain. This bounded change replaces the
categorical sampler's max-pool/broadcast/comparison/prefix-scan spelling with
a general first-maximum row selection, using existing generated reductions
and typed pointwise composition. No Dreamer-specific shader or random-law,
unimix, logits, RNG, model, precision-policy or optimizer change.

Artifact `runs/row-selection-20261014.8AcvPsxT`.
Reproduce the immutable GPU RNG/EMA control in isolated worktrees. Upstream
Meganeura2214412/Bladefe5f5b5 remain unchanged at02:08 UTC. Keep the control
nativea1c366a5; no installed pins/defaults or neighboring work changes.

The reduction tracks value and lowest winning column together. Equal maxima,
including signed-zero ties and infinities, select the first column; finite
input rows are the learner contract. Empty rows/invalid geometry are rejected.
NaN rows do not gain a new guaranteed learner meaning. Preserve deliberate
BF16 storage rounding, shared/public values and backward preparation roots.
Gradients through hard selection are zero, as in the previous comparison
graph; the existing straight-through estimator remains unchanged.

Before whole updates, CPU shader/geometry/cache/autodiff checks and one bounded
GPU qualification cover1/2/3/16/18/31/32/33/255/256/257/513/1025 columns,
row tails and actual sampling shapes. Compare one-hot outputs exactly against
independent CPU first-max values on the actual F32/BF16-rounded inputs.
Include all-equal and separated ties across lanes/strides, negative values,
signed zero, infinities, repeat execution/poison and explicit shared outputs.
Generated prologues and typed outputs must preserve stored rounding. Validate
an independently differentiated straight-through fixture against F64 with
pointwise bound2e-5*(abs(reference)+1e-4), and unchanged RNG.
Numerical or API failures stop for review; no gate relaxation or blind retry.

Then two24-update RGB/CDP smokes, followed after review by one ordinary
16-warmup+256-measured control/candidate pair each. Record all reports,
initial/first/final states and first categoricals/targets, logical operation
descriptors and compiler plans. Require exact initial states and finite
updates; any changed first selection must be explained by separately validated
producer rounding, not silently treated as categorical equivalence. Keep
later trajectory/packing differences explicit and make no learning-parity
claim. No extra geometry/fusion sweep if the full-update gain is negligible.

Report complete-update latency, dispatches and dependency depth together;
counts alone are not throughput or SM utilization. GPU jobs serialize under
the host guard in bounded persistent systemd services, Restart=no and
KillMode=control-group, expected5080 and≥2GiB sampled Vulkan headroom. Record
allocation warnings, no separate NVML polling or recovery. CPU build/analysis
uses one CPU,2GiB/no swap, never during matched timing; GPU work has no CPU
quota. No Atari actions or new learning allocation.

## Retained first qualification failure and bounded diagnosis

The first primitive job exits101 on9×1 F32, before learner smokes. No new
warning, API fault or unfinished child; the one-hot exact gate remains failed.
Keep its e257259b binary, original patches and guard. Before a numerical retry,
one separately bounded120-second diagnostic reads the nine selections and
scores, exporting the actual compiled plan/shader with graph optimization both
off and on. Same runtime, no learner update or threshold change. It is a
diagnostic export, not a passing qualification; review the cause first.

The diagnostic identifies two bad selections, exactly the±infinity rows, with
all nine input scores bit-exact in both optimization modes. Source explains
the fault: the static(1,0,1) reduction binding put zero into both trailing
uniform fields; the generated one-lane reducer therefore multiplied scores
by zero instead of the declared runtime one. Preserve those two fields in
the existing binding; no shader, tie rule or gate change. The unchanged
144-case qualification is separately rerun as `numerical-reviewed`, after
the CPU suite/build. Keep the original failed guard and diagnostic.

Both24-update learner smokes have exact first outputs, all24 metric reports
and all initial/first/final292 RGB /346 CDP tensors. The first CPU inventory
audit stops on one compiler-plan difference: an8×2048×256 RGB backward matrix
no longer absorbs its addend; that add appears in its generated pointwise
consumer instead. This is not missing arithmetic. Preserve the failed audit
and original script. The reviewed inventory permits exactly this one explicit
descriptor difference, requiring all other geometry/storage/kernel fields
and all other logical products to match. Full timing will measure this
compiler side effect as part of the candidate; no source or numerical gate
changes, and no claim of identical plans.
