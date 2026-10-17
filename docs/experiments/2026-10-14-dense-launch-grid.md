# Use the actual dense output grid for K-split selection

October 14, 2026. One bounded follow-up to the
[row-selection result](../results/2026-10-14-row-selection.md), not a tile,
threshold or geometry sweep. No Atari allocation or production promotion.

Unstarted: the03:03 UTC upstream check finds a relevant new Naga storage-
propagation fix. [Qualify that separately first](2026-10-14-storage-propagation.md);
do not combine compiler and launch-grid changes.

The [separate compiler comparison](../results/2026-10-14-storage-propagation.md)
now completes with unchanged reports/plans/tensors and no material speedup.
Start this still-unstarted geometry check from its immutableb3a086f native
and matching runtime sources (Blade96c1dc7/Naga a97559b), not the old compiler.
Its reproduced row-oracle rounding-boundary failure remains explicitly scoped;
all new dense-product numerical gates below remain mandatory and unchanged.

The isolated shared tiled candidate chooses dense K partitions with an assumed
32×32 output grid, then sometimes executes64×64 tiles. Grouped products already
use the actual grid. The CPU audit identifies55 RGB and53 CDP products affected
by applying the existing rule to the selected tile. This is source/plan evidence,
not measured utilization or a claim that more partitions will be faster.

Start from the refreshed row-selection candidate in a new isolated
artifact/worktree; preserve all older failures and source snapshots. Recheck
upstream before editing. Choose the existing kernel/tile before invoking
matrix_splits_for_workgroups. Keep128 target groups, K>=1024, at least256 K
elements/partition and16 maximum splits; retain fused-addend exclusion, F32
accumulation, BF16 storage rounding and all other kernel/learner choices.

First CPU-check exact selection/legal bounds, unchanged grouped/fallback/
addend paths, shader generation and scratch accounting. Existing independent
F64 typed product/gradient tests must retain their declared bounds; include
affected128/512-row forward and256-row transposed derivative shapes. Test
non-power-of-two partition counts, every output, input/output guards and
repeated execution. No timing before numerical review; no tolerance relaxation.
Changing reduction order may change stochastic trajectories, so exact final
weights are not a required equivalence claim. Initial tensors/configurations,
logical products, RNG consumption and precision policies must remain identical.

Then two24-update RGB/CDP smokes, followed after review by one ordinary
16-warmup+256-measured control/candidate pair each. Record complete stage
timings, all numerical reports, first outputs, initial/first/final tensors,
logical descriptors and actual scratch/headroom. Explain any compiler packing/
fusion side effect. Include partial reductions in timing. No extra candidate
if the gain is negligible; no additive estimate from isolated kernel timings.

Use existing persistent guarded GPU queues, bounded deadlines, Restart=no,
KillMode=control-group, expected5080 and>=2GiB sampled Vulkan estimated
headroom, no CPU quota for GPU work. Record allocation warnings; numerical/API/
hard faults stop for review, no blind retry, NVML polling or recovery.
CPU build/analysis uses one CPU,2GiB and zero swap, never during matched timing.
Publish the result and reassess expensive-work accounting; this check is not
an indefinite gate or a promise of JAX parity.

Started in `runs/dense-launch-grid-20261014.Upb23FRK`. CPU tests/Clippy pass;
nativef9cbeb98 keeps the same compiler as controlb3a086f. The first new GPU
test stops after its first case's outputs/partials/guard values pass: its
unchanged-parameter check incorrectly calls F32-only read_param on F16
storage. This is a test API assertion, not a numerical mismatch or wedge.
Retain the source/binary/failed guard. A test-only correction reads the packed
F16 bits using the existing raw buffer API; all values/gates/runtime sources
stay unchanged. Rebuild only the regression test, then run the reviewed
qualification from its first case. No learner timing before that review.

Reviewed qualification passes all six groups:14 affected-shape cases verify
1,769,472 outputs/6,684,672 partials exactly against F64 over three executions;
existing dense/gradient/convolution/grouped/staged checks pass unchanged gates.
Both24-update smokes have exact292/346 initial tensors and finite reports/states.
First RGB model tensors are exact; CDP first model relativeL2 is1.23e-5.
Optimizer states and later trajectories differ; categorical/forecast differences
are retained, not a learning-equivalence claim. The product audit permits only
the selected splits and necessary F32-partial/final-BF16 stores; all other
product descriptors match. The first CPU plan summary used the empty compiled
groups field; its reviewed replacement uses actual session metadata, retaining
the old file. CDP imagination adds90 dispatches/60 groups. No GPU rerun for that
metadata correction. Proceed to one ordinary pair/workload: RGB control/grid,
CDP grid/control. All preparation/analysis ends before timing.

Complete: [result](../results/2026-10-14-dense-launch-grid.md). Only1.00% RGB/
.92% CDP full-update improvement, with different trajectories. Preserve the
candidate, do not stack it as the performance baseline or run another grid
candidate. All12 guards pass after the retained test-only failure; four warnings,
1,136 full synthetic+144 small optimizer updates, zero game actions/workers.
Next re-profile the retained improvedb3a086f control's weighted residual costs.
