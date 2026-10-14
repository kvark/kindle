# Correct the demonstrated packed-layout dispatch limit

October14,2026. Follow the [B8/B32 result](../results/2026-10-14-batch-granularity.md).
This is the next bounded implementation check, not an Atari allocation or batch
sweep. No GPU job for this correction has started.

Four RGB B32 convolutions lose the packed path because their64MiB packing
buffer requires65,536 X workgroups. The existing32Mi-element bound permits
that allocation, but one-dimensional admission rejects the grid. Meganeura's
existing `linear_grid` represents it as32,768×2×1. Use that helper and flatten
the packing shader's actual X/Y grid; do not add a new BF16-specific kernel.

Keep the packing element/memory cap, shared-memory gate, convolution tiles,
precision, reduction partitions, numerical tolerances and fallback conditions
otherwise unchanged. Preserve the separate reviewed128MiB logical gradient-
scratch setting in both controls. Work in a new isolated source snapshot,
retaining all completed batch artifacts and latest-upstream review.

Before GPU work, prove exact address coverage at the65,535/65,536 boundary,
padding/transposition correctness, bounds and unchanged B8 selections. Use all
supported packing storage types. The four B32 admissions must be explained
exactly, not enabled by broadening unrelated resource limits. Qualification
must test real two-dimensional dispatches, including nonconstant input, padded
channels, a second grid row and guard/tail behavior. Retain existing numerical
derivative checks for the packed family.

Then use finite short full-learner checks and one matched complete B32 timing
pair with identical source/model/optimizer/seed/batch settings except the grid
correction. Preserve B8's exact control. B32 now selects previously qualified
packed kernels: arithmetic association and later trajectories may differ,
and must be reported rather than hidden. Require independent numerical checks,
finite reports/checkpoints and exact initials; no learning-equivalence claim.
All packing costs must be included in the full update, with command caching
and the ordinary four-world-chunk schedule. Do not time just the pack kernel.

A measured whole-update benefit justifies keeping the correction isolated;
it does not close the full performance goal or justify a tile/grid sweep.
If it fails correctness, review before follow-up. If it offers no meaningful
whole-update improvement, retain the result and return to measured algorithm/
compiler costs. No new JAX benchmark or learning allocation is required here.

Declare exact commands/binary hashes after the CPU implementation review and
before GPU execution. Serial host-guarded persistent services, bounded deadlines,
expected5080 and>=2GiB sampled estimated Vulkan headroom; record standalone
allocation warnings. No NVML polling/recovery, blind retry, production change
or concurrent learner. Preparation one CPU/2GiB/no swap; GPU workers no quota.
Reserved artifacts: `runs/packing-grid-20261014.3ge6SKP4`.
