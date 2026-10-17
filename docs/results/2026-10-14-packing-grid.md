# The packing cliff is real, but explains little of the gap

October14,2026. [Declaration](../experiments/2026-10-14-packing-grid.md) ·
[Data](2026-10-14-packing-grid.json) · [Backend plan](../backend_training_performance_plan.md).

**Correcting the four B32 convolution fallbacks saves only1.49% of complete
update time:68.344→67.326ms.** This is below the declared3% gate. Do not promote
the candidate or expand this into a packing/grid/geometry sweep. The remaining
ratio is2.48× retained complete JAX27.187ms; B8's1.79× gap is unaffected.
The admission defect does not explain the large performance discrepancy.

One ordinary matched pair,16 warmup+256 measured updates per arm, RGB12M/
B32/T16/H15/BF16. Both use cached commands, four world submission chunks and
the same128MiB logical gradient-scratch allowance. Packing, host work and
synchronization are included. JAX was not rerun; its retained complete boundary
includes seeds, metric materialization and final drain. Cross-framework
weights/RNG differ. This is not a statistical latency interval or learning result.

| Complete-update stage | Control | Grid correction |
| --- | ---: | ---: |
|Posterior|9.215ms|9.215ms|
|World training and sync|36.830ms|35.831ms|
|Imagination and targets|17.714ms|17.786ms|
|Behavior training and sync|4.586ms|4.493ms|
|**Whole update**|**68.344ms**|**67.326ms**|

## What the intervention proves

The existing32Mi-element packing cap permits each64MiB BF16 allocation, but
the selector rejects65,536 one-dimensional workgroups. The correction uses
the existing `linear_grid` helper and flattens actual X/Y coordinates in the
shared F32/F16/BF16 packing shader:32,768×2×1. Only two production source files
change. Memory bounds, convolution algorithms/tiles, precision, reduction
partitions and numerical tolerances remain unchanged.

Exactly four convolutions switch from checked to packed kernels: decoder32×32
forward/dW and encoder32×32 dX/dW,80.531G nominal MACs. Eight packs are added,
four two-dimensional. All other dispatch descriptors and dependencies match
after resolving pack buffers to their original inputs. Packing changes the
world's topological order, not its logical graph; every producer precedes its
consumers. The other three sessions are unchanged. The exposed posterior plan
is the same world plan, not another four independent changed operations.

**65,535 is the selector's portable bound, not the5080's actual X limit.**
A separate guarded capability query reports[2,147,483,647,65,535,65,535].
Every retained B8/B32 launch fits those advertised limits, including unrelated
one-dimensional casts above65,535. This was an unnecessarily conservative
selection rule, not an observed invalid GPU launch or driver fault.

## Qualification and retained failures

- All14 packing cases/28 executions pass:536,894,080 exact values across
  F32/F16/BF16, both storage orders, changed inputs, padding, full two-row
  grids, a partial second-row tail and64 canaries. Logged validation passes.
  Existing40 packed-convolution F64 cases/seven independent partial checks pass.
- Five CPU tests and24 typed shader variants pass. The CPU retained-plan check
  leaves B8 identical and admits only the four intended B32 convolutions.
- B8's24-update smoke reproduces every report/first output, five plans/pipeline
  inventories and292 tensors at initial/first/final exactly. B32 smoke and full
  runs retain exact initials and finite reports/states. The full control exactly
  reproduces its retained272 reports, first outputs, five plans/pipelines and
  all292 tensors at all three boundaries. The candidate reproduces its own
  smoke's first24 reports, first state/outputs and plans/pipelines.
- B32 arithmetic association changes.100 first-update and289 final saved
  tensors differ; aggregate relative L2 across parameters **and optimizer
  state** is.02974/.07614. This is not a policy-error measure or learning parity.
- Seven guards pass, two allocation warnings retained, zero GPU/API/numerical
  failures, recovery, separate NVML queries or unfinished children.592 synthetic
  updates, zero game actions. Both full arms sample12,665,225,216-byte minimum
  Vulkan estimated headroom, not physical free/peak VRAM.
- Retain four CPU failures: the test's missing clone; an overbroad portable-grid
  assertion on unrelated casts; a checker that assumed unchanged dispatch order;
  and a300-second stream-sorting audit timeout. The corrected graph check passes.
  Whole parsed-plan canonicalization retains every field/value and the same
  legacy-empty-output normalization; final review completes in35.52s within
  one CPU/2GiB/no swap. No GPU run or numerical tolerance is changed/repeated.

Formatting/release Clippy pass. Candidate9ad4b2b6, control e5d890dd, numerical
b7b7c86f; production/default pins and Atari remain unchanged. Sources, locks,
binaries, outputs and failed checks: `runs/packing-grid-20261014.3ge6SKP4`.
Upstream7574ad1/e7c3f7b rechecked unchanged; CI399 passes on preceding238a0d4.

## Next root-cause question

The [batch study](2026-10-14-batch-granularity.md) shows exact fourfold nominal
heavy work, but native throughput scales1.33× versus JAX1.87×. This intervention
rules out the four admission fallbacks as the dominant explanation. It does
not identify the remaining larger-batch bottleneck from operation counts alone.

[Next bounded attribution](../experiments/2026-10-14-batch-residual.md): reuse
the qualified cached fixed-session masks on the retained B32 control to locate
the growing convolution/matrix/other cost. One preparatory update, no learning;
preserve every dependency and check exact full restorations. Then match the
measured dominant operations to the actual JAX executable before choosing an
algorithm/compiler intervention. Existing kernel instruction waste and cheaper
CUDA graph dependency execution remain established structural differences.
No larger batch, local tile/dtype sweep, production promotion or whole-goal closure.
