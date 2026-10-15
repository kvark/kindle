# Extra exploration work and a missed compiler composition

October15,2026. [Declaration](../experiments/2026-10-15-cdp-imagination-work.md)
· [Data](2026-10-15-cdp-imagination-work.json).

**The current action-effects ensemble accounts for12.07ms of fixed imagination
under an exact-output intervention.** Replacing only its closed computation
branch with its saved bonus changes complete imagination GPU20.771→8.705ms,
58.09%. All seven outputs remain bit-exact. This additional workload is absent
from the RGB/JAX reference; it is not all backend overhead or a new JAX ratio.

Normal/substituted/substituted/normal block means are20.770/8.704/8.706/20.772ms.
All256 measured samples remain,16 warmups+64 measured replays per block. Queued
wall means follow the same difference. These are fixed-state replays, not an
ordinary full-update improvement, pure kernel latency or removable learning
cost: an actual actor still needs a computed bonus. Current complete CDP50.122ms
and RGB/JAX54.224/26.632ms remain unchanged.

## What the ensemble actually computes

Four members each process7,680 imagined states and18 actions. State projection
reuse and hidden-space centering already exist; the old18-way first-projection
diagnosis does not apply. The12 actual matrix descriptors/parameter paths give:

| Projection, four occurrences each | M×N×K per occurrence | Nominal MACs total |
| --- | --- | ---: |
| Reused state |7,680×256×2,560|20.13G|
| All-action second hidden layer |138,240×256×256|36.24G|
| Selected centered output |7,680×1,024×256|8.05G|

Total64.42G. The90 bonus-exclusive dispatches also have3.361GB of logical output
extents. These counts are neither measured DRAM traffic nor additive time.
The remaining802 dispatches contain the shared recurrent/policy/value workload;
their8.705ms remainder is not entirely overhead. Targets and slow-value work
outside this imagination graph are not part of the intervention.

## Concrete compiler miss

The saved plan contains four first-layer chains that expand each projected row
18 times, transpose it, then feed a typed normalization. Eight expansion/
transpose outputs total **1,132,462,080 logical bytes per imagination pass**.
Each is private, single-consumer F32 data; the normalization stores BF16 and its
rounding boundary must remain.

The backend already supports repeated-row input addressing. But its matcher
runs before typed scalar/normalization fusion exposes these inputs, and the
resulting precision-protection marker also excludes them from the older matcher.
The CPU audit verifies all four concrete chains against the matching conditions,
pass order, precision markers and producer/consumer lifetimes. This identifies
a missed composition in the isolated candidate, not a missing hardware feature
or a measured1.13GB memory-bandwidth penalty. Latest upstream99e3de27/e7c3f7b is
rechecked; upstream does not contain this experimental typed-row-fusion module.

Next test general repeated-row view composition through typed normalization,
retaining arithmetic order, BF16 rounding, protected buffers and all consumers.
Use numerical controls and one complete-update comparison, not a row/geometry
sweep or an exploration change. Its speed benefit remains unmeasured. This
addresses a concrete whole-program materialization difference; convolution
feeding and the separate RGB/JAX goal remain open.

## Validity and completion

A versioned read/producer graph includes fused prologue/epilogue/norm operands,
extra outputs and overwritten buffers. Independent CPU reconstruction verifies
the90-node closed branch and that no retained operation reads a skipped value.
The30,720-byte device-local snapshot stays resident in both conditions. Original
allocations/pipelines/dependencies remain; only branch workgroups become zero,
with an exact-word copy at the original bonus producer. This is a valid-boundary
intervention, unlike the earlier masks whose downstream values became invalid.

Seven finite outputs, all input/parameter hashes and346 saved tensors stay exact.
Initial/first tensors and first outputs/reports match the retained BF16 control;
full plan/pipeline/group restoration passes. Three complete tensor pairs and
1,797 source seals pass. One initializer,325 fixed complete imagination replays,
zero game actions; one guard passes with two retained allocation warnings.
Three CPU dependency tests and strict Clippy pass; no new failed process.

Native86d87c82; no backend algorithm or production change. Service finishes
06:11:11 UTC in28.53s; CPU review2.226s/573.4MiB under1CPU/2GiB/no swap. Minimum
sampled Vulkan estimated headroom12,368,216,064bytes, not physical/peak-free VRAM.
No worker, separate NVML polling, recovery, learning restart or goal completion.
Artifacts: `runs/cdp-imagination-work-20261015.UeGXjU8B`.
