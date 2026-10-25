# Separating loading and computing warps does not reproduce the library pipeline

October15,2026. [Declaration](../experiments/2026-10-15-feeding-family.md) ·
[Indexing repair](../experiments/2026-10-15-feeding-family-index.md) ·
[Data, every contract and receipts](2026-10-15-feeding-family.json).

**Reject this implementation.** Across all23 convolution occurrences, complete
operation GPU time rises22.412→46.671ms,108.24% slower. Every one of20 distinct
contracts regresses against both native and same-algorithm GLSL controls.
Numerical qualification passes; the performance gate fails decisively. No
Naga change, favorable-shape selector, geometry sweep or full-update trial.

| Occurrence-weighted complete family | Native | GLSL serial | Separate loaders/consumers |
| --- | ---: | ---: | ---: |
| GPU pass time |22.412ms|24.418ms|46.671ms|
| Queued wall time |22.472ms|24.483ms|46.749ms|
| Checked32×32 products,9 occurrences |17.463ms|19.288ms|39.418ms|
| Packed64×64 products,14 occurrences |4.949ms|5.130ms|7.252ms|

All packing, scratch and final reductions are included. Serial GLSL calibrates
within the declared10% aggregate band: +8.95% in GPU and queued-wall totals.
This is **not per-kernel calibration**: CNN0 forward is1.431 versus.590ms, and
several derivatives also exceed10%. Against the same GLSL compiler, the pipeline
still costs91.13% more overall and loses every contract. Keep these discrepancies
and all128 samples per arm; do not interpret the totals as an exact native
instruction attribution. Each arm has16 warmups and128 measured calls.

CNN0's GLSL samples have a large retained transient: first/last32 means
4.101/.542ms for serial and3.151/1.323ms for the pipeline; native is.587/.594ms.
No samples are discarded, no steady-state result is substituted, and the cause
is unmeasured. The other58 of60 arm/shape series have first/last changes within10%.
This does not explain the broader regressing
family or establish a driver/clock cause for the full-update gap.

## What this says about the fundamental difference

The [actual library captures](2026-10-15-library-convolution-feeding.md) show
larger reuse tiles, asynchronous transfers and markedly less instruction work
feeding tensor operations. This trial changes **scheduling**, not reuse: two
subgroups load the same operands/address calculations while four compute;
two shared operand slots and ordinary full-workgroup barriers replace the
serial loading/computing schedule. It adds64 invocations and more shared storage
without reducing the operand/address work. Current32/64 tiles,K32,B512, layouts,
split ranges,256-term compensated partials and output rounding remain fixed.

The result rejects a tempting shortcut: expressing a legal subgroup-uniform
branch is not, by itself, a useful solution. The [Naga restriction](2026-10-15-feeding-pipeline.md)
is real, but this does **not** justify a compiler feature project. Additional
resource pressure, fewer simultaneous loaders and waits are possible contributors
to the regression; their individual costs, hardware overlap, registers/spilling
and SM utilization are not measured here. No claim that every pipelined algorithm
must lose, or that Vulkan cannot approach CUDA performance.

The complete learner remains the retained **51.945ms native versus27.223ms
inclusive JAX,1.91×**. This standalone family does not remeasure either full update.
Production stays on its existing runtime; isolated67559ce5 remains the candidate.
CDP's different exploration workload and non-image execution costs remain distinct.

## Qualification and the corrected indexing contract

The repaired allocation passes60 numerical guards and60 timing guards. All
23 actual full-world shader/grid/pipeline keys match their native controls.
Numerical qualification has4,035,648 independent F64 comparisons and516,574,592
exact native/candidate value pairs; timing adds another516,574,592 exact pairs.
Changed(-.5)/restored inputs pass; the GLSL harness also checks inputs and buffer
canaries. No tolerance changes. Numerical fixtures activate identical first/last
images in the full512-image allocation, retaining all original dW splits and
requiring exact-zero interiors; timing uses dense dyadic inputs. This is bounded
fixture coverage, not broad random-gradient qualification. Earlier cancellation
failures remain unresolved and are not erased by these passing fixtures.

CPU preparation validates180 typed SPIR-V variants,4,165,339 checked-iterator
addresses and113,664 loader/output ownership checks. Required32-lane and full
subgroups are enforced. Packed F32 variants remain over the48KiB shared limit;
their offline compilation is not GPU qualification. Only BF16 runs on the GPU.

An important source-review failure remains: the original producer index used
local invocation IDs modulo64 inside a subgroup-ID branch. Vulkan does not
guarantee that correspondence, even with full subgroups.
[Identity contract](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupLocalInvocationId.html).
The allocation was stopped after five passing cohorts/15 guards; the sixth
native guard was interrupted and has no completion receipt. The corrected
index uses subgroup ID/lane directly. All720 assignment permutations pass;
the old formula fails288. Original sources/results and25 allocation warnings
remain. Those45 completed operation calls plus an unknown interrupted prefix
are extra work, not replaced by the fresh allocation.

The corrected study costs8,940 standalone operation calls, zero game actions
or learner updates.31 allocation warnings occur in numerical qualification,
none in timing. Minimum sampled Vulkan estimated budget headroom15,828,123,648B
is not physical free/peak VRAM. Numerical service ends13:39:35UTC in8m41.527s;
timing ends13:50:25UTC in8m55.523s. Independent complete reviews pass; no worker
remains. GPU services have no CPU quota; CPU preparation/reviews use1CPU/2GiB/
zero swap. Service host-memory peaks/swap remain in the data, not VRAM readings.

Two CPU orchestration failures also remain: the build manifest lacks env_logger,
fixed by removing only native-wrapper logging and reusing unchanged compiled
shaders/C++ harness; and the aggregate audit was launched before the completion
receipt existed, then passed unchanged after completion. Neither repeats GPU work.
Meganeura99e3de27/Bladee7c3f7b remain unchanged upstream; no backend refresh here.

## Next discriminator

Return to **reducing repeated loading/address work through larger reuse**, not
another loader-count/prefetch sweep. First audit the output-store contract on
CPU: the current64×64 implementation allocates16KiB of F32 shared output scratch;
a64×256 output tile alone would require64KiB, already above the48KiB limit.
Determine whether exact final/partial layouts, rounding, tails and reductions
can be written without a full shared output tile. That would distinguish an
implementation constraint from an unavoidable API limit before selecting any
new combined reuse/feeding algorithm. It is a question, not a proved optimization
or a new GPU allocation. No rejected-family revival or automatic learning restart.

Artifacts: `runs/feeding-family-index-20261015.y5CaEr4O`; original stopped
allocation `runs/feeding-family-20261015.FnlUqnyM`. The wider root-cause/parity
goal, five-game competence and matched RGB/CDP budget claim remain open.
