# Spatial metadata: complete family rejected

October16,2026. [Declaration](../experiments/2026-10-16-convolution-metadata-kernels.md)
· [Compact evidence](2026-10-16-convolution-metadata-kernels.json).

**The metadata implementation is correct but92.92% slower.** It replaces spatial
address arithmetic with a device-local table while retaining the existing loads,
tiles and arithmetic order. All20 contracts/23 occurrences complete;19 contracts
regress. Reject integration, a favorable selector and a full-update trial.
Keep native58aff452 and **RGB46.199/JAX27.223ms,1.70×**. Production and Atari
remain unchanged; the root-cause goal stays open.

## Complete-operation result

| Weighted family cost | Native | Paired control | Metadata |
|---|---:|---:|---:|
| GPU,ms |19.1012|19.1119|36.8706|
| Queued wall,ms |19.1610|19.1685|36.9284|
| Command recording,ms |.01565|.10213|.10561|

All original transformations/reductions and all128 ordinary samples per arm
remain. Queued wall regresses92.65%; both5% benefit gates fail. Control/native
calibration is+.0559% GPU/+.0388% queued wall, within the original5% aggregate
gate, not proof of per-kernel calibration. No profiling or compilation occurs
inside ordinary timing. These are complete standalone operations, not a complete
learner update or pure main-kernel latency.

The sole improving contract, dX32→3, improves6.97%; the other19 regress36.88–
190.10%. It does not justify selecting a subset. All7,680 samples stay. The
largest absolute first/last32 shift is4.21% in the dX32→48 control; no samples
are discarded and no new steady-state interpretation is imposed.

## What the counters establish

| All23 main products | Control | Metadata | Change |
|---|---:|---:|---:|
| Warp instructions |10.296B|11.970B|+16.26%|
| ALU instructions |6.819B|6.188B|−9.25%|
| LSU instructions |774.339M|991.779M|+28.08%|
| L2 bytes |15.976GB|54.675GB|+242.24%|
| DRAM bytes |1.633765GB|1.633962GB|+.0120%|
| Tensor instructions |180.224M|180.224M|exact|

CBU work increases22.57×; FMA-pipe work increases77.22%, which is not a claim
that the unchanged mathematical graph performs more floating-point arithmetic.
Active threads per warp instruction change31.997→31.390; local reads/writes are
zero in both arms. These ratios are not SM utilization or source-line attribution.

**Precomputing the coordinates does not make operand delivery cheap.** The
consumer now has a dependent table read before its source read; removing some
ALU work adds other instructions and traffic. The unchanged DRAM volume does
not mean this data path is free. Nor does aggregate L2 traffic explain every
regression: several packed forward/dX contracts read fewer L2 bytes while taking
over twice as long. This rejects this particular lookup implementation, not all
precomputation or a proven single-cause latency model. No dependency-stall,
driver-inlining or source-site counter attribution was collected.

All retained control instruction anchors, tensor counts, complete plans,
pipeline keys, bindings and original operation segments reconcile. All2,000
counter passes and6,728 offline metric values pass independent review.

## Implemented and checked

- One type-agnostic transform covers typed and packed forward/dX/dW. It changes
  only B spatial feeding, not the rejected A-address carry rewrite. Original
  cursors, A loads, matrices, K order, splits, compensation, stages and output
  handling remain exact source prerequisites.
- 320 paired F32/F16/BF16 modules plus the initializer validate offline;266 old
  control modules reproduce exactly.302 consumers change;18 dense/grouped
  controls remain byte-exact. All20 current contracts/23 occurrences are covered.
- Actual emitted expressions match26,163,200 table indices and30,428,168 source
  addresses. First/last batches are explicit; interior batches use bounded affine
  equivalence.36 small cases add177,993 checks. The initializer's actual source
  matches all272,000 map entries and7,624 masked tail invocations across actual/
  small cases. Four expression/initializer mutations reject.
- Independent source/SPIR-V review preserves non-B code, memory widths/bindings,
  cooperative operations and barriers. Only the declared u32 map reads are added;
  consumers cannot write the table. Binding4 and the48-byte initializer geometry
  layout reconcile. Old address cursors remain an unchanged qualified premise,
  not a newly independently implemented convolution oracle.
- Observer**fb6d0cc1** links unchanged backend libraries. Its CPU geometry test
  passes. The runtime requests device-local storage, GPU-initializes it, compares
  readback bits with the independent map and waits before exposing it. All
  consumer submissions track a final fence before destruction. All20 BF16 GPU
  contracts now pass their numerical/lifetime checks. No F32/F16 GPU or
  cross-Session cache qualification is claimed.

The first CPU reviewer compared module-local vector-component type IDs11/12;
the repaired semantic comparison retains dtype/width checks and seven regressions.
The next reader failed on rustfmt's wrapped `let values = unsafe`; a separate
whitespace-only reader repair passes. Both original scripts/logs remain; neither
caused a shader rebuild or GPU repeat. Offline review ends20:24:48UTC.

## GPU execution and setup

The first numerical pair passes two guards with zero allocation warnings:
524,288 independent F64 comparisons,67,108,864 full exact output pairs and102,400
exact GPU-initialized map entries. It is retained without repetition.

Its CPU reviewer then stopped because serde stores an OsString as tagged Unix
bytes. The separate path-schema repair validates the tag and bytes, compares the
sealed path and finishes the original F64/plan/output review. Original controller,
finisher and reviewer failures remain. **No GPU fault, recovery or repeated pair.**

The continuation starts**20:32:29UTC**, retaining that pair and running only138
unstarted guards. GPU work finishes**20:52:28UTC**, the aggregate20:52:36 and
independent raw-output/counter/timing review**20:53:01UTC**. All three persistent
services exit successfully; no worker remains.140 guards pass with10 allocation
warnings; none occurs during ordinary timing. No recovery or GPU cohort repeat.

All2,690,432 independent F64 comparisons and1,033,149,184 full bit-exact output
pairs pass, including changed-input/restoration checks.60 initializations and
60 readbacks verify2,126,400 u32 map entries.2,415 source seals and240 retained
output/counter artifact seals pass.11,600 convolution operation calls, zero
learner updates/game actions/backend rebuilds. Minimum sampled Vulkan estimated
headroom is15,827,730,432bytes, not physical free or peak VRAM.

Across all60 isolated processes, initialization GPU time sums.214944ms and
verification copies.763520ms; submission/wait sums4.541950ms. Complete setup wall
time sums376.987159ms, mean6.283119ms, range1.451279–73.661402ms. This includes
allocation, initializer pipeline construction, verification and observer I/O;
the interval ends before destruction of setup-only objects. It is not an
eight-map full-world startup measurement. The immutable table is reused by
ordinary timed calls; initialization is neither charged per convolution nor
silently subtracted from the reported repeated-operation cost.

## Decision and next question

Close this spatial-lookup implementation, without a table/layout/geometry sweep,
favorable subset, retime, integration or full-update trial. Reducing arithmetic
alone has again failed to improve operand delivery; library feeding remains an
open structural problem, not permission to revive rejected kernels.

Next reconcile the existing native host-target/readback dependencies with the
resident JAX update on CPU. The earlier7.528ms uncovered instrumented interval
is not pure CPU cost or an additive saving. Identify actual target computations,
transfers, precision/state dependencies and already-existing device paths before
choosing a resident-target implementation or new measurement. No new GPU
allocation, numerical relaxation, prior-batching timing or learner restart is
implied. Artifacts: `runs/convolution-metadata-kernels-20261016.c0XlqDMn`.
