# Retaining the posterior removes duplicate forward work

October13,2026. **RGB39.392→33.681ms,14.50% less time.** This is an execution
structure improvement on the qualified convolution-training candidate, not
another kernel change. It still costs **2.71× retained JAX12.417ms**. The goal
remains open; no production pin/default changed and Atari remains stopped.

[Declaration](../experiments/2026-10-13-staged-world.md) ·
[compact audit](2026-10-13-staged-world.json).
Artifacts: `runs/staged-world-20261013.djLV1cJT`.

## What changed

The old learner ran a posterior-inference graph, imagined targets, then ran a
world forward/backward graph that recomputed the posterior. The prototype makes
a preparation boundary from requested output dependencies, schedules/fuses each
side separately, and plans activation lifetimes over both. It returns h/z after
the prefix, then consumes late targets and finishes backward/clipping/optimizer.
Early inputs/parameters are immutable until completion; direct shared-buffer
callers retain the same ownership/queue-order contract. Unsupported multi-writer
plans, temporal accumulation and dispatch profiling reject this API.

Kindle uses this only for single-pass BF16 without the optional video encoder.
It retains the same CPU uniform stream, losses, reset rules, detached hard
sampling plus straight-through gradients, and old-weight consumers before
updates. No separate posterior session is allocated on that path. Existing
microbatch/truncated and other-precision paths stay whole-step. In particular,
Mixed uses BF16 inference but F16-input/F32-storage world training: reusing those
activations would change precision, so its fallback deliberately remains intact.

## One complete-update pair per recipe

Same12M/B8/T16/H15 RGB and12M/B32/T16/H15 CDP fixtures/checkpoints as the retained
controls. Each arm has16 warmup+256 measured updates, identical host range
annotations, no profiler/per-dispatch GPU timestamps or build/analysis overlap.
All packing, RNG, readbacks, targets, optimizer and synchronization are timed.

| Ordinary update | Control | Staged candidate |
| --- | ---: | ---: |
| RGB complete |39.392ms|33.681ms|
| RGB posterior + world work |25.695ms|20.189ms|
| RGB imagination/targets |11.065ms|10.913ms|
| RGB behavior/sync |2.632ms|2.578ms|
| CDP complete, unchanged fallback |111.040ms|111.212ms|

The new posterior label is the world prefix, not the old standalone graph;
the combined row avoids comparing mismatched stage boundaries. Prefix/tail are
4.694/15.495ms. CDP's0.15% difference is not a speedup or staged-CDP result.

Nominal RGB matrix/convolution MACs fall95.074→84.756G: **10.319G of duplicate
work removed**. Dispatches fall7,043→6,148, a reduction of895. The remaining
world/imagination/behavior/slow-value matrix and convolution descriptors match
the control exactly. All nominal products remain BF16. Scalar/reduction work,
packing/padding, hardware-issued instructions and utilization are not counted.
The staged posterior plan file aliases the world plan and is counted only once.

## Numerical qualification and limits

- Three CPU dependency/scheduling checks, RGB/CDP graph shape/parameter check,
  formatting and library Clippy pass. Retain the three CPU integration failures
  and snapshots: missing default field, pattern lint, test-only helper scope.
- Eight F32/BF16 × alias/no-alias × Adam/LaProp cases compare ordinary,
  whole-prepared and staged execution through144 updates. Forward values, clipped
  gradients, parameters/moments and counters are exact. Poisoned late inputs do
  not affect the prefix; internal hidden activations survive the boundary.
  Early writes/updates and invalid phase transitions reject. Two further updates
  show that changing late targets changes the tail, not prepared values.
  Publication review catches the scope error: AGC modifies gradient buffers.
  A separate eight-case/24-backward-pass check clears the optimizer, verifies
  unchanged parameters/moments/counters and **exact raw gradients**, under both
  precisions and aliasing modes. Its supplemental source/binary are retained;
  runtime/Kindle binary are unchanged and no timing is rerun.
- RGB initials292/292 and first categorical samples are exact. All215 first
  world tensors and every first model parameter/slow value remain exact.
  Deterministic states differ: t0 relative L2 4.44e-8, full sequence5.15e-4.
  Twenty-six behavior moment tensors differ; aggregate first-state L2.1631
  must not be described as a model-weight error. This is consistent with
  forward compilation rounding amplified by recurrence, not proof of its
  precise lowering cause. The final292 tensors are finite, only one exact,
  relative L2.0871. **No bitwise trajectory or learning-parity claim.**
- CDP24-update smoke and272-update pair retain exact reports, first outputs and
  all346 initial/first/final tensors. Its production path is unchanged.

Nine GPU guards pass, four allocation warnings retained, zero hard faults,
NVML polls, recovery or game actions. There are1,136 full synthetic updates plus
146 small-graph optimizer updates and24 backward-only passes. Minimum sampled
estimated headroom is14.06GiB
for staged RGB and10.51GiB for CDP, not physical-free or peak-memory measurement.
Timing ends20:55:26 UTC; its CPU review ends20:55:33. The supplemental raw-gradient
guard ends21:01:06. No worker remains.

`audit-smoke.json`, `smoke-difference-review.json`, `audit-timing.json` and
`arithmetic.json` preserve the checks. The RGB smoke's final-tensor entry checks
finiteness against itself, not equality with a272-update control. An initial
CPU audit accidentally allowed BLAS oversubscription under one CPU and took76s;
the final audit explicitly uses one BLAS thread and needs3.04 CPU seconds.

## Decision

Keep the staged mechanism as the next connected-execution prototype. This
causally establishes redundant forward computation as a material part of the
gap; it does not explain the remaining21.26ms or qualify Mixed/video/BPTT paths.
The native candidate is still about2.08× the retained JAX control **without**
CUDA command buffers, so command-buffer replay alone is not an adequate answer.
Next attribute the residual on this actual candidate, then connect device
targets/RNG/EMA and remove unnecessary materialization. Do not add historical
speedups, restart Atari or close the larger goal on this isolated gain.

Same backend base056562b with qualified typed dense/grouped/convolution changes,
Bladefe5f5b5/Nagaa4b4183; upstream2214412 reviewed. Exact source/binary hashes,
base revisions and controls are in the JSON. `backend.patch`, `kindle.patch`,
both lockfiles and immutable test binaries are retained in the artifact folder.
`backend-raw.patch`/`backend-raw-tests` add only the final raw-gradient test.
