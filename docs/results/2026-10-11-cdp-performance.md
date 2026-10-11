# Performance pivot: cheaper exploration and wider learning

**The selected B32/F16 candidate processes replay positions2.66× faster than
the old B8 recipe.** This is measured synthetic-update throughput, not yet
faster learning to a score. The [three-seed paired Freeway screen](../experiments/2026-10-11-cdp-performance-learning.md)
is running to test that remaining question; the stopped Pong queue stays cancelled.

## Implemented and measured

|12M resident workload | Full update | ms/replay position |
| --- | ---: | ---: |
| Original CDP, B8/Auto |107.92ms |.8431 |
| Reuse exploration state projections, B8/Auto |86.02ms |.6720 |
| Also project only the selected action effect, B8/Auto |72.33ms |.5651 |
| Also widen convolution splitting, B8/Auto |65.40ms |.5109 |
| Wider grouped RSSM, B32/F16 |161.72ms |.3159 |
| Latest Meganeura, same B32/F16 |162.00ms |.3164 |

All timings include posterior inference, imagination/targets, world/behavior
training, synchronization and slow-critic updates:16 warmup +256 measured
updates. Synthetic input generation, replay/environment and construction are
outside the timer. B32 processes512 positions rather than128; latency per
update grows, but at replay ratio32 it updates every16 rather than4 actions.
T16, capacity, four exploration heads and reward scale are unchanged.

- **Exploration:** share the state part of the first linear projection across
  actions. Center the last hidden layer over all18 actions, then apply the
  final linear projection only to the selected effect; its bias cancels.
  This preserves the all-action baseline for categorical/zero-reset actions,
  while removing about88% of the ensemble's dense multiply-adds. No action
  subset, disabled bonus, privileged hint or new target is introduced.
- **Batching:** remove the accidental128-row ceiling on grouped RSSM products.
  B16/B32 had fallen back to serial sliced graphs. Imagination dispatches now
  stay1,229 at B32 instead of3,113 in the first candidate. B8/B16/B32 were
  measured, not an exhaustive batch-size search or a proof of optimality.
- **Convolution gradients:** use the existing split-reduction implementation
  below256 workgroups, chunk4096,64MiB scratch cap. The old48-group cutoff
  missed expensive gradients. First-step split-versus-unsplit checks cover154 raw
  parameter gradients and638 saved state tensors across CDP/RGB; maximum
  relative gradient errors are7.44e−6 and1.02e−5.
- **Precision:** checkpointed `--training-precision auto|f32|f16`. F16 allows
  half-input products in training while retaining F32 storage, weights,
  accumulators, reductions and optimizer. It is not BF16/full mixed storage.
  F16 alone saves only about1–2% here; most gains come from less work and
  batching. Default Auto remains until the learning comparison is reviewed.
- **Upstream:** update2c42b448→430f14b, then to latestd7a7293 when five more
  fixes land during this session; Blade56f0565 stays unchanged. Historical
  measurements keep their own revisions. The latest refresh adds no material
  speedup on this fixture; it is not credited for the algorithmic savings.

## Native versus JAX: there is still substantial room

Matched-shape RGB,12M/B8/T16/H15, no exploration: native Auto initially
**101.00ms**, optimized native F16 **71.37ms**, upstream JAX float32
**24.36ms**, upstream JAX BF16 **12.40ms**. JAX BF16 B32 is26.08ms.
Use these as framework throughput comparisons, not numerical/learning parity:
weights, RNG, recurrent initialization, graph decomposition and precision
paths differ. JAX uses highest matmul precision, completes device work and
consumes pending policy synchronization each update. The initial capture
without repeated policy synchronization is also retained.

Native and [JAX traces](https://docs.jax.dev/en/latest/profiling.html) are
captured separately from ordinary timings. The original native RGB weight
gradients sum to35.75ms of instrumented dispatch time; one decoder gradient
takes19.12ms versus about.34ms for the corresponding BF16 JAX operation.
This led to the split-reduction change above. Meganeura still has
[no cooperative weight-gradient convolution path](https://github.com/kvark/meganeura/blob/d7a7293378db1b0e8ee3b94a1e6d73ef14a1a32c/src/kernels/conv.rs):
that and reduced-precision convolution coverage are concrete backend work,
not reasons to replace the Rust learner. Final optimized profiles are retained
alongside the original captures so priorities can use the remaining costs.
After splitting, RGB weight gradients still sum to13.48ms instrumented, versus
about1.48ms in the BF16 JAX trace. At CDP B32, the largest remaining dispatch is
a12.29ms split weight gradient (512 images,32→48 channels,32×32,5×5).

Pointwise/layout traffic and recurrent launch/fusion cost also remain. Simply
allowing half operands does not halve F32 activation traffic. Per-dispatch
instrumentation strongly perturbs some native stages; GPU activity spans are
not SM utilization. **SM utilization remains unmeasured**, with no separate
NVML polling or CPU learner workaround.

## Validation, scope and next decision

GPU numerical checks cover state-projection reuse, selected/zero-reset action
effects, grouped products through512 starts, gradients, finite full updates
and real4096-action train/1024-action frozen restores at both B8 and B32.
The first real preflight measures60.88 versus105.15 aggregate actions/s;
its longer B32 prefill and fractional final update debt are disclosed, so it
is not a steady-state speed or learning-quality claim. Frozen tensors remain
exact and initial tensors match across batch sizes.

Formatting, strict workspace/Python Clippy,118 CPU Rust tests and1,223 Python
tests pass. All50 GPU guards pass with10 recorded warnings. The latest refresh
retains exact first outputs and984/984 final tensors across the three272-update
fixtures; this is scoped agreement, not universal parity. Early preparation
failures, the original and corrected source
archive, and a log-scan-overlapped timing are retained; isolated measurements
replace the latter for speed claims. No GPU fault, recovery or tolerance
relaxation after a failed GPU check occurred in this study.

Cleanup removed regenerable Rust debug artifacts, freeing approximately63GiB
of filesystem space at that point; rebuilding restores them. No unique run,
checkpoint, video, user `TASK.md` or feedback file was deleted. New measurements
consume additional space, so this is not a current-free-space claim.

The paired learning screen compares optimized B8/Auto against B32/F16,200k
actions each across three seeds, with natural final/initial frozen cohorts,
whole videos and score-versus-actions/time curves. Budget/precision changes
are a package, not isolated causal effects. No Atari-wide promotion, concurrent
seeds or old long-run extension follows from these throughput results.

[Compact receipt](2026-10-11-cdp-performance.json) ·
[Raw artifacts](../../runs/cdp-performance-20261011.7SpyzGUa) ·
[PR31 dashboard](https://github.com/kvark/kindle/pull/31).
