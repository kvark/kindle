# Joint Tiny: attention backward repaired

The independent native check found a real compiler defect, not an encoder
capacity or learning-stability result. Tiny forward features matched, but its
last attention block's value gradients were exactly zero while Q/K matched.
138 of 148 parameter-gradient comparisons failed; the patch projection's
relative L2 error was 0.60438.

Meganeura allocated a value-gradient buffer, aliased reshape views to it, then
remapped only the gradient node to a new fused dK/dV output. The views still
read the unwritten old buffer. A two-attention CPU regression reproduces the
wrong buffer mapping. [Meganeura PR223](https://github.com/kvark/meganeura/pull/223)
reserves the destination before creating views; no shader or learning setting
changes. The producer-order check and missing-dV scratch destination remain.
Kindle pins `13b19d33`, based on current upstream `6268ea5`; Blade stays
`e349cddf`. The user's adjacent backend worktrees are untouched.

## Evidence

- All 47 compiler tests and strict library/oracle Clippy pass.
- Final pinned Tiny: scalar loss, two frame outputs and **all 148 gradients**
  match the independent PyTorch/F64 oracle. Maximum relative L2 error across
  these 151 comparisons is **1.111e-6**, without relaxed tolerances.
- One Adam step moves the input projection and stays finite. This is **not**
  independent optimizer-value parity or a task-learning result.
- The variance/covariance regularizer matches its separate F64 value/gradient
  reference. Acting-cache refresh matches fresh encoding after a weight change,
  including unequal stream arrivals, episode resets and chunk rollover.
- These three explicitly selected checks pass under the host guard in **13.10s**,
  with no new allocation warning. The standalone small attention regression
  passes GPU/F64 and finite-difference checks in **4.16s**; it records one exact
  startup allocation warning within its declared limit. Both seals verify.
- Four synthetic native RGB updates on merged6268ea5 now pass **1,524** new
  upstream value/raw-gradient/common-gradient optimizer/EMA comparisons. The
  attention-only13b19d3 patch does not change this RGB path. The guard/seal pass
  with no new warning; this does not independently compare Tiny optimizer values.
- 103 workspace CPU tests,1,004 Python tests, formatting and strict workspace/
  Python-binding Clippy pass. The first Python invocation failed collection
  because the selected venv lacked its declared Pillow/safetensors test extras;
  installing those extras resolves all17 import errors. Backend CI856 passes.

This affects the new native **joint-encoder backward path**. Historical frozen
Tiny inference and its learning comparisons did not use that path; do not
invalidate or relabel them as joint-Tiny results.

## Retained failures and limits

The first numerical check and expanded all-gradient diagnostic both failed and
remain intact. The expanded diagnostic also recorded one exact startup warning.
An initial local fix passed the two Tiny checks in9.42s before the final
producer-order assertion cleanup; the final pinned run above supersedes it.

I mistakenly launched the broad backend library suite as CPU-only. It contains
unignored GPU tests and ran outside the host guard. It completed325 tests with5
ignored, but recorded another startup allocation warning and
`VUID-vkDestroyDevice-device-05137` cleanup errors in the pipeline-construction
test, which creates raw pipelines without their normal session-owned cleanup.
It is **not accepted as clean GPU qualification**. The service had already
exited when a stop was requested; no process remains, no Xid/hang/device loss
or recovery is recorded. The journal is retained. Subsequent native checks use
explicit test selectors and the guard; no exception was added for cleanup VUIDs.

All GPU work uses RTX5080/580.178.04. Tiny checks require at least2GiB sampled
Vulkan estimated headroom; this is neither physical free nor peak memory.
The known shader-layout VUID remains logged and non-blocking by user direction.
Startup-warning exceptions remain exact-message, bounded and diagnostic-only.
No NVML polling, recovery or learning campaign is implied.

Three complete Size1M/N8/B8/T16/H15/microbatch1 synthetic updates completed with
finite losses and encoder movement, and saved a checkpoint. Their debug-build
mean is3.466s/update (2.964s in world training); construction took71.35s under
one CPU. These are **not optimized-build throughput estimates**, and their empty
encoder-chunk boundary omits live-prefix refresh cost. The combined120s guard
stopped during the second construction for restore, with no new warning/fault.
The failed guard and complete updates/checkpoint are retained. A separately
guarded restore-only check then passes all148 encoder tensors exactly and frozen
acting with zero updates in78.88s, with no new warning (nine-file seal valid).

## Optimized full-update cost

Both arms now pass a separately guarded optimized-build probe with272 synthetic
actions, three complete updates, a live three-frame encoder prefix and checkpoint
save. Recipe: Size1M/N8/B8/T16/H15/R32, microbatch1, capacity8,192 (only280
entries populated). Timing includes replay, posterior, world/behavior training,
sync and live-cache refresh, with GPU timestamps enabled in both arms.

| Mode | Mean full update | World training | Construction | Sampled used memory after updates |
| --- | ---: | ---: | ---: | ---: |
| Frozen | .487s | .244s | 9.35s | 4.59GiB |
| Joint | 2.211s | 1.942s | 12.96s | 7.10GiB |

Joint costs **4.54x** as much per update here. Frozen encoder weights remain
unchanged; joint weights move and all reported losses stay finite. Live-cache
refresh adds about28ms/update in the joint probe. Both guards/seals pass without
new allocation warnings. These three-update synthetic timings are not sustained
game throughput, peak/full-replay memory, or a learning advantage.

The ordinary optimizer-free world pass (one microbatch) measures242.04ms wall
versus210.63ms of GPU pass timestamps (87.0%). This is **not device utilization**.
Windowed per-dispatch profiling identifies attention backward as70.4% of its
instrumented kernel-time sum, but increases wall time12.05x and changes execution
timing; use it for localization, not additive production-cost estimates. Frozen
world-pass timing is30.18ms wall/2.01ms GPU, showing substantial host/dispatch cost
in that arm. Parameters/optimizer state are preserved by post-update profiling.

Next: the declared three paired8,192-action early-learning/collapse screen
(roughly4–5GPU hours); current-backend upstream optimizer comparisons now pass.
This avoids treating a multi-day200k-action replication as a development test.
Task/value/world losses reach Tiny; the Dreamer actor loss remains separate.
No gameplay result or noncollapse-over-training claim exists yet.

[Compact result](2026-10-03-joint-tiny-backward.json) ·
[Protocol](../experiments/2026-10-02-joint-tiny.md) ·
[Local evidence](../../runs/joint-tiny-native-20261003.06z9Gp)
