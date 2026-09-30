# Bounded small Dreamer replication

Declared September 30, after the [numerical prerequisite](../results/2026-09-30-small-dreamer-replication.md)
and before any new gameplay. The user caps this round at **10 attempts total**,
upstream and native combined; failures and interruptions count. Target three
paired seeds (six runs); at most four pilot/debug attempts. Reuse unchanged
qualifying pilots. No automatic JEPA matrix follows this allocation.

## First pilot pair

- **Seaquest**, learner seed1009. This title is absent from Tiny's video corpus
  and already has held-out representation probes. Later seeds, if justified:
  2017 and3019. Upstream first, native second, serially; inspect each guard/result
  before proceeding. A failed run has no automatic retry/successor.
- Both use Size1M, F32, learned RGB64 encoder/convolutional decoder, **N8/B8/T16/
  full BPTT/M8/H15/R32**. Shared parameters, optimizer defaults, warmup1000,
  capacity100,000 arrivals, context1. Upstream's eligible-start capacity is99,872.
- **200,000 aggregate actual actions** per run (25,000 per stream). Resets are
  not actions or training credit. The learner drains the same schedule: R32 /
  (B8×T16) = one update per four aggregate actions after readiness. Sequence
  starts, recurrent states and resets remain stream-local.
- Full18 actions, sticky.25, repeat4, zero reset no-ops, 100,000 executed-frame
  episode cutoff. Environment seeds are learner seed + stream×1,000,003 modulo
  2³². Cutoffs end the replay episode but do not imply environment termination.
- One antialiased RGB64 resize: native GPU implementation versus upstream
  Pillow. No JEPA, pretraining, intrinsic/shaped reward, privileged observation,
  action override or restore. No 64→224 upscale.
- Pinned upstream `e3f02248`, native Meganeura `22c31b94` / Blade `7cca6377`.
  Initializer distributions match, not random draws or replay samples. The
  upstream policy synchronization behavior remains unchanged.
- RTX5080/driver580.178.04, one heavy GPU process, two CPU-equivalents and12GiB
  host-memory ceiling per service, no swap. JAX pool fraction.5; this reservation
  is not measured live-array/peak memory. Require≥2GiB sampled Vulkan estimated
  headroom. No application NVML polling; utilization is unmeasured.
- **One-hour hard deadline per run; target under one hour for the pair.**
  Initial/JIT compilation, collection, learning and final save are reported;
  model construction is separately reported too. No local compilation during
  timed execution. Store live logs/guards locally.

## Decision and reporting

Review complete score-vs-actions/time curves, initial/final online windows,
every completed episode, unfinished tails, counters, finite state and guard
results. A pilot's score rise is provisional evidence, not a seed-level
significance claim or frozen competence. Upstream must show useful learning
at this capacity/budget, not merely finish quickly, before filling the three-seed
comparison. If learning is weak or cost excessive, diagnose/revise within the
remaining ten attempts instead of blindly launching more identical runs.

After three seeds, report per-seed curves and learner-seed bootstrap uncertainty
using the existing curve format. Published Dreamer scores remain reference
evidence under their own protocols, not expected scores for this smaller recipe.
Lower replay ratio/BPTT/capacity is a changed recipe, **not** an optimization
speedup over the cancelled 12M study. This pilot cannot settle JEPA's value.

Execution/artifacts: `runs/small-dreamer-replication-20260930.vRE7ag/learning-*`.
Current done/running/next and attempt count stay in [PR31](https://github.com/kvark/kindle/pull/31).

## Reviewed throughput replacement (September 30)

Attempt1's upstream control completes in8m37s. Attempt2's native control is
retained as an operator interruption at90,264 actions; its51.31 actions/s
projected beyond the unchanged one-hour deadline. It still consumes an attempt.
The replacement starts fresh at seed1009 with **identical learning settings**,
not replay/checkpoint continuation. Upstream attempt1 remains the paired control.

Meganeura `75d08173` uses existing split-reduction kernels for low-parallelism
convolution gradients, with512-position partitions and64MiB logical partial
budget. Independent F64 and full-update checks pass; matched synthetic update
medians improve72.62→30.58ms. The summation order changes, not model capacity,
pixel processing, optimizer, loss or replay ratio. The fresh gameplay pilot
must verify actual throughput and learning before more seeds are scheduled.
[Evidence and retained diagnostic failures](../results/2026-09-30-small-rgb-profile.md).

**Pilot outcome:** the fresh native run completes 200,000 actions / 49,939 updates
in 26m10s, at 127.42 actions/s. Its online first20/last50 means rise from 83.0 to
328.8; all completion audits pass. The successful upstream/native pair takes
35m31s including construction, meeting the sub-hour target. The interrupted
29 minutes remain extra compute and a counted attempt: **3/10 used**. This
qualifies the recipe for additional seeds, not frozen competence or a native
performance advantage. [Curves and audits](../results/2026-09-30-small-replication-learning.md).
