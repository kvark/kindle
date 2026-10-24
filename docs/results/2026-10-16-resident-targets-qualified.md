# Resident targets cut complete updates by about6%

October16,2026. [Evidence](2026-10-16-resident-targets-qualified.json) ·
[return repair](../experiments/2026-10-16-resident-return-precision.md) ·
[full-model scope](../experiments/2026-10-16-resident-target-model.md) ·
[encoding-contract correction](../experiments/2026-10-16-resident-target-observer-contract.md) ·
[timing declaration](../experiments/2026-10-16-resident-target-timing.md).

**Complete RGB updates fall5.79%, CDP6.09%. RGB is now1.59× inclusive JAX.**
Keep isolated nativee45b9c43 in `runs/resident-target-model-20261016.tEVYDi5C`;
production and Atari remain unchanged. This is scoped numerical/integration/
performance qualification, not bitwise trajectory or learning parity. The
original exact-slow-encoding smoke remains failed; its corrected contract is
explicit below. The broader root-cause goal remains open.

| Complete B32/T16/H15/12M update | Fresh CPU-target control | Resident targets | Reduction |
|---|---:|---:|---:|
|RGB|45.852ms|43.197ms|5.79%|
|CDP|43.657ms|40.997ms|6.09%|

RGB/JAX is43.196507/27.222569ms. CDP is a different workload. Fresh controls
are.75%/.64% below retained46.199/43.939ms, within the declared5% calibration
band. All256 measured samples per arm are retained, after16 warmups. Candidate
first/last32 means are43.199/43.067ms RGB and40.874/41.000ms CDP. No retiming or
sample trimming. Target work changes stage boundaries; complete update is the
primary result, not an additive interpretation of individual stage differences.

## What changed

Imagined/replay returns and weights, percentile/EMA, weighted actions, dense
two-hot targets, replay-feature handoff and world late-target writes stay on GPU.
Existing means, losses, optimizer, precision, masks, CDP arrival bonuses and RNG
remain. Return accumulation keeps a local F32 residual; output buffers remain
ordinary F32. No backend/compiler change or extra return buffer is required.

Select `KINDLE_DEVICE_TARGETS=1` alongside device means, copy composition, word
staging, image views and cached recording. The flag is presence-based: leave it
unset for the control. Keep original58aff452 as the preceding qualified control.
The new path supports at most8192 returns and staged full-BPTT/full-microbatch
recipes. Its598,892 owned bytes and81 device-copy regions/6.486MB are logical
extents, not measured traffic. The existing posterior context readback and
production replay-context refresh remain; this is not a fully resident learner.

## Numerical and integration evidence

- The [earlier component failure](2026-10-16-resident-targets.md) stays failed,
  including original CPU failures under the same F64 envelope and the rejected
  affine rewrite. Retained-data analysis separates recurrence cancellation from
  encoding error. One compensated recurrence passes8.330M CPU-preflight values.
- The repaired GPU component passes all19,652,262 original comparisons, including
  zeros/world copies, on identical144 input arrays. All759 output arrays and301
  restored files pass;12 generations, no learner updates. Local GPU arithmetic
  is not bit-identical to the CPU compensated model; no lowering proof is claimed.
- Actual12M RGB/CDP first targets/metrics pass their original numerical envelopes.
  Copied posterior/replay masks, complete programs/pipelines/groups/configs and
  initial tensors match. Fresh controls reproduce all272 retained metric reports
  and initial/first/final292 RGB/346 CDP tensors exactly. Each timing arm reproduces
  its own24-report smoke prefix, initial/first tensors and first targets exactly.
- Eight actual checkpoint save/drop/restore pairs preserve every arm's complete
  model/moment/slow tensors, normalizer and counters exactly. This does not restore
  replay/RNG/live belief as an uninterrupted lifetime.

The original full-model audit incorrectly demanded exact CPU/GPU **slow** encoding
despite the component contract permitting F32 encoding rounding. It fails5,219
CDP imagined-slow coefficients; the full retained audit finds310 replay-slow
differences too, maximum absolute5.96e-8 and2/1 ULPs. RGB slow targets are exact.
The revised smoke explicitly applies the original1e-5+2e-5-relative dense-target
envelope to all two-hot targets; other exactness checks remain. Fifteen positive/
negative CPU tests pass. This revises an exactness requirement, not a numerical
tolerance, runtime implementation or GPU cohort. Do not erase the failed screen.

Cross-arm trajectories differ. Separating learner parameters from optimizer
moments: RGB's101 learner tensors are exact after one update and differ by
.00025595 relativeL2 after272; CDP's119 differ by2.58e-7 after one and.054343
after272. Slow-critic and moment differences are separately recorded in JSON.
Finite short-run states and target correctness do not establish equal learning.

## Accounting and remaining work

Nine new GPU guards, six allocation warnings,12 component generations and1,184
synthetic updates including warmups/smokes; zero game actions. Last GPU observation
22:45:15UTC; final independent CPU audit22:49:19UTC. No GPU worker remains.
Preserve the provisional incorrect RGB/one-ULP descriptions and correction, the
exactness failure, the common timing observer's stale96-update metadata field
(actual1,088 independently verified), and the CPU auditor's file-descriptor
exhaustion. Releasing each mapped tensor pair fixes that observer without raising
limits, changing results or repeating GPU work.

This removes about2.66ms per update, not the remaining15.97ms RGB/JAX difference.
Convolution data feeding remains the largest identified structural issue; existing
native/JAX counters and rejected families remain authoritative. No rejected kernel
revival, geometry/precision sweep, prior-batching timing or automatic promotion.
Continue the root-cause work with the qualified candidate and existing counter/
dependency evidence; require a distinct mechanism and CPU proof before any new
GPU allocation. Keep learning-quality validation separate and Atari stopped.
