# Resident return precision

The [original component qualification](../results/2026-10-16-resident-targets.md)
remains failed. Separate retained-data stage analysis finds both GPU/CPU encoders
pass when conditioned on their actual scalar inputs (largest error below1.1e-7).
Return cancellation creates the failing end-to-end coefficients. Casting ideal
imagined/replay returns to F32 before the original encoder passes all unchanged
gates. Rounding the imagined bootstrap to F32 before computing replay returns
also passes, using at most71.62% of its original envelope. These conditional
checks are not a replacement qualification or a tolerance change.

## Selected repair

One compensated F32 recurrence uses two-component local accumulators and explicit
multiply residuals. Keep ordinary F32 output buffers and the existing imagined-
to-replay boundary; no extra storage, loss/normalizer/mask change, F64 GPU feature
or compiler project. This addresses cancellation without repeating the rejected
affine rewrite. The CPU preflight passes8,330,103 values on all12 retained cases
under the original end-to-end gates. It is not a GPU result.

Preserve both original GPU/CPU failures, their sources and the rejected affine
preflight. Implement only the selected recurrence in a new isolated source/cache.
Reuse the original component observer and unchanged backend artifacts. Review
the exact source scope, offline Naga/SPIR-V and CPU tests before declaring GPU
work. No changed input, cherry-picked coefficient or relaxed tolerance. Heavy
preparation/analysis: one CPU,2GiB,no swap,720seconds per service.

This declaration authorizes CPU source/build review only. Any repaired GPU
qualification and complete RGB/CDP comparison is separately declared after
review. No timing, production promotion or learner restart. The broader goal
remains open; qualified native58aff452 is46.199ms versus inclusive JAX27.223ms.

## Repaired component qualification

Offline review passes22:26:22UTC: only `targets_returns.wgsl` changes; unchanged
host bindings/consumer contracts, nine other shader modules, ten validated
Vulkan modules, retained CPU tests and Clippy pass. Nativeb78e9d1d reuses the
exact retained backend. The return module uses F32 only with explicit FMA;
this does not establish how the driver lowers compensation.

Run one guarded process,240seconds,8GiB host memory,no swap,no CPU quota, on the
same four shapes and three initial/changed/restored rounds as the failed cohort.
Twelve target generations, zero learner updates/actions. Require identical
fixture inputs/configs and unchanged original F64 gates; retain every output,
source and failure. Check means canaries, exact consumers, state carry/restore,
all world copies and finite report/status. The CPU oracle/review runs separately
with one CPU,2GiB,no swap,720seconds. No automatic repeat or full-model timing.

Require device0x2c02,driver580.178.04,boot3e89d55c-a9e5-472f-a18a-06508c5bafa7
and >=2GiB sampled Vulkan budget headroom. Record allocation warnings; API,
numerical,hard-fault/deadline failures stop for review. No NVML polling or GPU
recovery. The earlier failed qualification remains failed whatever this repair
does. Full12M smoke/checkpoint and complete RGB/CDP timing remain undeclared.
