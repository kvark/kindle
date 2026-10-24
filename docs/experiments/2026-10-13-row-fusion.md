# General typed row fusion

The staged RGB candidate remains33.776ms versus retained JAX12.417ms. The
declared residual study isolated5.021ms of zero-work dependency-barrier cost.
Test a mechanism that reduces real materialization/dependency boundaries:
general reduction prologues/epilogues absorb single-use pointwise operations,
including broadcast bias and explicit BF16 storage. No Dreamer-specific kernel,
matrix geometry change, real-work barrier elision or precision-policy change.

Artifacts `runs/row-fusion-20261013.vif6rSDw`. Start from qualified
staged-world source/binary; backend056562b+qualified patches, effective
Bladefe5f5b5/Nagaa4b4183. Upstream2214412/fe5f5b5 reviewed unchanged.
Keep production pins, neighboring user work and game allocations unchanged.

Preserve explicit rounding boundaries, shared/saved intermediate outputs and
backward lifetimes. New fusion must not traverse a narrowed intermediate by
silently replacing it with F32. Existing generic bindings and reduction trees
remain; scalar storage types enter pipeline identity. No tolerance changes.

CPU shader/plan checks cover fused arities, broadcast addressing, storage types,
protected outputs, fan-out, explicit narrowing and malformed combinations.
Independent existing F64 reference gates cover inference and raw gradients
under fused/unfused builds, strict/BF16 storage and representative narrow/wide
rows. Require the expected5080 and≥2GiB sampled estimated headroom. Retain any
failed source/binary before a reviewed correction, with no blind retry.

After numerical review, one RGB and Mixed-CDP24-update smoke each, then one
ordinary paired272-update control/candidate comparison per workload. Keep
initial weights/configuration, actual batches, all reports/first outputs/saved
states and dispatch/group inventories. Exactness where observed; disclose
rounding/trajectory differences rather than asserting stochastic parity.
Judge complete updates, not removed-dispatch counts alone. No timing sweep,
production promotion, learning claim or performance-goal completion.

GPU work stays serialized in bounded guarded systemd services, no CPU quota,
allocation warnings retained, no NVML polling/recovery. CPU preparation/tests/
analysis use one CPU,2GiB/no swap and one BLAS thread; no compilation or heavy
analysis during matched timings. Review every failure before follow-up.

First qualification stops before smokes: the128×2048/F32-input/BF16-output
case has three post-rounding F64 mismatches, worst1.03125 versus1.0390625.
Retain the failed binary/source and unchanged tolerances. One separately
bounded diagnosis compares fused and all-dispatch-fusion-disabled executions,
also reading their F32 pre-rounding values. Check the existing F64 gate before
rounding and independent CPU RNE bit patterns afterward; save both raw arrays
and every mismatched location. No optimizer updates or full-learner timing.
This diagnostic does not turn the original failed gate into a pass.

Diagnosis passes: fused/unfused have identical three failures, identical full
F32/rounded arrays, zero RNE errors and passing F64 pre-round values. Example:
F64 activation1.035156471 rounds above the BF16 tie; both F32 paths produce the
exact tie1.03515625 and correctly round down to even. The next separately
declared qualification keeps the original test unchanged and checks quantized
outputs compositionally: existing F64 gate before the output conversion, then
every element bit-exact against CPU BF16 RNE. Full loss/raw-gradient F64 gates
remain unchanged, including BF16-output training. Same84 cases, fused/unfused,
no tolerance-value adjustment or unexplained-failure suppression. A failure
still stops follow-up. This replaces the unsuitable post-quantization output
criterion, not the originally retained failure record.

The second check stops on one raw weight-gradient at column949 of the same
wide case (8.51e-8 error,3.63e-8 allowance). Before any timing, a bounded
diagnosis runs fused/unfused original checks and an independent analytic
RMSNorm/SiLU/MSE backward calculation. It conditions the loss derivative on
the separately F64-before-rounding/bit-exact-RNE-validated stored activation,
while retaining the original tolerance and original reference error scales.
CPU tests compare this analytic chain with the existing F64 autodiff chain.
Retain both original failed gates; a conditional derivative is not an
unconditional ideal-F64 trajectory comparison. No runtime or precision-policy
change. Full-model smokes/timing remain unstarted pending this review.

The bounded diagnosis passes: fused/unfused loss and both complete gradient
arrays are bit-identical and both fail the original same weight entry.
Independent conditional F64 gives.000168939660761 versus GPU.000168939659488;
all original error scales/tolerances pass. The analytic chain also passes60
CPU comparisons with F64 autodiff. Complete the original84-case coverage with
this explicit conditional check for BF16-output backward, unchanged original
F64 backward for F32 outputs, and compositional output checks. Retain original
BF16-output F64 reports as diagnostics; do not relabel their failures. Require
all checks before starting the already declared two full-model smokes/timing.
