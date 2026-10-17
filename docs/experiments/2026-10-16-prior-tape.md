# Diagnose the retained prior-reference discrepancy

October16,2026. The [prior-batching qualification](../results/2026-10-16-prior-batching.md)
remains failed. CPU shader-ordered F32 normalization/SiLU still leaves340 final
logit envelope failures in the second12M fixture and twelve failed array checks
across both arms. Correctly rounded CPU intrinsics are not bit-exact Vulkan
intrinsics. Do not tune a reference against final outputs or relax its gates.
CPU nullable-epilogue and Tiny packed-row reader failures/repairs remain in
`runs/prior-rounding-20261016.Zct8P2SA`; no GPU call was made there.

One new diagnosis-only process loads the retained12M batched execution plan
and parameters, using the unchanged qualified backend libraries. Verify the
complete plan/pipeline identity before executing. Capture the three prior
products, bias additions, two normalization results and BF16 activations at
their actual lifetimes, by ending submissions at existing barrier groups.
Use the two already retained input variants; stop after final logits, without
running the complete backward/optimizer. Require exact saved final logits and
unchanged parameters. Preserve original storage, kernels, rounding and groups;
do not use debug/no-alias mode, change output registration or recompile kernels
to expose different intermediates. Account for every executed dispatch.

This is additional forward-tape evidence, not a replacement numerical cohort,
qualification retry or timing experiment. Independently check each recorded
matrix/normalization/activation boundary and propagate the observed finite-
precision tape for diagnosis. Report unexplained values; the original aggregate
F64 failure remains until its cause is established. No whole/staged-world check,
full-update timing, production promotion or Atari restart is authorized here.

CPU preparation/review: one CPU,2GiB/no swap, bounded persistent services.
Compile only the small standalone reader against sealed existing Rust libraries;
do not rebuild unchanged Kindle/Meganeura/Blade. GPU process: existing host guard,
180-second deadline/240-second service, no CPU quota, expected5080/580.178.04,
>=2GiB sampled estimated Vulkan headroom. Record allocation warnings; API/hard
faults, identity/numerical mismatches and deadlines stop for review. No driver/
clock changes or separate NVML polling. Upstream f648a453/4845d652 is unchanged.
Artifacts: `runs/prior-tape-20261016.bFYniS7c`. Keep834241b6 and the1.77× gap.
