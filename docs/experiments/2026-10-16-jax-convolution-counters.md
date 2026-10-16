# Complete actual JAX convolution-family counters

October16,2026. The full-update trace still places39.650ms inside native GPU
passes; the qualified ordinary comparison stays46.199/27.223ms,1.70x. Actual
library instruction observations cover only two encoder derivatives. Keep
58aff452, production and stopped Atari unchanged; this is not a kernel trial.

First, CPU-prove all23 current native convolution products and weight/gradient
paths against the retained JAX convolution contract. Reconcile all20 existing
native counter contracts/multiplicities/source versions without repeating GPU
work. Their main-kernel counters are current; standalone helper ranges are not
automatically current connected-image-view helpers.

From all eight retained actual JAX updates, use CUDA correlations and innermost
CPU HLO thunks to prove the complete23-product scope, including all18 helpers:
41 launches/update. Establish an exact-name selector with no out-of-family
matches, identical name/geometry/resources and complete membership in every
frame. Main/helper identity classes must remain disjoint. Retain any ambiguous
individual layer ownership; do not infer it from output order alone. Verify
full semantic compiled work against the current JAX control. Review CPU failure
before any follow-up; no GPU work without a complete coverage receipt.

If that receipt passes, one unchanged full-JAX benchmark executes272 synthetic
updates with its already qualified profiler API interval around update16.
Nsight Compute uses the existing seven sections and16 native-comparable
instruction/shared/local counters, kernel replay, graph-node mode, no clock/cache
control or kill. Keep all41 profiles, replay counts, opcode distributions,
numerical reports and tails. Exact272 reports/288 initial-final tensor fingerprints,
full HLO and the complete selected identity multiset must match. Capture timings
are not ordinary performance; no new native GPU calls or backend build.

Compare all23 main-product counter aggregates first. Report the18 actual JAX
helper profiles separately; do not compare stale standalone native transforms
as though they were today's connected graph. Preserve output precision/layout,
padding, compensation, workspace and synthetic-native-versus-live-JAX input
differences. Ratios are instruction work, not speedup or an additive gap budget.
Any role/subset report needs unambiguous membership; no favorable omission.

CPU preparation/review1CPU,2GiB,no swap,<=720s each. One GPU guard<=600s, service
<=720s,12GiB,no swap/no CPU quota,Restart=no,KillMode=control-group. Expected5080,
driver580.178.04/current recorded boot,>=2GiB sampled Vulkan budget headroom.
Record allocation warnings; API/numerical/hard faults/deadlines stop for review.
No NVTX-filter retry, separate NVML polling, driver recovery, geometry/staging/
precision sweep, prior-candidate timing, promotion or Atari allocation.

Upstream12df95c/4845d652 rechecked unchanged. Artifacts:
`runs/jax-convolution-counters-20261016.1F8DAYTk`.
