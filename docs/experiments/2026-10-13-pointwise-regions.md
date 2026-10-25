# Preserve shared values inside fused pointwise regions

The current isolated RGB learner takes32.103ms versus JAX's complete12.836ms
caller boundary. Real dependency chains establish a~2.05µs Vulkan/ordinary-CUDA
floor versus~.65µs CUDA graphs. Widening scalar inputs alone removed346
dispatches but only62 dependency groups. The next mechanism is general
multi-output fusion: compute shared intermediates once and retain every needed
output, instead of ending a kernel merely because an intermediate has fan-out.

Artifacts `runs/pointwise-regions-20261013.vz7sFGG9`. Start clean from the
qualified wider-binding/staged-world source snapshots, not the subsequent
diagnostic barrier modifications. Production pins/defaults and neighboring
user work stay unchanged. Upstream2214412/fe5f5b5 was rechecked unchanged.
No Atari allocation, CUDA learner, unsafe barrier removal or geometry sweep.

Keep eight storage bindings total, materialization barriers, externally visible
and backward-saved outputs, explicit BF16 round-to-nearest-even boundaries,
old-weight ordering, memory lifetime and precision policy. A region may move
a producer only when no intervening reader/writer changes its semantics.
Single-output kernels and matrix/convolution geometry remain unchanged except
where the same general scalar region eliminates a dependency. Any scalar
backward expression exposed to this pass must keep its original operation
order and receive independent value/gradient checks.

CPU tests cover shader validation, fan-out, duplicated operands, protected
outputs, alias/write hazards, explicit materialization, BF16 boundaries and
the binding cap. GPU tests independently check F64 values/raw gradients,
exact stored BF16 rounding and staged forward/backward lifetimes. Keep the
existing tolerances; no trajectory-equality assumption for newly fused F32
expressions. Retain every failed source, guard and receipt before corrections.

After numerical review, two24-update RGB/CDP smokes, then one ordinary
272-update wider-control/candidate pair per workload. Same batches, configs,
initial checkpoints and actual precision; audit initial/first/final states,
all metrics, matrix/conv descriptors, dispatches and dependency groups. Report
arithmetic drift separately from speed; no learning-equivalence or additive
speedup claim. Stop and inspect a failed check before any follow-up.

Serialize all GPU jobs through the existing guard in bounded systemd user
services, Restart=no/KillMode=control-group, expected5080 and≥2GiB sampled
Vulkan budget headroom. Record allocation warnings; no NVML polling/recovery.
CPU builds/analysis use one CPU,2GiB/no swap, never during matched timings;
GPU workers have no CPU quota. Full root-cause/performance goal stays active.

CPU preflight retained two Clippy failures, then a structural assertion that
the BF16-input fan-out was not fused. Source review finds that the existing
conversion-folding pass marks typed kernels with the same barrier flag as
explicit materialization. The revised pass follows the existing typed-row
rule: explicit materializations remain blocked; typed edges become explicit
region input/output types and internal RNE operations. Keep the failed source
and binaries. Public outputs also remain fusion boundaries, because prepared
posterior outputs must not acquire a consumer's late-target dependencies.

Numerical qualification and both24-update smokes pass. The stricter physical
matrix-descriptor audit fails on CDP world training: four same-shape products
are horizontally grouped3+1 in the wider control and2+2 after region fusion.
Retain that failed audit and the two removed/two added descriptors. Before
timing, expand only verified horizontal batches (matching z extent, inputs
and outputs) and require identical per-product geometry, precision, epilogue/
prologue and multiplicity. This is a scheduling consequence of changed
dependencies, not a tile search or a claim of identical physical dispatches.
RGB's24 reports/all saved states are exact; CDP's first-state difference is
one tensor at≤2.98e-8, with later drift. No trajectory-equivalence claim.

The timing controller stops after three successful272-update GPU jobs because
its inherited CDP receipt expectation mistakenly says8 measured updates,
although the declared/configured run produces256 plus16 warmup as intended.
All three outputs/guards remain; no job is repeated. Review those receipts,
then run only the unstarted CDP control with the corrected256 expectation in
a new queue. The failed controller/specification stays part of the evidence.
