# Precision/dataflow attribution — October14

The complete isolated native22.881ms/JAX12.836ms gap remains1.78× after
[vector staging](../results/2026-10-14-matrix-staging.md). Test whether the
different intermediate-precision policies explain a material part, without
changing native production or claiming that aggregate dtype counts prove cost.

First trace the actual posterior obs0 forward/backward block in saved native
plans and reachable JAX HLO. Distinguish BF16 activation gradients from F32
master-gradient accumulation, and fused logical casts from stored buffers.
Recheck upstream and preserve all source/configuration/precision contracts.

Then qualify one diagnostic JAX policy: F32 intermediate computations/interfaces,
explicit BF16 dense/grouped operands and rounded forward outputs, F32 matrix
derivative outputs/accumulation. Retain F32 parameters/optimizer. Convolutions
remain BF16 islands, including their derivatives: asking the JAX/cuDNN path
for F32 outputs previously widened operands too, which would confound the
intermediate-storage question. This is a bounded partial precision-policy
control, not an exact native clone or a proposed learning recipe.

Check custom matrix VJPs against independent F64 values with explicitly rounded
operands, dense/transposed/grouped/batched layouts and changed inputs. CPU
checks precede one guarded GPU qualification; no tolerance relaxation. Inspect
compiled HLO to verify reduced heavy products and unchanged shapes/nominal
work, not just configuration names. Stop for review on an unsupported contract.

After review, one fresh baseline/policy complete RGB12M/B8/T16/H15 pair:
16 warmups+256 measured synthetic updates each, same initial parameter tensors,
input/seed schedule and complete seed/metric/drain accounting. Save every finite
report and final fingerprint; changed trajectories are expected and are not
learning equivalence. Report ordinary timing, not utilization. A >=10% slowdown
motivates one bounded attribution capture after reviewing the executable;
otherwise close this hypothesis's large-cost claim without a precision sweep.
No native lossy change, Atari allocation or automatic training restart.

Artifacts `runs/precision-flow-20261014.YXhwiW88`. Current native42f00066 remains
unchanged. CPU preparation/analysis one CPU/2GiB/zero swap; GPU workers have no
CPU quota. Existing serialized persistent guards, expected5080/580.178.04/boot,
>=2GiB sampled Vulkan budget headroom, recorded allocation warnings. Numerical
jobs<=300s, full jobs<=900s; all other faults stop the affected job for review.
No NVML polling, driver action or invalid native profiler retry.

Review before timing: the first GPU fixture stops on AT forward single-rounding
equality (max difference.00048828125), after the normal case's three calls.
CPU checks pass; no full-update job has started. Retain the failed guard/source.
One<=120s diagnosis of that exact AT case compares the unchanged JAX BF16
forward, policy forward/VJP and an F32-output product to F64, saving outputs and
HLO before assertions. No tolerance change or automatic timing follows.

The diagnosis retains a Python tuple/None preparation failure, then its reviewed
correction saves all three outputs/HLOs before stopping on the original
single-rounding assertion. Baseline BF16 forward, F32 output and custom F32
derivatives match F64; the policy alone differs. Its optimized HLO has removed
the F32→BF16→F32 round-trip and returns the raw F32 product. No GPU fault or
library-arithmetic failure is established. Preserve both failed guards.

Directly requesting a BF16 result also loses rounding in the CPU executable;
its failed source/HLO remains. A generated-fixture syntax error is retained too.
The revised diagnostic uses explicit ReducePrecision(8,7) after the F32-result
product, with F32 derivatives. No global compiler flag, tolerance or convolution
policy changes. Repeat the CPU fixture, then one reviewed GPU qualification
with saved per-case HLO. Require the rounding operation in the executable and
F32 derivative results, plus unchanged exact F64 checks. This diagnostic adds
F32 forward storage too, not an exact native clone. Only a reviewed pass
releases the original full-update pair; no extra precision arms or sweep.

Explicit ReducePrecision passes all CPU values. The GPU fixture passes its
first12 calls, then stops before executing the grouped case because the HLO
checker counts only library calls: that small case has one cuBLAS product and
two compiler-emitted dot products. All three have BF16 operands/F32 results;
the explicit rounding is present. Retain this assertion/source/guard. Repair
only the classifier, checking every dot and library call's operand/result
types, and validate it against all five saved executables before one new
qualification. Arithmetic, all numerical gates and the timing allocation stay
unchanged.

The repaired checker passes; the grouped forward then fails the unchanged
F64 gate by.0009765625. No full-update timing starts. One<=120s, three-call
diagnosis saves the unchanged baseline, F32-result product and policy VJP,
comparing both F64 and explicitly rounded-product/accumulator references.
This distinguishes a custom-policy fault from the small compiler-emitted
dot's arithmetic; no tolerance relaxation or automatic follow-up.

That diagnosis passes as a diagnostic, not qualification: the default BF16
baseline matches F64, but the F32-result emitted dot exactly matches separately
BF16-rounded products summed in F32; its derivatives also differ from F64.
The reviewed final repair specifies JAX's BF16_BF16_F32 algorithm explicitly,
retaining ReducePrecision and all gates. CPU checks precede one<=300s GPU check.
If this cannot qualify, close this proposed timing control as unsupported
instead of extending the fixture/compiler workaround into another campaign.

The explicit-algorithm fixture passes:4,809 exact F64 values,71 frontend values,
1,272 unchanged convolution-island values, all eight HLO contracts and the
guard. Minimum sampled headroom7,964,327,936 bytes; zero allocation warnings.
Retain XLA autotuner hint-fallback warnings, not numerical failures. CPU/GPU
results and source hashes agree. Release only the declared two full jobs,
adding a pre-update HLO gate for explicit forward rounding/F32 matrix results.
The BF16_BF16_F32 algorithm is part of this diagnostic; algorithm selection
may consequently differ. Compare actual HLO/work before attributing timing
solely to memory precision. No native source or production recipe changes.

The baseline completes272 updates. The wide arm compiles and matches all288
initial tensors, then stops before any update: the older full-work inventory
does not handle compiler-emitted dot instructions inside Triton GEMM fusions.
Preserve its guard/HLO/source. Extend only CPU accounting for dot dimensions,
validate against eight saved numerical executables and the exact baseline
inventory, and review the failed arm's saved full HLO before re-declaring only
its unstarted timed updates. This is not a numerical/API failure or a new
precision variant; baseline is not repeated.

CPU review retains a tuple/JSON comparison error, then the original exact
library-rectangle-MAC assertion fails. Every difference is now checked:
18,776,064 padding MACs disappear (18 actions padded to24,255 bins to256),
32,768 scalar outer-product multiplies become a dot, and projection packing
changes with zero net arithmetic difference. All18 metadata groups reconcile
exactly; no percentage tolerance. Raw heavy rectangles83.589070848G→83.570327552G
differ by.0224%, not a changed model or omitted work. The baseline's scalar
outer-product broadcasts are traced explicitly. Keep the original failed
assertion. The admitted timing contrast is the qualified wider-intermediate
policy plus its resulting compiler/library choices, not a storage-only causal
claim. All actual rounding/operand/convolution/initial-state gates remain.
