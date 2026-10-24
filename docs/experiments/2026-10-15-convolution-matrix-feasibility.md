# Explicit convolution matrices: bounded feasibility screen

Declared October15,2026 before the new shader build or GPU work. The CPU
inventory at `runs/convolution-matrix-lowering-20261015.Eol3OoJ2/inventory.json`
reconciles all23 actual RGB B32/T16 products:8 forward,7 input gradients and8
weight gradients. Fully expanded inputs total17.406GB; the complete proposed
lowering writes18.922GB of logical intermediates and requires4,604 dispatches.
Reusable scratch peaks at111.85MB with each allocation below67.11MB. These are
logical sizes, not measured traffic, bandwidth or a latency lower bound.

Do not build the whole lowering on those counts alone. First test one fixed,
optimistic prerequisite: materialize the input-patch matrices, starting with
images already in NHWC storage with the native channel padding. Preparation
of that layout is outside timing. Include every patch/chunk of all23 products;
retain native K order, fixed32/64 output geometry for chunk sizing, the existing
4,096-position dW partitions and a maximum67,107,840-byte patch. No GEMM,
weight/gradient transform, output scatter, final reduction or changed learner
is implemented by this screen. Zero cost for those missing operations makes
this a deliberately favorable feasibility check, not an integrated comparison.

Use one bit-preserving16-bit Vulkan storage-copy shader; no arithmetic or
Naga/Blade change. Before timing, compile/validate its SPIR-V and independently
check all output bits for small asymmetric images spanning every actual
channel/role/order contract, plus stride2/asymmetric-kernel/padding cases.
Include channel/K tails, cross-image chunk boundaries, padding, changed/restored
input and source/output canaries. For actual-sized timing, check independently
selected first/last-chunk output samples and guards. Preserve all failures.

One numerical job followed, only on success, by one ordinary all-contract job:
16 warmup plus128 measured calls per product, every chunk included. Retain all
samples, GPU intervals, queued wall and recording time. No counters, parallel
build, clock manipulation, geometry/layout/chunk sweep or fastest-subset result.

Decision gate: if the sum of patch-only GPU means is at least15ms, close this
explicit-materialization implementation before adding GEMM. The retained exact-
value convolution intervention is about20ms and a useful10% complete-update
improvement needs about5.4ms; this screen asks whether enough room plausibly
remains. This is an engineering gate, not a mathematical latency lower bound
for another fused/cache-aware implementation. Different observation coverage
must not be subtracted into a promised saving. Below15ms permits a separately
reviewed complete-family implementation, not promotion or a speedup claim.
That implementation must preserve the existing dW256-position compensation;
plain generic GEMM does not currently supply it.

Serialize both jobs under the existing host guard in a persistent systemd user
service, Restart=no,KillMode=control-group,10min service/180s numerical/300s
ordinary bounds. Require the5080 on580.178.04, boot
`3e89d55c-a9e5-472f-a18a-06508c5bafa7`, and at least2GiB sampled Vulkan estimated
headroom. Record standalone allocation warnings; API/numerical/hard/deadline
failures stop the affected work for review. No GPU recovery, NVML polling,
game actions or learner updates. CPU preparation/build/review:1CPU,2GiB,no swap.
Seal sources/binaries/inputs before GPU work. Upstream99e3de27/e7c3f7b was
rechecked at07:11:35UTC and unchanged. Production and all rejected candidates
stay unchanged; the full RGB/JAX root-cause goal remains open.
