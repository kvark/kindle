# Preserve a packed image view across training consumers

The [staging trial](../results/2026-10-14-convolution-spatial-staging.md) is
rejected. Its CPU audit finds ten transpose→BF16→pack chains and eight duplicate
packs in the retained qualified world plan. This allocation tests one connected
physical-layout/reuse lowering, not another packing-grid, vector-width, tile,
split or precision sweep.

Extend the existing typed channel pack with an explicit input order and input/
output storage types. Compose a compatible channels-last producer and exact
conversion with packing; reuse equivalent immutable-source views across forward/
backward consumers. Elide intermediate writes only when no visible output or
remaining consumer needs them. Honor schedule locks, bounds, padding, precision
and staged lifetimes. Keep the existing convolution selection and arithmetic.
This is not a global NHWC rewrite or a promise that static byte counts close
the whole gap.

Base: qualified spatial-tape candidate3551bc53, not the rejected vector loader.
Artifacts: `runs/connected-layout-20261014.9VuGzPsL`.
Meganeura99e3de27/Bladee7c3f7b rechecked unchanged before implementation.

CPU shader/selection/dataflow checks precede one bounded typed-pack numerical
process (300 seconds; exact cases and calls sealed before launch), including
connected packed-convolution forward/backward/staged consumers. The retained
first-CNN fixture alone does not exercise these newly composed packs, so it is
not the qualification gate and will not be run merely for compatibility.
Require independent exact layout/conversion outputs, retained F64/large-output
checks, changed-input/alias/poison checks, expected5080 and at least2GiB sampled
Vulkan estimated headroom. Hardware native F32 cooperative multiplication is
not implied by F32 storage coverage.

If numerical/source contracts pass, run one ordinary full RGB12M/B32 pair:
16 warmup+256 measured updates/arm,300 seconds each, cached four-chunk control.
Require exact initial/first/final292 tensors, all272 reports/first outputs,
audited plans and no unexpected arithmetic change. At least3% less complete
update time is the keep gate. No microbenchmark speed gate or duplicate-count
claim substitutes for the full update.

Serialize heavy work in guarded persistent user services; record allocation
warnings, stop/review numerical/API/hard faults and deadlines, no blind retries.
No CPU quota on GPU workers or compilation during timing. Preparation/review:
one CPU,2GiB, zero swap. Preserve failed evidence. No separate NVML polling,
unsafe Vulkan capture, production repin, Atari allocation or goal closure.

The first numerical process completes all162 typed-pack cases/486 calls, then
hits a CPU fixture shape assertion while constructing its first connected
graph: convolution requires the transposed input to be reshaped flat. No
connected graph runs. Preserve the failed process, source and binary. Add the
missing reshape and CPU construction checks for every connected shape; the
production lowering/shaders remain unchanged. After reviewing the saved pack
outputs and logs, run only the unstarted connected cases in a separate guarded
300-second process. Do not repeat successful packs or relax numerical gates.
