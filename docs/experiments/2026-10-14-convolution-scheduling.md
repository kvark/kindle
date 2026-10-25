# Generalize the independently established convolution ordering boundary

The [single-boundary result](../results/2026-10-14-convolution-boundary.md) gives
5.29% complete RGB12M/B32 improvement. Its new ordering follow-up finds storage
and execution-only boundaries both4.168ms versus7.429ms intact, with exact
outputs/states and two passing guards. Explicit storage-access scopes are not
needed for that local gain. This does not measure hardware cache behavior.

Artifacts: `runs/convolution-scheduling-20261014.BJ5AVZFt`. Test one general
recording rule: within an existing dependency group, insert an ordinary compute
barrier whenever adjacent dispatches change between convolution-product and
non-convolution-product kernels. Keep consecutive convolutions together and
consecutive non-convolutions together. Cover the existing spatial convolution
product variants independent of dtype/shape; exclude pooling, upsampling,
packing and Winograd transforms. No hard-coded dispatch index, size threshold,
kernel change, reorder, extra submission, new Vulkan API or blanket serialization.

Apply the rule while recording the existing groups, after allocation; retain
all original dependencies, memory layout, cache chunk ranges and staged prefix.
Share the recording helper between cached/ordinary and untimed profiler paths.
CPU tests cover scalar/typed classification, homogeneous/mixed/empty runs and
the observed auxiliary→dX/dW sequence. Export the automatically chosen boundaries
for every runtime plan and verify them independently against its descriptors.

One three-arm complete-update comparison: unchanged grouping, previously
qualified single-boundary control, and automatic grouping. All use identical
RGB12M/B32 config/initial state and16 warmup+256 measured synthetic updates;
GPU timing stays off. The first two reuse qualified nativef3bdb0ac with only its
existing diagnostic boundary option differing. The automatic candidate uses
the original Blade96c1dc7 timing branch, not the new execution-only probe API.
Require exact272 reports, first targets and initial/first/final292 tensors,
five plans/pipeline keys, cache/submission counts, buffer capacities and all
original group boundaries; disclose the single control's one extra group.

Keep the general candidate if it is at least3% faster than unchanged grouping
and no more than3% slower than the fresh single-boundary control. This tests
retention/generalization of the established gain, not an automatic claim of
additional speedup. Require the selected boundaries to include the previously
established location. Any failure closes this rule without a threshold, dtype,
geometry or grouping sweep. No new JAX capture, CDP/B8/learning promotion,
production repin, Atari work or root-cause-goal closure.

Three300-second host-guarded jobs in a930-second persistent service;
Restart=no/KillMode=control-group, no GPU CPU quota. Require5080 and >=2GiB
sampled Vulkan headroom; record allocation warnings. CPU preparation/review
1CPU/2GiB/zero swap. Review all failures, no blind retries.816 synthetic
updates and zero game actions; timings cover complete updates, not SM utilization.
