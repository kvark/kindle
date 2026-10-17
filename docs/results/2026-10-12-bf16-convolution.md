# Native BF16 convolution storage: useful, with a CDP limitation

October12,2026. [Declaration](../experiments/2026-10-12-bf16-convolution.md)
· [Data](2026-10-12-bf16-convolution.json).

| Complete synthetic update | Dense-only BF16 | Also BF16 convolution | Reduction |
| --- | ---: | ---: | ---: |
| CDP12M/B32/T16 |149.274ms|146.315ms|1.98%|
| RGB12M/B8/T16 |66.508ms|61.019ms|8.25%|

CDP posterior20.142→16.849ms supplies the gain; world training61.493→61.871ms
still regresses slightly. RGB world training41.158→36.735ms improves. Keep the
useful complete-update candidate, not a claim that uniform BF16 solves CDP world
training. The [next fixed mixed-stage check](../experiments/2026-10-12-mixed-world-precision.md)
tests the measured faster world-gradient path. No game training resumes.

## Scope and checks

Native BF16 convolution operands and stored results reuse the dense cooperative
loop, with implicit NCHW gathers and checked partial tiles. No im2col-sized global
temporary or new backend. F32 masters, convolution derivatives, grouped products,
optimizer and sensitive reductions remain. Explicit precision selection only;
Auto and live acting are unchanged. Winograd cannot consume the new BF16 buffers.

Sixteen independent F64 forward/F32-master-gradient cases cover RGB and partial
channels, rectangular kernels, strides, padding, multiple images and tiny gradients.
Both F32 and BF16 output storage are exercised; rounded output uses a format-aware
bound, while unrounded forward/gradients retain the original strict pointwise/L2
gates. All nine existing BF16 checks also pass, including reduced derivatives,
1,035 RNE conversions and32 finite Adam updates. Numerical processes take2.00s
and10.24s, with zero allocation warnings. Four CPU precision/shader checks,
12 Kindle config tests and Meganeura/Kindle strict Clippy pass.

Four ordinary arms use one binary/checkpoint/RNG and16 warmup+256 measured
updates. All272 reports/first targets and initial/first/final F32 tensors are
finite; exact346/292 initial tensors match. Dense-only controls reproduce the
retained git-pinned fixture's reports, targets and all saved tensors exactly.
Actual BF16 convolution counts are4 posterior/4 world for CDP and4/8 for RGB.
CDP world dispatches4,424→4,439, standalone casts262→277. Minimum sampled
headroom10.28→9.90GB for CDP decreases: reduced operand storage does not establish
lower whole-agent allocation. No GPU-utilization or learning-quality claim.

All six guards pass; three allocation warnings and known VUID10684 remain.
No recovery, separate NVML polling, game actions or frozen episodes. Initial CPU
preparation errors (test arity, binding/capability setup, Clippy override placement)
are retained; no failed GPU qualification or tolerance relaxation.

## Provenance

Qualified Meganeura17c9287 is the same tree as published66eee3a in
[PR250](https://github.com/kvark/meganeura/pull/250), onb0ba23c with Bladeb52965b
and Naga forkc9fd6a85a. Neither weight-gradientPR248, row-layoutPR249 nor
alignmentPR252 is stacked. Kindle0010772 plus archived explicit-precision/
diagnostic patch; temporary path override is disclosed, not its embedded679ea87.
Source archives, locks, binaries, full reports/checkpoints and independent CPU
audit remain in `runs/bf16-convolution-20261012.Y61SZP2u`. Only the user merges.
