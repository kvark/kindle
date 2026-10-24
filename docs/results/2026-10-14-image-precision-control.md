# Wide image precision does not explain the residual JAX gap

**JAX stays fast with F32 image intermediates.** Complete RGB12M/B32 changes
26.632→26.876ms, only0.244ms/0.92%. Retained native65.736ms is still2.45× the
wide-image result. That native number is historical, not a new simultaneous
native/JAX pair. This closes the omitted image-precision hypothesis as a
sufficient explanation, not the performance goal.

[Declaration](../experiments/2026-10-14-image-precision-control.md),
[machine-readable review](2026-10-14-image-precision-control.json).
Raw artifacts: `runs/image-region-20261014.2wQILUdF`.

| Complete-update measure | Original JAX | F32 image path |
| --- | ---: | ---: |
| Mean,256 measured updates |26.632ms|26.876ms|
| Median |26.723ms|27.058ms|
| Including seed/metric host accounting |27.333ms|27.527ms|
| Compiler-reported temporary bytes |675,878,448|750,131,760|

One paired diagnostic,16 warmup+256 measured updates per arm. No learning,
SM-utilization or statistical-equivalence claim. Temporary bytes are compiler
accounting, not measured peak VRAM or DRAM traffic.

## Actual executable contract

Convolution results still round to BF16, then widen **before** F32 channel bias.
Pooling, image RMS normalization and SiLU stay F32. Each next convolution
narrows its input; encoder tokens narrow at the RSSM boundary. Decoder spatial
normalization widens, but its dense MLP and other networks remain unchanged.
Parameters/optimizer stay F32. Upstream source remains untouched; exact scoped
method transformations are retained in `method-edits.json`.

Both executables have112 heavy-operation records and334,356,283,392 nominal
MACs/update. Multiplication operand/result shapes, BF16 dtypes and invocation
counts match. Eight convolution calls lose their fused bias operand, as
intended. Precision changes fusion/library selection: **not a storage-only
ablation**. Full call descriptors remain in `convolution-calls.json`.

The executable audit finds eight image max reductions changing BF16→F32;
image SiLU exponentials change17 BF16→21 F32 occurrences, including compiler
duplication. Three decoder dense-MLP exponentials remain BF16. These are
compiled operations, not GPU kernel counts. Actual288 initial state tensors
match; both arms have272 finite metric reports and finite final state.283 final
tensor fingerprints differ; no trajectory or learning equivalence is asserted.
JAX's convolution gradients, pooling tie derivative, layouts and some rounding
boundaries still differ from native. This is not an exact native-program clone.

## Consequence for the compiler

Wide JAX still combines BF16-to-F32 conversion, channel bias, pooling and
channel-square reduction in one fusion (`fused_reduce.85`), returning only
pooled F32 values and F32 statistics. F32 arithmetic does not require a global
image for each logical operation. Native's qualified pool-tape path already
removes the encoder's separate bias image, but still emits pool→transpose→
normalization→SiLU→transpose and separate corresponding backward operations.
The [whole-program evidence](2026-10-14-jax-image-profile.md) also isolates
substantial non-matrix GPU work and a separate convolution gap.

**Next implement connected, type-agnostic image forward/backward lowering on
the existing precision contract**, including decoder blocks. Fuse remaining
normalization/activation/layout work, retain only needed backward state and
share gradient/reduction work. Require independent values/derivatives,
visible-output/staged-lifetime checks and ordinary complete-update improvement.
No preceding dtype/layout/row/tile sweep. This removes the rationale for changing
precision merely to imitate JAX; it does not guarantee a native fusion speedup
or imply that BF16 is useless for our less-fused executable.

Convolution algorithm/staging quality remains a measured issue; host decoding/
targets remain a separate~5.38ms opportunity. These are not additive savings
forecasts. Keep native3551bc53; production pins, Atari and the full goal remain
unchanged/open. No new native kernel is qualified by this JAX-only diagnostic.

## Qualification and retained failures

Eight nonconvolution image-tail cases (channels3/32, pooling/no pooling,
random/tied windows) pass22,120 independent F64 values/derivatives. Worst
relative L2 is3.68e-7 against3e-5; pointwise gates pass too. This does not retry
or rehabilitate the earlier failed miniature convolution comparison.

Three guards pass. Two failed processes stop on CPU harness/audit errors before
their numerical/update loops: reversed Ninjax state/output unpacking, then a
checker including the decoder's unchanged dense activation. Only unstarted work
runs after repair; the completed original baseline is retained. Two later CPU
report failures confuse moved convolution-bias operands with multiplication
operands and then dense fused-bias operands. Original sources/logs remain; the
final audit reconciles exactly eight moved biases and preserves all other
operands. No numerical limit or runtime policy changes after qualification.

544 synthetic learner updates, zero game actions/native updates; minimum
sampled Vulkan estimated headroom7,930,773,504 bytes. One standalone allocation
warning during qualification is retained under the existing permission; no
hardware fault, NVML polling, recovery or remaining worker. All sources and
guard/review receipts are retained.
