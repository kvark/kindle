# Image layouts explain a structural mismatch, not a single missing selector

October15,2026. [Data](2026-10-15-image-layout-contract.json) ·
[Packing-limit declaration](../experiments/2026-10-15-packing-launch-coverage.md).

**The image graph does not preserve a convolution-friendly physical layout
across operations.** JAX's23 convolution interfaces use BF16 NHWC activations
and KRSC filters/weight gradients. Native's14 packed convolutions consume
channel-contiguous operands, but return NCHW activations or ordinary parameter
gradients. Surrounding image operations materialize NCHW intermediates, casts
and fresh packs. Selecting more packed convolutions is not enough: the bounded
launch-limit repair below saves only2.13% of complete RGB updates.

Keep isolated control`076f3135` and the established complete RGB/JAX
**54.113/27.223ms,1.99×** comparison. The trial is not promoted; production and
Atari remain stopped/unchanged. CDP has a different exploration workload.

## Actual executable interfaces

The CPU audit reads the retained complete JAX executable and current native
versioned dispatch inventory, not a miniature convolution benchmark. JAX's
enabled/disabled command-buffer controls previously established identical
fusion/library contracts. Every convolution maps to its encoder/decoder layer.

| Interface | JAX executable | Current native graph |
| --- | --- | --- |
| Eight forward convolution outputs | BF16 NHWC, bias fused | BF16 NCHW |
| Seven input-gradient outputs | BF16 NHWC | F32 NCHW |
| Eight weight-gradient outputs | BF16 KRSC | F32 CoCiKhKw, with partial reduction buffers |
| Normalization/activation to next convolution | Fusion produces BF16 NHWC | F32 NCHW image output → BF16 cast → channel packing |

The native image kernels preserve F32 pooled/inverse values and pooling
winners for backward. Removing those required values is not an optimization.
The eight forward image outputs each have one immediate consumer: a cast,
reshape-related operation or upsampling. Decoder upsampling and subsequent
conversions are explicit. A versioned read/write audit also proves eight
duplicate packs,83,886,080 logical output bytes, including forward activations
packed again for weight gradients. Each duplicate has the same source version,
shape/type/transpose and an unmodified earlier packed output through all uses.
This is a CPU opportunity proof, not a measured saving or memory-traffic count.

The [closed-region intervention](2026-10-15-spatial-boundary-cost.md) already
established that image materialization matters. This audit identifies concrete
producer/consumer contracts behind it. It does **not** attribute the remaining
gap solely to layout:14 convolutions already consume packed inputs, wider JAX
image intermediates previously cost only.92%, and the library's instruction/
reuse/staging advantage remains. Logical bytes are not measured DRAM traffic.

## A real launch cliff, but a small complete-update effect

Four products fall back solely because packing requires65,536 workgroups,
one beyond the selector's65,535 one-dimensional limit: decoder conv0 forward/dW
and encoder cnn1 dX/dW. Their64MiB packed operand passes the existing32Mi-element
allocation bound. A bounded32768×2 dispatch plus coordinate flattening admits
all four. Allocation bounds, other eligibility checks, kernels, tiles, splits
and precision remain unchanged. Five three-channel fallbacks remain.

Independent direct-F64 structured B512 checks cover every output, the last
image, zero padding, dense signed-dyadic scaled inputs, changed input and exact
restoration. Both arms pass403,574,784 runtime oracle comparisons; CPU rechecks
134,524,928 baseline values. This qualifies those structured cases, not arbitrary
training inputs or bitwise equivalence after changing accumulation order.
CPU selection tests prove14→18 packed products and four two-dimensional packs;
existing storage-type shader validation, formatting and Clippy pass.

| Ordinary full updates,16 warm +256 measured/arm | Control | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| RGB12M/B32/T16 |54.117ms|52.963ms|2.13%|
| Centered CDP12M/B32/T16 |47.677ms|46.207ms|3.08%|

The declared RGB≥3% gate fails. No subset, retiming or promotion. All samples
remain. The control matches all272 non-timing reports, first targets and all
292 RGB/346 CDP initial/first/final tensors from retained076f3135. Candidate
initials/first targets match; later trajectories differ, with no learning-parity
claim. The data separates actual parameters from optimizer moments.

Full plan/pipeline exports were inadvertently disabled in the timing jobs;
their declared equality gate is **incomplete**. Sealed source/CPU actual-plan
tests, matching control precision summaries/groups and candidate packed counts
are supporting evidence, not substitutes for the missing exports. Do not call
this a fully qualified compiler promotion. The original CPU reviewer also
incorrectly compared elapsed/stage timings. Its source/failure remain; a
separate observer repair checks every non-timing field and exact retained
tensors, without GPU repetition or relaxed numerical/performance gates.
The first CPU publication retains too many open tensor maps and fails its file
descriptor limit; explicit per-pair closure fixes it without raising the limit.

Six passing GPU guards/four allocation warnings,48 standalone operation calls,
1,088 synthetic learner updates, zero game actions/recovery/separate NVML queries.
No worker remains. CPU preparation/build/review uses1CPU/2GiB/no swap; matched
GPU jobs have no CPU quota or concurrent compilation. All2,440 timing-source
seals pass. Native trial`66f24c3c`; raw evidence under
`runs/packing-launch-coverage-20261015.veO7PGEx` and
`runs/image-layout-contract-20261015.3rtIyFpF`. Upstream99e3de27/e7c3f7b was
rechecked unchanged before implementation. CI430 passes the preceding commit.

## Next mechanism

The next connected lowering should let image producers write the representation
their consumers already require: preserve typed rounding/tapes, fuse conversion
and spatial expansion at those boundaries, and reuse identical packed versions
across forward/dX/dW. Prove complete consumer coverage and escaping values on
CPU before allocation; retain required external NCHW views. Do not silently
change weight layout, image detail, normalization or gradient precision.
Measure the complete RGB/CDP update, including conversion/scratch and both
derivatives. This is a concrete structural route, not a promise of closing2×;
efficient convolution feeding and non-image execution remain separate work.
