# Larger convolution reuse passes correctness and the instruction screen

October 21, 2026. The [fixed policy](2026-10-21-convolution-reuse-contract.md)
now uses actual optimized convolution gathers, not virtual dense operands.
Across all 23 convolution products, composed warp instructions fall
**10.302B → 8.101B (21.37%)**, with identical tensor instruction counts.
This passes the declared 3% instruction gate. **No ordinary timing has run.**
Retain isolated79bc7cec: RGB42.808ms/CDP40.563ms,1.642× inclusive JAX.
Production, Atari and the five-game objective are unchanged.

| Composed complete convolution family | Control | Candidate |
| --- | ---: | ---: |
| Workgroups |283,600|105,936|
| Warp instructions |10,301,837,184|8,100,610,944|
| ALU-pipe instructions |6,480,686,720|5,323,256,448|
| LSU-pipe instructions |1,139,600,000|720,136,832|
| Tensor-pipe instructions |180,224,000|180,224,000|

Nine typed products were freshly captured: all seven changed products reduce
warp instructions; two unchanged products remain exact. Fourteen unchanged
packed products retain their prior actual-model counters, not new captures.
The nine controls reproduce all 35 current-model anchors across seven grouped
regions. This is a complete-family instruction composition, not a fresh learner
execution, time decomposition or speedup prediction. Native and cuDNN still
use different kernels, tiling and feeding schedules; native SASS is unavailable.

## Implementation and qualification

The isolated Rust generator retains original incremental address recurrences,
predicates, vector loads, bindings, BF16 rounding, K order and weight-gradient
compensation. Larger A/B staging feeds more output fragments per workgroup.
A 4KiB fragment output buffer is reused with publication/reuse barriers.
Actual subgroup IDs determine matrix ownership; local invocation IDs only
assign independent loads/scatters. The observer explicitly enforces full32.

All 46 actual-binding modules validate offline, including 23 byte-exact control
SPIR-V pairs and 16 byte-exact unchanged candidate pairs. Independent address
checking covers 30,627,840 mask/address slots and 28,048,368 valid addresses.
First/last outer slices are enumerated; intermediate images/splits follow an
integral-image translation proof, not exhaustive full-batch enumeration.

GPU checks pass 3,372,544 F64 comparisons, 372,703,232 paired values and
742,033,920 exact-zero checks, including capture rechecks. Fixtures activate
first/last images with full512 grids. Changed inputs, restoration, immutable
inputs/uniforms and256-byte tails pass with unchanged tolerances. These are
primitive fixtures, **not broad actual-model gradient or learning qualification**.

There are 18 passing guards, zero allocation warnings, 1,008 primitive
dispatches, 450 counter passes and 1,566 offline-reproduced metric values.
Zero learner updates, game actions or ordinary timing samples. GPU work ends
13:32:24UTC; publication checks467 receipts at13:33:15UTC. All workers terminal.
Byte-verified hardlinks reclaim2,260,205,568 duplicate bytes while preserving
all36 output paths and contents; shared archives must remain immutable.

Preserve both CPU failures: missing scopes around repeated input-gradient
component locals, and an observer-source comparison differing by one terminal
newline. Separate repairs pass before GPU execution; neither caused a GPU retry.
No Naga change or native learner build occurred. Latest Meganeura main remains
a75c38e1880de8e25e44ca169cc5f737812385ae.

## Next: measure cost without weakening the runtime contract

Integrate this fixed policy into an isolated native runtime, with explicit
geometry/cache identity, enforced full32 support and unchanged fallback kernels.
The current source-bundle keys are not runtime cache integration. Ordinary
Blade exposes a default subgroup width but does not enforce full membership;
do not substitute that property for the observer's explicit pipeline contract.
Qualify complete model plans/gradients and then separately declare complete
cost comparisons, including transforms, partial reductions and epilogues.
No geometry sweep, rejected loader-pipeline revival, production promotion or
Atari restart follows from an instruction-count result.

Artifacts: `runs/convolution-reuse-impl-20261021.cnlSFFuq` (sources, modules,
addresses); `runs/convolution-reuse-gpu-20261021.Sa9MlzPA` (`publication.json`,
all numerical/counter outputs, declaration, final audit and retained failures).

## Runtime prerequisite: explicit full subgroups (CPU only)

In `runs/convolution-reuse-runtime-20261021.fTithS5D`, an isolated Blade copy
now exposes optional compute-pipeline full-subgroup requirements and device
capabilities. Vulkan queries/enables the two required features, checks compute
stage support and supported widths, and attaches the required-size chain plus
full-subgroup flag only for explicit requests. Defaults retain no chain/flags;
Metal/GLES reject explicit requests instead of silently ignoring them. Width,
X divisibility and subgroup-count limits are checked without overflow.
The contract follows the [Vulkan pipeline-stage requirements](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateInfo.html).

Two CPU tests cover capability boundaries and the actual Vulkan stage builder,
including unchanged default configuration and unsupported requests. Native
library compilation and strict Blade/Meganeura/Kindle library Clippy pass.
No GPU initialization, backend cross-platform build or runtime enforcement
observation has run. All native callers still request defaults: convolution
selection and shared cache identity remain to be connected before GPU use.

Preserve the two test-build failures: the helper unnecessarily tied its
descriptor's inner lifetime to its temporary pipeline borrow; reconstructing
the test's base descriptor alone did not repair that. Separating the lifetimes
passes. Initial successful-check source and both failed-test sources remain
alongside the logs. Separate target files do not share mutable inodes with
sealed caches. No Naga edits, GPU work, production changes or new speed claim.
