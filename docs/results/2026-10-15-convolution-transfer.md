# Transfer feeding helps locally; Vulkan exposes a smaller shared-memory budget

October15,2026. [Compact evidence](2026-10-15-convolution-transfer.json).
The root-cause goal remains open. Latest isolated full RGB12M/B32/T16 stays
54.224ms versus JAX26.632ms,2.04×; no full update is remeasured here.
Production, Naga, Blade and Atari are unchanged. No worker remains.

## Cost observation: real, too small to explain the gap

One shared typed convolution implementation compares8-byte BF16 vector feeding
with16-byte packed-bit feeding at unchanged64×64/K32 geometry. It covers all14
currently packed occurrences/11 distinct forward, input-gradient and weight-
gradient contracts. Native channel padding, layout transforms, split schedule,
256-position dW compensation, eight-lane final sum and output rounding stay
fixed. Complete-operation time includes every packing/scatter/reduction pass.

| Weighted complete operations | GPU intervals | Queued wall |
| --- | ---: | ---: |
| Fresh native control |4.9331ms|4.9585ms|
| Generic typed8-byte control |5.1136ms|5.1567ms|
| Generic packed16-byte variant |4.1697ms|4.2023ms|

The typed control is3.66%/4.00% above native, within the declared10% weighted
calibration gate. Its dW contracts individually differ by12–16%; retain these
differences, not a claim of per-contract calibration. The packed variant saves
15.48%/15.25% versus native and18.46%/18.51% versus typed. All11 contracts and
all128 individual samples per variant remain; no faster-subset selection.

That is **.7634ms**, only1.41% of the retained54.224ms complete update as a
scale comparison, not measured whole-update savings. It cannot explain the
remaining27.592ms gap. Source vector width does not establish hardware load
width, asynchronous copy instructions or native SM utilization.

## Numerical qualification failed; observation is not promotion

The [original qualification](../experiments/2026-10-15-convolution-transfer.md)
stopped at40 forward cancellation-envelope failures. Its separately declared
[diagnostic](../experiments/2026-10-15-convolution-transfer-diagnostic.md)
then stopped at1,277 dX failures. Both failures occur in the unchanged native
control and both variants, with **bit-identical outputs**. Independent integer
contractions confirm every failed F64 reference:

- Forward64→48/HW16: exact sum−.01629638671875, BF16-rounded reference
  −.016357421875, GPU−.01611328125. Error.000244140625 versus bound.000230859375;
  full-fixture relativeL2 6.467e−6. Cancellation condition69,704.5.
- dX48→64/HW16 contract: full-fixture relativeL2 1.615e−6, maximum absolute
  error.000205994, worst bound fraction1.67312. Failed sums have cancellation
  conditions201.6–12,354.3. The contract name uses original convolution CI/CO;
  the derivative maps64 incoming channels to48 output channels.

No tolerance was relaxed. These are real failures under the declared stress
envelopes, not transfer-induced differences or established learning/driver
faults. The exact hardware accumulation cause is not proved.

After review, a [separate observation](../experiments/2026-10-15-convolution-transfer-observation.md)
finished only the last untouched numerical contract, stopped at CPU review,
then ran the wholly unstarted timing set. No completed numerical cohort or
timing was repeated. Qualification remains failed even though cost improves;
no compiler feature, backend promotion or full-update trial follows.

Numerical accounting:6,273,024 independent reference values/18,819,072 F64
comparisons, including the two failed fixtures, and12,546,048 exact native/
candidate value pairs. Inputs, canaries, changed/restored checks and native
shader/geometry/pipeline contracts pass. All132 GPU guards pass; six allocation
warnings remain. Total5,115 optimizer-free operation calls, zero game actions
or learner updates. CPU address/coverage checks and198 F32/F16/BF16 shader
compilations pass; only BF16 executes on GPU.

The first aggregate reviewer retained a stale scalar/map membership check;
the corrected CPU audit verifies both exact failure counts and all seals.
No GPU repetition. Before any GPU work, source review corrected the dW final
sum to the native eight-lane tree; original development sources/binaries stay.

## A more fundamental resource contract

A separate properties-only query confirms this5080/580.178.04 exposes
**49,152 bytes (48KiB) of Vulkan explicit workgroup shared memory**. Matrix2
also reports8,192 reserved bytes. One guard passes, zero warnings, no logical
device or compute dispatch. No over-limit shader or recovery was attempted.
This is the limit on shader-declared Workgroup variables, not physical VRAM or
a measurement of all compiler-internal storage. [Vulkan limits](https://docs.vulkan.org/spec/latest/chapters/limits.html).

The retained actual full-JAX capture uses more than48KiB in **19 of23** main
convolution products. Its profiled dX/dW examples use77,824/81,920 bytes per
block. CUDA supports larger opt-in allocations; the documented compute-
capability12.0 ceiling is99KiB per block. [NVIDIA Blackwell guide](https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-tuning-guide/index.html).

For the captured256×32/K32 four-stage BF16 dX schedule, the ideal complete
operand stages alone require73,728 bytes. The64×256/K32 four-stage dW requires
81,920. A literal explicit-shared-memory port does not fit48KiB. At most two
such complete input stages fit before considering other storage. This explains
why copying a library tile is not sufficient; it does **not** quantify how much
of the speed difference is caused by capacity, rule out another legal schedule,
or establish a performance ceiling for compiler-managed cooperative matrices.

Those19 library kernels account for7.674ms of the10.940ms instrumented main-
convolution total. These are captured library attribution times, not native
cost or additive whole-update savings. The exact raw audit hash and all23
published launch/schedule fields match. The first CPU review incorrectly
compared compact published rows directly with raw rows; its failure remains.

## Next decision

Keep the small transfer benefit isolated; do not turn it into another compiler
project while treating it as the missing multiplier. The useful next causal
question is whether the library stays fast when constrained to a resource
contract our Vulkan path can express, separating resource capacity from its
better addressing/staging implementation.

First audit the installed reference's supported controls. Latest cuDNN docs
describe a shared-memory-limit attribute, but it is absent from this reference's
9.24.0 headers; a stages knob exists. Do not invent enum values, silently upgrade
the reference or assume a particular engine supports that knob. No new GPU
allocation is declared here. [cuDNN engine controls](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html).

The CPU feasibility audit now finds saved stage choices in22/23 products;
observed choices2/3/4 correspond to captured3/4/5-stage kernels, not literal
stage counts. The original profiled cnn1 dX/dW retain engines40/33, both choice3.
The installed header also exposes a shared-memory-used query. Next is one
plan-only capability check of these exact engines and a supported two-stage
counterfactual, keeping all other knobs fixed, before declaring any timing.
Do not infer legacy execution from the serialized frontend=false field: the
pinned XLA source rebuilds engines from IDs and supported knob choices.
[Pinned XLA implementation](https://raw.githubusercontent.com/openxla/xla/3d5ece64321630dade7ff733ae1353fc3c83d9cc/xla/stream_executor/cuda/cuda_dnn.cc).
Raw feasibility evidence: `runs/convolution-resource-20261015.b2Jg6Sbe/control-preflight.json`.

No revival of the rejected spatial family, geometry/dtype/calibration sweep,
failed profiler/assembly retry or unsafe dependency weakening. Upstream remains
Meganeura99e3de27/Bladee7c3f7b. Remaining host targets and dependent execution
are separate mechanisms; the whole gap is not yet causally apportioned.

Raw roots: `runs/convolution-transfer-20261015.IzSzHAYI` and
`runs/convolution-resource-20261015.b2Jg6Sbe`. All sources, gates, samples,
stopped controllers, warnings and CPU failures remain; compact JSON links
their hashes and terminal service receipts.
