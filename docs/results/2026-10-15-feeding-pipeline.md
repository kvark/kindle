# A pipeline the Vulkan interface can express, but Naga currently rejects

October15,2026. [CPU-only declaration](../experiments/2026-10-15-feeding-pipeline.md)
· [Data and receipts](2026-10-15-feeding-pipeline.json).

**There is a concrete compiler restriction beyond BF16 or tile selection.**
A two-buffer producer/consumer schedule validates as Vulkan SPIR-V, but current
Naga rejects cooperative matrix operations inside a subgroup-uniform consumer
branch. This is an interface/validation limitation, not evidence that fixing it
will close the remaining1.91× gap. No GPU execution or compiler modification.

## Separate interface limits from implementation choices

The [captured library](2026-10-15-library-convolution-feeding.md) uses larger
reuse tiles and asynchronous copies;19/23 main convolutions exceed this driver's
48KiB explicit shared-memory limit. A literal port does not fit. SPIR-V's explicit
group async-copy operation also requires Kernel capability, rather than Vulkan
Shader. Neither fact proves that every useful overlapping schedule is unavailable.
[SPIR-V instruction contract](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html).

The CPU scheduling skeleton instead assigns two32-lane subgroups to load operands
and four to tensor computation. They use different slots in a two-buffer ring.
Every invocation reaches the same initial/loop/final workgroup barriers; there
are no spin loops, named/split barriers or early exits. This permits thread-level
load/compute overlap in principle, **not** an assertion of `cp.async`, TMA,
hardware overlap, native instruction counts or cuDNN's implementation.

All23 current native convolution selections still match the retained contracts:
nine32×32 and fourteen64×64 products, K32. No geometry sweep. The compiled
skeleton covers their three staging layouts and all three scalar types:

| Layout, including two operand slots, padding and output scratch | BF16/F16 | F32 |
| --- | ---: | ---: |
| Checked32×32 |12KiB|20KiB|
| Packed64×64 spatial |36KiB|52KiB — exceeds limit|
| Packed64×64 weight gradient |34KiB|50KiB — exceeds limit|

All reduced-precision layouts fit; the two over-limit F32 variants do not become
GPU-qualified merely because SPIR-V validates. No device is created for any
variant. Actual image indexing, split partials, compensated accumulation and
forward/dX/dW correctness remain a later actual-kernel gate.

## The compiler restriction is reproducible

Nine small probes use the **actual candidate's linked Naga a97559b**. For each
of F32/F16/BF16, whole-workgroup control flow validates, both subgroup-uniform
`subgroup_id < 4` and deliberately divergent lane control fail `COOP_OPS`.
The latter rejection is necessary; the former blocks this producer/consumer
mapping. Cooperative operations require convergence at their matrix scope;
workgroup barriers still require whole-group participation. [KHR contract](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_cooperative_matrix.asciidoc).

The equivalent shared typed GLSL scheduling skeleton emits nine validated
Vulkan1.3 modules with subgroup-scoped16×16 matrices,192 invocations, three
static workgroup barriers, correct memory semantics and checked shared-byte
totals. This is offline legality/resource evidence, not GPU qualification.
The existing native generator's workaround keeps extra subgroups doing matrix
work to satisfy the conservative analysis; it cannot express dedicated loaders.
Do not disable uniformity validation or falsely mark subgroup IDs workgroup-uniform.

The newer upstream memory-model branchc0b23d16 has an **identical analyzer**
(SHA25687783df5); it does not already fix this restriction. Its other fixes are
not assessed here. Neighboring work, compiler pins and all completed evidence
remain unchanged; no Naga commit or permission request is needed for this audit.

## Proof limits and next gate

1,645,920 schedule-event checks cover every ordering of six subgroup events
within each epoch, for ten reduction lengths including zero/one/tails/long loops.
Five negative checks reject missing initialization, same-slot access, premature
reuse and unequal barrier participation. Loader ownership and output coverage
are exact; this is not a convolution address/derivative oracle.

The next useful discriminator is one actual-kernel numerical and complete-family
comparison of this safe schedule, preserving current arithmetic/layouts and
counting every transform and scratch buffer. A standalone Vulkan diagnostic can
test its merit before any Naga feature project. Extra loading warps, register
pressure, full-group waits and unchanged address work can negate overlap. No
GPU allocation is declared here, no favorable tile subset or speed prediction.
Kernel feeding and whole-update device/host execution remain separate open work.

Three CPU preparation failures remain: the reader initially required ID-form
LocalSize instead of the emitted legal literal form; the Naga probe reused a
reserved identifier, then lacked its storage binding. Five execution-mode
regressions and explicit identifier/binding repairs pass. Original sources,
binaries and logs remain; no GPU retry, relaxed numerical bound or fabricated
compiler success. Final CPU review finishes12:55:34UTC; zero GPU calls/actions/
updates and no worker. Keep qualified isolated67559ce5 and production unchanged.
