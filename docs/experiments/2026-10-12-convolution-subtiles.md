# Convolution hardware-row alignment

Declared October12,2026 before GPU execution. Full-trace attribution rejects
the previous weight-gradient candidate. No repeat of that candidate or kernel
sweep. Inspecting actual selected shapes identifies a separate legality issue:
forward and input-gradient selection require batched channels to align with
the32-row workgroup, although their stores guard16-row hardware sub-tiles
independently. Kindle's48-channel layer is unnecessarily rejected.

Change only that row-alignment condition. Keep the conservative column gate,
small-channel rejection, strict precision policy and all shaders unchanged.
Check48-channel forward/input gradients across32 distinct images with the
existing independently indexed F64 forward/input/weight oracle, using rounded
F16 operands to isolate indexing/accumulation, plus unrounded scalar/tiny-gradient
controls. Require actual cooperative selection, not fallback parity. CPU tests
must retain rejection of truly partial hardware rows. One120-second numerical
check precedes any full timings; stop/review failure without relaxing tolerances.

Then compare control and candidate in ordinary CDP12M/B32/F16 and RGB12M/B8/F16,
the same retained checkpoints/RNG and16 warmup+256 measured updates,600 seconds
per process. Meganeura control7ba39d8; candidate contains its unchanged tracing/
BF16 foundation, the behavior-equivalent deadline-test repair and only the new
alignment selection change. No PR248 weight-gradient kernel or layoutPR249.
Native5080,>=2 GiB sampled headroom, finite complete reports/checkpoints, exact
initial346/292 tensors and positive new forward/input-gradient selection are
required. Report whole-update/stage time; do not promote a microbenchmark gain.

Use archived source/immutable binaries, serialized non-restarting systemd and
the existing host guard; no compilation during matched timing. Retain allocation
warnings. No game actions, frozen episodes, default precision change or learning
quality claim. Artifacts: `runs/conv-subtile-20261012.h7EJMiRL`.

## Failure review

The first numerical process stops after3.12s; no timing arm starts. Forward and
input gradients pass, but the unchanged unsplit scalar weight reduction fails
at index11344:−.0020698346 versus F64−.002082056555, exceeding the unchanged
absolute-plus-relative gate. Source, binary and failed guard remain; no new
allocation warning or GPU recovery. This is not yet attributed to the new gate.

Before follow-up, run one120-second baseline diagnostic on7ba39d8 plus the
behavior-equivalent deadline-test repair, without the alignment change. Use
the identical F16-rounded x/dy and forced64-wide scalar gradient plan. Require
the same exact failed output bits and F64 error; retain the original numerical
failure, not a relaxed tolerance or a claimed qualification pass.

The baseline diagnostic reproduces the exact failed output bits and F64 error
with all cooperative paths disabled. The changed alignment affects forward
and input gradients, not that scalar weight reduction. Retain the broader
failed gradient qualification; do not call it a successful F64 weight check.

A separately scoped correction tests those two changed primitives directly,
with the same two shapes/random data and unchanged F64 tolerances. Require both
cooperative shaders, every forward/input-gradient output across all32 images,
and unrounded scalar/tiny-gradient controls. No altered weight-gradient shader,
summation rule or smaller fixture. The original unsplit failure remains a
limitation. One120-second check, then the four previously unstarted full-update
timings only if it passes; new output directories, no blind retry.

The scoped check and four complete updates pass; CDP161.51→158.12 ms.
Before publishing the independent upstream change, repeat only that numerical
test against current maind7a7293/Blade56f0565 plus the identical two-file patch,
one120-second guarded process. Its full-update evidence remains on the disclosed
BF16 foundation above; no unsupported same-binary claim. CPU preparation's
private helper error and the empty-spec controller failure started no GPU work
and remain alongside their corrected sources/logs.
