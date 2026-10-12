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
