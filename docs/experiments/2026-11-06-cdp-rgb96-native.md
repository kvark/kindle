# Guarded native qualification for RGB96

Select one execution of each of the three prepared RGB96 fixtures after the
[CPU checks and packaging](../results/2026-11-06-cdp-rgb96.md). Production source
is `1279a4153a0d1e7e2c6cad2edb4df7492abe28c4`. Native SHA256:
`7704d25c9df5d37c48d5223f2d12c1d5d3c0a812c1f68cf393c87a637d010ab0`.
Test executable SHA256:
`0845d39e0ae101a39eb4724bb247deec7ed5a46aeb61e75854cc13640bf32588`.
The build receipt pins256 source files, packaged Python, wheel, library and test
binary. Installed-package checks pass1,208 cases with10 constructor cases
deselected andzero native construction attempts. Original CPU fixture failure
and its separate repair remain preserved.

Run individually with `--exact --ignored --nocapture --test-threads=1`:

- `vision::preprocess_gpu::rgb96_tests::gpu_rgb96_resize_matches_scalar_and_preserves_small_objects`
- `dreamer::networks::rgb::tests::tiny_rgb96_encoder_matches_independent_reference`
- `dreamer::agent::rgb96_tests::rgb96_learns_and_restores_parameters_moments_and_geometry`

The resize fixture compares all centered CHW values to independent scalar
Pillow-equivalent arithmetic, maximum absolute error below5e-4. It checks2×4
colored objects at six positions in native160×210 frames, exact CPU-upload versus
strided resident RGB/RGBA/BGRA agreement and exact inactive-stream invariance.
Require the resized red peak above one RGB8 level; report each peak. This is a
synthetic visibility check, not a claim about learned object perception.

The encoder fixture uses Tiny/B2, seed103, the96px geometry and an independent
F64 convolution/max-pool/channel-RMS/SiLU reference. Require output/loss absolute
error below3e-4 and four selected parameter derivatives within
`3e-4 + .003*abs(reference)`. Finite differences use representable F32 endpoints
around1e-5. Do not loosen tolerances or repeat a failed GPU stage automatically.

The learner fixture collects16 synthetic RGB observations and performs exactly
two Tiny CDP learner updates. Require finite reports, changed first-layer encoder
weights, exactly two world/behavior optimizer steps and exact restore of all
parameters, moments, optimizer counts and geometry. Restored replay is empty.
Save only to the fresh `native-checkpoint` below the experiment root. These
fixtures use zero Atari actions and do not qualify the full12M learning recipe.

Use the existing host guard with Auto on AMD RX7900XT/RADV. Check live boot,
kernel and driver identity, no live Kindle service and no render holder. Pin
source, binaries, protocol, controller, queue and helpers before execution.
Cap each child at120s and the controller at420s. Stop on real faults/nonfinite
results; no reset/reload/reboot or automatic retry. Preserve original failures.
Use sealed guard results for terminal child identities and verify idle closure.

Root `/x/Temp/kindle-rgb96-20261106`. Success permits consideration of the
separately declared8,192-action cost check; it does not select it or a full learner.
The geometry also expands the CDP predictor's hidden/output widths and the four
disagreement output heads. Actor/critic and RSSM widths remain unchanged; this
is a joint capacity/compute/geometry intervention. The35-minute padded200k cap
from the development protocol remains fixed. Retain143.667 greedy/70.750 sampled;
original864/90%-two-wall and five-game gates remain unfinished.
