# Guarded native qualification for the stride-two encoder

Select one execution of each of the two prepared fixtures after the
[CPU checks and packaging](../results/2026-11-06-cdp-rgb-stride2.md).
Production source is `22c262ea2a9b1eb2d221b227f1e2f022908de1ef`.
Native SHA256: `cde123e1aa0d68f61cd6111fe392d605d2ae6864de39b4349a139c82301f6b3e`.
Test executable SHA256:
`da5dcc9901bba568e260d0db99f5546f3c373c0c163223e944ab4930a1687374`.
The build receipt pins260 source files, packaged Python, wheel, library and test
binary. Installed-package checks pass1,246 cases with10 constructor cases
deselected andzero native construction attempts. The initial CPU fixture compile
failure and its separate repair remain preserved.

Run individually with `--exact --ignored --nocapture --test-threads=1`:

- `dreamer::networks::rgb::stride_tests::tiny_rgb_stride2_encoder_matches_independent_reference`
- `dreamer::agent::rgb_stride_tests::rgb_stride2_learns_and_restores_parameters_moments_and_geometry`

The encoder fixture uses Tiny/B2, seed103, RGB64 and an independent F64
stride-two convolution/channel-RMS/SiLU reference. Require every output and
loss absolute error below3e-4 and four selected weight/bias/normalization
derivatives within `3e-4 + .003*abs(reference)`. Finite differences use
representable F32 endpoints around1e-4. Keep declared tolerances and retain
original failures; no automatic repeat of a failed GPU stage.

The learner fixture collects16 synthetic RGB transitions and performs exactly
two Tiny centered-CDP updates. Require finite reports and tensors, changed
first-layer weights, two world/behavior optimizer steps, and exact restore of
all parameters, moments, optimizer counts and config. Restored replay is empty.
Use the fresh `native-checkpoint` under the experiment root. Zero Atari actions;
these fixtures do not qualify12M gameplay or learning efficiency.

Use Auto on AMD RX7900XT/RADV and the existing host guard. Check live boot,
kernel and driver identity, no live Kindle service and no render holder. Pin
source, binaries, protocol, controller, queue and helpers before execution.
Cap each child at120s and the controller at300s. Stop on real faults/nonfinite
results; no reset/reload/reboot or automatic retry. Audit sealed guard results,
terminal child identities, source invariance and idle closure.

Root `/x/Temp/kindle-rgb-stride2-20261106`. Successful native qualification
allows separately declared default-off frozen compatibility and then the
8,192-action cost/memory check. Neither those allocations nor a full learner
are selected by this protocol. Retain the original RGB64 teacher/recipe and
143.667 greedy/70.750 sampled. Original864/90% two-wall and five-game gates
remain unfinished.
