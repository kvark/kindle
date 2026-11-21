# Optional stride-two RGB encoder

Select native implementation and qualification of a default-off encoder option,
`rgb_stride2`. In each of the four encoder stages replace stride-one5×5
convolution followed by2×2 max-pooling with stride-two5×5 convolution, padding2.
Keep channel counts, normalization, activation and flattened output layout.
For supported even input sides32/48/64/96, every stage still halves the side.
Parameter names/shapes/count, replay observations, latent width, CDP heads and
actor/critic structure remain unchanged. Decoder convolutions remain stride1.

The motivation is compute-constrained learning. Each convolution computes one
quarter as many output positions and needs no following max-pool. That is an
operation-count statement, not a measured training-speed claim. The sampling
and nonlinear reduction change; learning may improve or regress. Earlier pixel
diagnostics recover accurate ball position fromRGB64, so resizing alone does
not explain the learned-feature errors. RGB96, crops, coordinate channels and
frame pairs remain closed; this is a distinct encoder computation experiment.

Keep native max-pooled Atari frames and the existing single GPU resize. No CPU
image transform, object label, emulator state or extra temporal channel enters
the model. Config defaults/legacy serialization omit the false flag. Fresh CLI
`--rgb-stride2` requires learned RGB; restore reads the checkpoint option and
rejects a fresh override. Reject enabled use with old native packages before
emulator/GPU construction. Report the option in preprocessing/architecture logs.

## Qualification

CPU checks cover config roundtrip/invalid encoders, all supported geometries,
unchanged parameter names/shapes/count and latent/control shapes, quartered CNN
convolution output sizes, no encoder max-pool, unchanged decoder operations,
world backward graph construction and all encoder gradient owners. CLI checks
cover fresh on/off settings, package capability, restore preservation and
override/evaluation rejection without constructing a native model.

Prepare an independent scalar F64 strided CNN reference. Its CPU fixture must
exercise nonconstant spatial sampling, channel order and all output tokens.
Native qualification compares Tiny RGB64 encoder values/loss and selected
weight, bias and normalization derivatives against finite differences through
that independent reference. Use existing tolerances:3e-4 forward/loss absolute,
3e-4 + .003*abs(reference) for derivatives, step1e-4. Also run two synthetic
Tiny CDP updates and exact parameter/optimizer/config restore, zero Atari
actions. Do not rerun completed historical native fixtures.

Build/package with pinned native dependencies and preserve exact source/binary
receipts and any failures. Qualify the new installed package and default-off
compatibility with bounded frozen retained/actual-initial traces before a new
learner. No backend optimization, dependency update or neighboring-worktree
change is selected. Release disposable build caches only after packaging.

After these checks, separately declare an8,192-action cost/memory check using
the retained12M/RGB64/B16/T16/R16 Breakout recipe, Auto, eight environments,
teacher/guidance fixed and only `rgb_stride2` enabled. Require finite updates,
exact initial parameter identity, own frozen restore invariance, independent
action/reward/frame/guidance replay, at least2GiB conservative full-replay
headroom and a padded200k projection no greater than25 minutes. Changed encoder
computation can change initial outputs despite identical parameters; retain
actual-initial controls rather than claiming identical behavior.

No full learner is selected by this development declaration. A subsequent
fixed-budget learning comparison requires its own protocol and is judged by
frozen own-initial gain per training minute, with sampled/greedy natural-episode
results and original864/90% two-wall gate unchanged. No automatic extension,
sweep or successor follows. Keep current native recipe until useful learning
is measured; the five-game goal remains unfinished.

Root: `/x/Temp/kindle-rgb-stride2-20261106`.
