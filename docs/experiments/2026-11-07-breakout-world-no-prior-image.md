# Remove the extra prior-image objective from the isolated world learner

The [cross-entropy comparison](../results/2026-11-07-breakout-world-pixel-cross-entropy.md)
finishes with0/112 development balls reconstructed. Output gradients and some
fitting brightness change, but the observation-model failure remains. This
selects a single-term ablation rather than another capacity or data change.

Change standalone `prediction_length`16→0. Keep observed soft-RGB cross-entropy,
reconstruction coefficient100, RSSM dynamics/representation KL1/.1/free-nat1,
reward/continuation, seed41001, RGB64,12M/B8/T32/Auto and4e-4 rates. Keep the
same CNN, categorical latent/deterministic state, decoder, initialization,
uniform source sampling and fixed evaluations. No actor/critic learns.

The removed term compares an independently sampled prior future with a realized
future. Under uncertainty, pointwise squared error and cross-entropy can favor
mean images; sharing the decoder and latent alignment permits interference
with reconstruction. This motivates the ablation without establishing a cause.
RSSM KL still trains action-conditioned latent dynamics. Open-loop evaluation
still forecasts1/4/8/16/32 decisions; setting this training term to zero does not
remove forecast tests or redefine success as an autoencoder.

## Existing runtime and bounded preflight

Reuse native build535de479, SHA256
`adfc2f9ec8e9174b56628ab8c390516ee6acee31dc4151418208bcfdf7f59c54`,
from `/mnt/data/kindle-breakout-world-bce-20261107/runtime`. Rust and packaged
Python-library sources are unchanged; no rebuild or repeated native primitive
qualification is needed. The example runner now accepts0/16, preserves source
samples and handles nine versus eleven loss values. Explicit empty masks avoid
Python's `[-0:]` selecting the entire sequence.61 CPU tests pass, including
equal sampling/exposures except zero prior-terminal counts. Earlier34 CPU Rust
world tests, native F64 loss/derivatives and exact matching-package checks remain
valid; they are not selected for repetition.

Before a learner, declare one guarded full-size H0 preflight on the same source
data. Require204 initial tensors and18 initial arrays to match the completed
cross-entropy H16 control; restore that reference with its metadata intact.
Execute one finite full-size update, verify nine-term weighted loss arithmetic,
save/frozen invariance of204 parameter/moment tensors, and exact18-array restore
of the trained H0 model with optimizer age1 and prediction length0. This is a
new graph variant, not a rerun of the completed H16 preflight. Preserve failures
and stop; no automatic retry or host recovery.

## Fixed learner after preflight review

Select one fresh8,192-update learner and its final frozen restore after reviewing
the preflight. Reuse fitting16,384 actions/65,503 emulator frames and separate
development8,192/32,755. No collector repeats. Exact initial tensors, source
sampler and evaluation origins must match the completed H16 control. Recount
normal exposures; prior-image terminal exposures must be zero. Replayed arrivals
are not new actions/frames. Retain checkpoints0/2048/4096/8192.

Keep four-draw curves at0/1024/.../8192, natural/contact/ending slices, all
forecast horizons, final256 fitting/128 development exports, physical image
metrics, reward/endings, same-support controls and frozen fitting-only latent
readouts. Verify204 frozen tensors and18 restored arrays. Report native training
and whole-loop wall-time separately. Declare exact sources/native/runners/host
and input hashes before execution; use the current AMD host guard and Auto.

Primary observed reconstruction diagnostic remains90% within1 resized pixel on
supported development balls, against H16's0/112. Report detections and errors,
fitting results, false positives, other objects and all draws. Passing this
intermediate diagnostic does not qualify world dynamics or broad reliability.
Development has been reused; untouched tests, prospective action branches,
calibrated stochastic outcomes, late-game/tunnel/second-wall coverage and fresh
confirmation remain unfinished. Actor/critic stays parked.

Root `/mnt/data/kindle-breakout-world-no-prior-image-20261107`.
Implementation and61 CPU tests are ready at publication. No GPU phase has
launched at publication. Do not extend the closed H16 learner or repeat its
qualification, collector, evaluations or audits. No unchanged extension, broad
seed campaign, capacity increase or automatic successor is selected.
