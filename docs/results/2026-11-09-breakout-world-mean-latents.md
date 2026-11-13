# Mean-latent world state: native qualification

The optional mean-latent state passes implementation qualification. Useful
learning and the full Breakout world-model goal remain unproven.

[Protocol](../experiments/2026-11-09-breakout-world-mean-latents.md).
Root `/x/Temp/kindle-world-mean-latents-20261109`.

Both posterior and prior states use mixed categorical probability vectors in
training and inference. Their Gumbel uniforms,one-hot samples andstraight-through
paths are absent. The softmax code has true derivatives for this deterministic
objective. Keep logits,KL1/.1/freeNat1,mixing.01 andall68 parameters/8,852,675
scalars/204 saved tensors/11 reports. No new head orimage weighting. This changes
state representation andgradient estimator; it does not isolate sampling noise.

Four requested draws are identical,not independent stochastic futures. A
nonlinear readout of mean probabilities isnot the mean of sampled forecasts.
This is not a calibrated stochastic world model. Original branch/distribution
and natural-event tests remain; no easier deterministic-only goal is substituted.

149 world CPU tests pass,43 GPU tests ignored;29 config tests,strict core/binding
Clippy and31 packaged Python checks pass. Five targeted mean-mode CPU checks pass
after a fixture range-loop style correction. Initial strict Clippy failure and
source remain; no native stage repeats. F64 values/true softmax derivatives,CPU
probability agreement,no RNG advancement,no argmax/noise inputs,causal ancestry,
parameter/inference-graph identity,default serde/header/constructor checks pass.

Native primitive F64 values/derivatives andCPU probability parity pass. The
new default-off path matches all204 old initial tensors and32 cached initial
prediction arrays exactly. Mean mode matches the same204 initial tensors but
changes predictions as intended. All eight model arrays match across four
requested draws,24 equality comparisons,at both initial andtrained weights.
Full RGB64/12M/B8/T32/H16/Auto one-update qualification gives11 finite reports,
RGB100 total/reset identities and65/68 changed weights,including
encoder/core/prior/decoder. Own204 initial/trained frozen tensors and32 trained
restore arrays are exact. Flagv24 survives restore; requested draws stayidentical.

Default-off source computation remains compatible; old checkpoint headers
cannot silently enable mean states. Standalone-only restrictions reject Gaussian,
spatial,previous-frame,latent-only andauxiliary-head combinations beforeGPU.
Whole-agent construction/restore rejects the mode. The inference graphs remain
unchanged; the CPU probability feed changes state semantics. This primitive
CPU/GPU agreement isnot bitwise end-to-end training/inference trajectory parity.

One successful guarded qualification takes25.791s,including primitive,
default-off compatibility,new-mode update/readbacks andrestore. One world
optimizer update,zero main learner updates/emulator actions/frames. One host-only
guard/nine sealed files;no NVML/reset/reload/reboot ornative retry. These costs
are not a backend speed comparison. Own811,446,272-byte build cache isreleased;
wheel/native/test executable/source/logs/checkpoints remain.

Sourceaffedefa/native SHA256
`cc6022e36fbb0c3d145058fe85f5fee66c38172e5e54335bfc051cdcbcf07494`.
CPU test SHA256
`ce559f996de4d320486055d9cac5f535397da957dfeeb3038507be5ff2edafd1`.
Invocationbda0d545b1f548fbb0f75b3d7541eef4.

Only a separately declared fresh8,192-update broad learner may follow. Original
H16/RGB100 data/settings andfixed observed≥90%,one≥89/109,eight≥28/107 gates
remain. New actual-initial predictions must be exported,not borrowed from the
sampled control orqualifier age1. No learner isallocated bythis qualification.

Full precise physics,natural reward/life/game-over calibration,removed-brick
memory,late/tunnel/second-wall/above-top,untouched families andfresh confirmation
remain required. No fitting/implementation score completes thegoal. Actor/critic,
CDP comparison andfive-game work stayparked;earlier jobs remain closed.

Closure verifies157 pins,one terminal successful service/nine sealed files;811,446,272 bytes ofown build cache released.
