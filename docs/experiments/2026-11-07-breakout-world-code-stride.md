# Observation-only code: strided encoder and lower categorical mixing

Select a bounded practical recipe after the
[code-precision decomposition](../results/2026-11-07-breakout-world-code-precision.md).
Shared position bias and draw variation are both material; neither reaches the
predeclared dominance threshold. This does not identify an encoder or sampling
cause. Test two existing settings together rather than infer an isolated remedy.

Root `/mnt/data/kindle-breakout-world-code-stride-20261107`. Reuse qualified native
780f00a8/SHAe6ca7ca4 and runner68e9bbb7; no native rebuild, new loss or data.

## Fixed intervention

Keep `rgb_latent_only=true`, set `rgb_stride2=true` and categorical `unimix=.001`
instead of .01. Four stride-two convolutions replace stride-one convolution plus
encoder max-pooling. Parameter names/shapes/count, RGB64 replay, one GPU resize,
embedding width, RSSM/core/heads and z-only decoder remain unchanged. Encoder
convolutions visit one quarter of their former output positions; this is not a
measured whole-model speedup. Sampling/receptive fields/nonlinearity and prior/
posterior probability mixtures change together. Disclose the combined recipe.

Lower mixing reduces the uniform probability floor, not learned posterior entropy.
For an idealized fully peaked32-slot/16-class code, the probability of at least
one off-class sample from the uniform floor falls from about26% to3%. This is not
a measured corruption/error fraction or a claim that sampling causes the failures.
Genuine uncertainty from sticky actions remains in the task and must be evaluated.

Both settings have negative completed whole-agent CDP comparisons: the
[stride-two test](../results/2026-11-06-cdp-rgb-stride2.md) and
[lower mixing test](../results/2026-11-04-cdp-breakout-latent-unimix001.md).
They remain negative/closed. This new offline, observation-only RGB code with H16
latent overshooting is a different architecture/objective/data setting, not a
repeat or evidence that those old failures are repaired. Keep the retained agent
configuration unchanged. Do not repeat their native primitive qualifications.
The encoder implementation prefix in native780f00a8 is byte-identical to the
already-qualified22c262ea stride-two source; retain that source identity review.

## Qualification and allocation

CPU preparation verifies that only the two settings change from the closed
observation-code recipe, all66 logical/8,065,219 scalar/198 weight-moment shapes
remain, and the complete original8192-row4/2/1/1 plan/exposures/source selections
match. No labels, crop, extra resize, pretrained encoder or actor/critic.

Separately declare one guarded full-shape composition check(900s), using the
existing exact runtime and native test executable. Require all198 actual initial
tensors match the prior architecture's retained initial checkpoint; actual initial
predictions may differ and must be retained. Check native posterior/decoder
h independence at the combined settings, finite one-update reports/masks/total,
expected trainable-family changes,198 frozen tensors and18 exact initial/trained
restored arrays. Persist both settings through v4 restore. Do not rerun already
passed legacy forecasts, version rejection or unchanged primitive tests.

Only after qualification, separately declare one fresh8192-update learner(3600s)
and restore(600s), then new-final-only84-group oracle assessment(1800s). Keep
RGB64/size12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,loss scales,original
sixteen fitting/eight development streams and full original plan. Use the existing
observation-code runner with matched256/128 dense initial/final and original
periodic cohorts; retain initial/2048/4096/final checkpoints. Fresh does not mean
an independent confirmation seed. No emulator action/frame or old-model query.

Keep all observed/one/eight-step,primary16/32 energy,serve/ending/false-alarm and
per-plan freeflight-presence gates. Report all actions/horizons/reward/paddle/brick
regressions, actual-initial gains and measured wall scopes. No easier slice,
unchanged extension, sweep, concurrent learner or automatic successor. Use AMD/
Auto and existing host guard; stop faults/nonfinite state,no reset/recovery/NVML.

Intermediate success does not complete the goal. Precise causal physics, event
calibration, persistent brick memory, late/tunnel/second-wall coverage, untouched
families and fresh confirmation remain necessary. Actor/critic and CDP efficiency
comparisons stay parked.

## Actual qualification and allocation

First composition qualification passes137 pins in18.789s,with198 exact initial
tensors matching the prior observation-code initial checkpoint,18 initial/trained
restore arrays,finite11-report update,63 changed parameters across expected
families,198 frozen tensors and34 invariant visual parameters/exact h separation.
Legacy qualification and existing primitive evidence are reused without repetition.
The separately declared175-pin fresh8192 learner/restore and1130-pin84-group
new-final assessment are complete. All execution checks pass; quality fails
retention. Native calls489.334s,loop622.496s,observed worker guard span640.489s;
branches128.003s. See the [closed result](../results/2026-11-07-breakout-world-code-stride.md)
for every unchanged gate and timing scope. Keep coverage; no extension,repeat,
sweep or automatic successor. No GPU allocation remains active.
