# Lower-noise training while retaining high-noise forecasting

The [six-step intervention](../results/2026-11-09-breakout-denoising-six-steps.md)
improves one-step ball matches but increases pixel error3–4×. Extra inference
passes alone are not a fix. The current sigma_data.05 model trains its unforced
half with lognormal mean−1.2/std1.2, giving only about3.36% of all targets below
sigma.05. Select one fresh bounded learner with log mean−3, keeping half the
targets forced to sigma5. The expected low-noise fraction becomes about25.07%.
This tests noise-range allocation, not a proven unique cause of contrast or
rollout failure. Root `/mnt/data/kindle-breakout-denoising-low-noise-20261109`.

Add optional log_noise_mean, default−1.2/omitted when default, finite in[−6,1].
Keep the original default training-sigma function untouched; use an alternate
mapping only for non-default mean. Preserve Gaussian draw count/order, std1.2,
sigma bounds.002..5 and the alternating forced-max half. Only sigma values of
the unforced half change. Non-default mean requires checkpointv6; defaults keep
v1/v2/v3/v4/v5 and original inference/sampler naming. Restore validates the
configuration/header before GPU. No weights, graph nodes, loss expression,
backend primitive, actor/critic change or pretrained encoder.

Run focused Rust CPU checks against independent F64 sigma values, exact default
math/RNG/forced-max behavior, graph/parameter identity and header compatibility;
strict release test-target Clippy. Package the matching native wheel and run
focused packaged denoising CLI/restore tests plus the four runner checks. Reuse
qualified weighted-loss value/derivative receipts; there is no new differentiated
graph or affine EDM coefficient change. Keep preparation bounded rather than
repeating backend fixtures.

Pin all source/native/wheel/runner/config/data/plan and prior completed evidence
before allocation. First run a guarded fresh default-mean v5 fixture with the
new runtime. Require135 initial/frozen tensors and20 full per-draw initial
rollout arrays exact to the sigma_data.05 control. One qualification update must
match all135 cached first-update tensors and all six losses exactly; retain its
checkpoint and count it separately from the main learner budget. Finish its
terminal guard review before learner launch, enforced by wrapper. Never query
an older trained model; use cached tensors/initial outputs only.

Then one fresh seed41001 learner: same sigma_data.05/history normalization.5,
RGB64/trainB32/inferB8,Auto,rate.001,LaProp epsilon1e-8/AGC.3,context-motion100,
residual/half-max modes, actual four-frame history for each update and unchanged
three-step sampler. Same2,048 five-arrival calls/updates,65,536 targets,224 unique
source arrivals,588 positive prefix exposures and32,768 forced sigma5 targets.
Exact target/pixel-normal draw order; unforced sigma amplitudes intentionally
differ. No new emulator actions/frames, labels, data collection or extension.

Before learning, require135 initial tensors/20 full per-draw rollout arrays exact
to control. First real update counts in budget; before update2 require six finite
reports/correct total,45 changed weights,135 finite tensors, matching event
losses and all57 first event weights/moments exact. Image loss need not match.
Final57 event tensors must remain exact, since actual event conditioning and
its objective are unchanged. Keep all losses/checkpoints and ages0/509/1021/
1533/2048. Require135 frozen tensors and20 restored arrays exact under v6.

Use unchanged fitting exports/gates: four draws83001/83003/83009/83023, every
supported1/4/8/16 ordinary ball within one pixel and>=90% of110 endpoints per
draw; actual-history one-step>=90% of198. Missing predictions fail;18 unsupported
per task/draw stay explicit. Keep own-initial/copy controls, the actual-history
sigma_data.05 scores, every ball/paddle/brick/event regression,1,376 endpoints
and16 fixedlane0/draw0 lossless frames. No best-age substitution or new tolerance.
Loss levels change with noise distribution; judge physical predictions.

Existing AMD host guard,300s qualification/900s learner/300s restore caps, one
learner, Auto; no reset/reload/NVML. Stop on real faults/nonfinite data. Terminal
services/sealed guards precede CPU review; release own disposable build cache
after packaging. Preserve original failures and any separately declared repairs.
No broad learner, extension, collection or automatic successor. Full physical/
event precision, natural calibration, changed-brick memory, late/tunnel/second-
wall/above-top, untouched families and fresh confirmation remain required.
Actor/critic, CDP comparison and five-game work stay parked. No fitting score
completes the full goal; all earlier trials/failures/repairs remain closed.

Completed: [result](../results/2026-11-09-breakout-denoising-low-noise.md).
One-step161/161/158/155 of198 improves, ordinary18/20/23/23 of110 regresses;
3/24 individual gates pass, overall fitting test fails. Retain mean−1.2.
The40 packaged checks cover native config/restore parsing, not a new CLI flag.
All default first-update/initial/frozen/restore/event-state checks pass; both
services terminal, three guards/27 files and285 closure pins verified, cache
released. Original unpublished0/24 report prose remains with a separate CPU
correction to3/24. No score or native assessment is repeated. Additional event
metrics show scalar reward RMSE65.604/59.913 on zero-only ordinary windows;
coverage clarification preserves the original review. No successor selected.
Full goal remains unproven.
