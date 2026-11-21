# Breakout short-horizon consistency leaves substantial return error

The sampled critics have one-step real-trajectory residual RMSE **0.732 / 0.941**
for dynamics4e-4/1e-4, while complete-return RMSE remains **6.682 / 6.691**.
Their own future predictions supply the short-horizon bootstrap. Consequently,
this smaller residual does not establish accurate long-term values or good
fitting of the native training targets.

The [fixed CPU-only diagnostic](../experiments/2026-11-02-cdp-breakout-value-consistency.md)
uses the already-completed critic captures, with no new model execution or
emulator replay. It reproduces all four original complete-return metric sets
and episode-start records exactly. Original game scores and gate outcomes stay
unchanged.

| Dynamics / actions | Test rows | One-step residual RMSE | 15-step residual RMSE | Complete-return RMSE |
| --- | ---: | ---: | ---: | ---: |
| 4e-4 sampled |16,145|0.732|2.546|6.682|
| 1e-4 sampled |17,532|0.941|3.121|6.691|
| 4e-4 greedy |21,361|2.000|6.118|27.261|
| 1e-4 greedy |21,566|1.545|5.511|34.391|

Each arm retains the first three natural episodes per eight lanes. The first
episode per lane is calibration and the next two are the16 test episodes.
Targets sum actual external rewards and the recorded action-conditioned
intrinsic bonus, with the original F32 discount0.9969969987869263 and F64
accumulation. The one-step and15-step targets bootstrap with the same frozen
critic at the horizon endpoint; they stop at natural termination with zero
bootstrap and never cross resets. Complete-return targets have no bootstrap.
These are not the native replay lambda targets, which use imagined-return
bootstraps. No new loss or gradient is measured.

## Values near recorded endings

Each final15-decision slice contains240 rows from the16 test episodes. There is
no bootstrap in its15-step target, so that target equals the complete return.

| Dynamics / actions | Last15 mean value | Last15 mean realized return | Last15 complete-return RMSE | Final-decision mean value | Final-decision mean realized return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4e-4 sampled |14.782|0.377|15.667|11.669|0.0051|
| 1e-4 sampled |19.343|0.592|20.640|15.493|0.0034|
| 4e-4 greedy |30.014|0.885|32.977|15.517|0.0059|
| 1e-4 greedy |28.708|0.600|30.771|27.119|0.0035|

The sampled15-step residual RMSE outside those final15 decisions is1.695/1.991,
versus15.667/20.640 inside. This locates a substantial residual on the observed
endings, but does not show that the critic should have known each ending in
advance. The slice is conditioned on subsequent realized termination, even
though its definition was fixed before analysis. A correct expected value can
exceed a realized return on an unlucky trajectory. Stochastic actions, partial
observations and the small episode count limit a calibration claim for this slice.

Sampled arms remain primary because their action distribution matches the
imagined actor. Greedy changes both policy and visited states. Differences
between models are also on different trajectories. Rows within episodes are
correlated; the76,604 test rows are not independent trials.

The result supports examining propagation of realized outcomes into value
learning. It does not isolate representation, continuation prediction,
bootstrap quality, actor training or terminal-state coverage as the cause.
In particular, it does not justify replacing the existing objective solely
because its short-horizon residual is smaller. No new learner, loss, cap,
inference planner or automatic successor is selected.

## Execution and evidence

Analysis completes **02:27:46.302 UTC November2/source9010df1** in6.366s
(6.472s wrapper). Five hand-calculated target/terminal/input fixtures pass.
All32 diagnostic inputs,73 original capture pins and912 active Qbert inputs
remain unchanged. The original QbertPID882846/invocationd78987af is unchanged
before and after. All per-episode metrics and target arrays are retained.
No native module, model forward pass, emulator construction or learner update
occurs. Retain the CPU co-host cost without correcting Qbert training time.

The first attempt fails during import because the ordinary environment lacks
`kindle._pixel_guidance`, before selection or analysis. Its original source
and0.105s failure receipt remain pinned. A separate correction uses the existing
matching Breakout runtime and blocks emulator constructors. Importing ALE is
required by the existing analysis helper dependency chain; it does not construct
an emulator. No successful capture, analysis, replay or collector repeats.

Root `/x/Temp/kindle-breakout-value-consistency-20261102` is complete. The fixed
protocol snapshot, declaration, original import failure, corrected execution,
`analysis.json` and four compressed target-array archives remain there. Do not
rerun the completed analysis. The original Qbert confirmation continues.

DeclarationSHA`c8f3ef87a9e015052eae7ba33d5efaa92f95aef9d0a4913680fdddd5aef41749`;
analysisSHA`2f7605b3afce3bb7cc70237d935409d0aaa545dfd5511674996f46d08d876a18`.
