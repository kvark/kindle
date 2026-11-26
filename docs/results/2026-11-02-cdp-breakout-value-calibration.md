# Scalar calibration recovers a small Breakout value signal

A correction fitted on eight earlier episodes reduces the stronger model's
sampled-return RMSE from **9.231 to 7.698**, versus **7.855** for the constant
fitted on those same episodes. It beats that constant in six of eight test
lanes. There is a small recoverable value signal; an offset alone still trails
the constant. Most complete-return variation remains unexplained.

This [fixed CPU diagnostic](../experiments/2026-11-02-cdp-breakout-value-calibration.md)
uses the four completed [critic captures](2026-11-02-cdp-breakout-replay-value-critic.md).
It introduces no model execution, gameplay, learning, or inference correction.
All original scores, training costs and zero-of-24 two-wall outcomes stay fixed.

## Earlier-episode fits and later-episode error

The first natural episode per lane supplies calibration rows; the next two
per lane supply sixteen test episodes. Fit a constant, an additive correction,
and one unconstrained affine correction `intercept + slope * value`. The raw
critic and all corrections use the same test rows. No test return enters a fit.
Targets include actual external rewards and the frozen intrinsic bonus through
natural termination, with the recorded F32 discount and zero bootstrap.

| Replay-value / mode | Raw RMSE | Offset RMSE | Affine RMSE | Constant RMSE | Affine slope | Affine intercept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 sampled |9.891|9.922|8.288|8.384|.078787|12.957702|
| 1.0 sampled |9.231|8.313|7.698|7.855|.168761|15.999586|
| 0.3 greedy |26.611|26.401|26.735|29.067|.623129|10.727171|
| 1.0 greedy |23.708|19.732|19.780|21.152|.569686|22.209321|

The sampled affine slopes strongly shrink variation in the raw predictions.
For replay-value 1, this lowers RMSE by16.60% from the raw critic but only1.99%
from the constant. Affine test R² is−.001, compared with raw−.439; correlation
stays.231 because a positive affine correction preserves ordering. For0.3,
affine R² is.003 and correlation stays.218. This is limited state discrimination
in the scalar value output, not evidence that useful information is absent
from the learned representation.

The fitted sampled offsets are−3.544943 for0.3 and+5.627584 for1.0. Their test
RMSEs still exceed the respective constant, so a global offset alone does not
account for the observed error. Greedy changes the target policy: its larger
bias is partly recoverable by an offset, without establishing an error in the
sampled-policy training objective. The two models visit different states.

| Replay-value / mode | Affine MSE minus constant | Conditional lane95% interval | Lanes better than constant |
| --- | ---: | --- | ---: |
| 0.3 sampled |−1.606|[−7.204, 2.136]|2/8|
| 1.0 sampled |−2.432|[−5.359, −.471]|6/8|
| 0.3 greedy |−130.109|[−258.687, −12.686]|7/8|
| 1.0 greedy |−56.146|[−96.850, −20.147]|7/8|

Intervals use5,000 whole-lane resamples with fixed seed20261102 and common
resample indices across arms. Each resample retains row weighting and both test
episodes per selected lane. They condition on these reused development lanes,
fitted coefficients and realized futures. They do not measure independent-learner
reliability, eliminate outcome noise, or establish a causal weight effect.

Calibration-value quartiles fix the bins before reading test outcomes. For the
stronger sampled critic, the four test bins have mean predicted/realized returns
**8.591/15.223**, **10.241/14.472**, **11.081/15.448**, and **17.196/19.618**.
The upper bin has greater realized return; the lower three are not monotonically
ordered. All four arms' bin boundaries, counts and metrics remain in the artifact.

## Implication for training and retained evidence

Real rewards and episode endings already enter the replay value loss. Its
targets also use imagined-return bootstraps, and its loss includes slow-value
regularization; see the [native target review](2026-11-02-cdp-qbert-replay-value1.md#what-the-existing-replay-value-term-already-uses).
This analysis measures complete realized returns, not the actual targets or
gradients seen during training. It focuses further work on target construction
and fitting. Applying these fitted coefficients to training or acting has no
measured gameplay benefit, and no such change is selected.

All seven fixture groups pass in.078s. The single analysis completes
**08:44:53.508 UTC November2/source6a90267** in**.918s** (.973s wrapper).
All20 diagnostic and74 original input pins pass before and after. All four
original complete-return metric groups reproduce to1e-12; original episode-start
records reproduce exactly. Native and ALE are never imported. No preparation
or analysis failure, successful-stage repeat, GPU work, or learner update occurs.

Root `/x/Temp/kindle-breakout-value-calibration-20261102` is complete. It retains
the fixed protocol, qualified source, declaration, four compressed arrays,
coefficients, bin/lane statistics and execution receipts. Do not rerun the
completed analysis or original captures. No Kindle allocation is active at
closure. The five-game goal remains unfinished; no new learner is selected.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `112fd441630719f02f0edad8461ea29efffcee1c4c3650fcb5b657efd33de8b0` |
| Analysis | `337bc12e7972442207549953333cf15d59ed5255a5b3024affbc1bdabfa20427` |
| Execution | `98208dc258073b866931ad889740726e070b2abe9a095311e8335126dcfd094b` |
