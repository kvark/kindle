# Breakout latent overshooting: better ending endpoints, inaccurate timing and tracking

The16-step latent-overshooting learner is **closed and fails its declared
learning comparison**. It improves four-step tracking and32-step ending
prediction,with few non-ending false alarms. Eight-step tracking regresses and
primary16/32 ball energy improves only4.34%,below the25% gate. Ending predictions
also arrive too early. Retain this distinct development capability; do not
replace the trajectory control,extend the learner or claim a proved world model.

## Intervention and identity

The [protocol](../experiments/2026-11-07-breakout-world-overshooting.md) selects
`motion_latent_overshooting`,prediction length16. Preserve observed motion RGB
cross-entropy,one-step KL1/.1 and observed reward/continuation losses. From the
detached posterior state at index15,roll16 prior transitions under real submitted
actions and match each prior to its detached image-informed posterior. Average
mixed categorical KL across the full16×B,without a free-nat floor,and add it at
the existing dynamics scale1. Mask from the first suffix reset onward; retain
terminal arrivals. Future images/posteriors enter targets only.

No additional trainable parameters,inference inputs,decoder executions,sampled
reward/termination scores or behavior heads. The intermediate categorical samples
retain Kindle's straight-through gradients; this is not globally unbiased BPTT.
This tests a different objective,not a scalar adjustment or a demonstrated
variance reduction of the [failed score-function estimator](2026-11-07-breakout-world-score-function.md).

Root `/mnt/data/kindle-breakout-world-overshooting-20261107`. Implementation/source
`e311b2160341f12eb67e120a1645b433960486ca`; native SHA256
`965f2bd2c4b619efe24e841fb0cb325cc85d208743ccbe1aec4fc32728a3f1a0`,
matching root/runtime. This bounded variant is inspired by [PlaNet's latent
overshooting](https://arxiv.org/html/1811.04551v5#S4),which stops posterior targets;
it differs in its single suffix seed,categorical RSSM and detached seed. The
paper did not find consistent RSSM benefit,so the external tests decide retention.

Fresh8,192 updates,8,852,675 parameters,RGB64/12M/B8/T32/Auto,seed41001,rates4e-4.
Keep exact coverage corpora and all8,192 coverage4/2/1/1 source locations. Reused development
initialization/data are not fresh confirmation. Replay2,097,152 arrivals,including
975,941 valid suffix rows and9,570 suffix terminal arrivals. Other exposures match
coverage:10,909 observed terminals,34,440 positive rewards,10,488 natural resets,
75,808 context resets,32,849 sampled unique arrivals of32,850 available. No new
environment actions or emulator frames are collected.

## Shared dense development

All target/label/location/mask arrays match the retained control. Counts below
are supported ball targets within1 pixel under one fixed draw,not independent
episodes or a complete stochastic calibration test.

| Measure | Coverage control | Latent overshooting |
|---|---:|---:|
|Observed ball|108/112|109/112|
|One-step ball|94/109|93/109|
|Four-step ball|33/109|43/109|
|Eight-step ball|14/107|7/107|
|Sixteen-step ball|3/114|2/114|
|Thirty-two-step ball|0/95|2/95|
|Observed brick balanced error|.223954|.200453|
|Native learner-call time|529.515s|580.173s|
|Training loop including exports|796.900s|825.609s|

Observed>=90% and one-step>=89/109 gates pass. Eight-step>=28/107 fails. Better
observed bricks do not establish removal memory; two accurate32-step samples
do not establish reliable long forecasts. Native time rises9.57%,loop time3.60%.
These observed wall times include host interaction and are not isolated kernel
benchmarks. The much more expensive joint-energy learners remain separate results.

`development-curves.png` retains all fixed intermediate checks against coverage,
ST energy and score-function learners. The256 fitting/128 development dense
exports and all horizons remain in root/train; no favorable draw or checkpoint
was selected instead of the fixed8,192 endpoint.

## Frozen prospective-branch comparison

Only the new checkpoint runs: history16 primary/32 secondary,eight fixed draws,
84 groups,300.814s. Reuse the retained oracle with standard sticky.25/repeat4,
full18 commands,eight lanes and33 distinct source contexts across seven categories.
All oracle RGB/action/reward/termination arrays match. Duplicate contexts/repeats
are not independent. The runner's historical fresh-development string is not an
untouched-test claim. No emulator collection or old-model rerun occurred.

| History16 measure | Coverage | ST energy | Latent overshooting |
|---|---:|---:|---:|
|Primary16/32 ball-feature energy|.169502219|.151784428|.162147130|
|NOOP8 false-ball draws|0/64|0/64|0/64|
|Last-life RIGHT32 ending probability,oracle100%|50.211%|97.182%|93.344%|
|Ending32 Brier,all four plans|.144566|.006129|.003192|
|Non-ending origin/plans above5% ending probability at32|0/304|65/304|0/304|
|Maximum non-ending origin/plan probability at32|1.907%|25.990%|1.367%|

Primary energy improves4.34%,failing the declared25% reduction to.127126664.
Observed,one-step,serving,ending32 and non-ending-group-mean checks pass; eight-step
tracking and primary energy fail. Keep per-case false alarms visible alongside
the group gate: the largest new non-ending group mean is.2715%,and all304
individual origin/plans stay below5%. This improves the ending/false-alarm tradeoff
of ST energy,but does not preserve coverage's short trajectory capability.

The retained oracle's empirical ball entropy reference is.004812590 under the
same category/horizon aggregation. New energy exceeds it by.157334540. It is a
descriptive finite-sample reference,not a certified achievable floor,population
calibration bound or new acceptance gate. It does not explain away the gap as
unavoidable stochasticity.

### Endpoint success hides timing failures

At step16 the ending-category oracle has no game-over in any plan. The new
model assigns25.448% ending probability on average,versus3.321% control;
ending16 Brier worsens.003345→.156492. RIGHT alone assigns73.352% at16 versus
oracle0%,with Brier.540567,before reaching93.344% at32 versus oracle100%.
Accurate eventual action-conditioned ending is not accurate event timing.

Other failures remain substantial. Free-flight8 ball energy worsens.020588→
.546234 and life-loss8 worsens.004527→.285502. Some later endpoints improve:
free-flight32.305965→.133285,brick16.049654→.025289; wall32 instead worsens
.296554→.335440. Report the entire curve rather than promoting its best horizon.
Ending32 reward-event Brier regresses.000173→.403468; reward forecasts already
include survival weighting and the terminal arrival. Head fitting versus wrong
latent trajectories is not isolated by this result.

`branch-comparison.json` retains every category,horizon,history,event and action
plan,plus finite-mixture observable energy and fixed controls. `branch-forecasts.png`
shows preselected six cases,oracle/model draw0 at8/32 and survival weights. These
are examples,not paired optimal-draw matches or proof from visual similarity.

## Validation and closure

54 CPU world tests(24 GPU tests ignored in that CPU run),53 explicitly selected/
53 packaged Python checks and strict workspace/all-target plus bindings Clippy
pass. New CPU checks cover mixed masked KL values/derivatives,stopped targets/
masks,causal ancestry,unchanged shapes and auxiliary gradients restricted to
dynamics. Native F64 derivative fixture and full-shape preflight pass. All204
initial tensors and18 initial arrays match;18 arrays reproduce through restore.
The first update changes65 parameters,with three reward-hidden tensors correctly
fixed behind zero output weights. The full run changes all68 logical parameters.

All8,192 finite11-report rows,loss/validity identities,exact source plan/counts,
204 initial/frozen tensors and18 final restore arrays pass. Qualification87,
learning96 and frozen-assessment856 input pins revalidate at closure. All five
host guards and their45 sealed-file pins pass; all204 final tensors match both
branch snapshots. No fault,recovery,worker retry or active Kindle job remains.
`closure-review.json` verifies these execution facts,not learning success.

Preserve original literal-syntax,F32-reference and scalar-zero-sentinel fixture
failures in their logs. The references were corrected for exact F32 graph
constants; a dead scalar parameter's zero gradient is verified rather than
misclassified as a behavior gradient. The initial plot fails because Matplotlib
is absent; separate CPU-only local plotting dependencies produce the chart,with
the failure/log retained. No learning/evaluation repeats. Release the808,714,668-
byte disposable binding build cache; retain wheel,native and separate
`cache-release.json`,leaving the originally pinned build receipt unchanged.

The full goal remains open: precise causal1/4/8/16/32 trajectories,reward/event
timing and calibration,removed-brick memory,late-game/tunnel/second-wall coverage,
untouched families and fresh confirmation. Latent distribution consistency did
not solve these. It does not directly constrain the deterministic rollout state
to the image-informed deterministic state; whether that mismatch explains the
remaining errors is a focused next question,not an established cause. No new
loss,learner,collector,unchanged extension or actor/critic successor is selected.
