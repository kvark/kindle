# One-step prior continuation investigation

The full comparison does **not** pass its predeclared improvement rule. Direct
one-step prior continuation supervision scores **147.625 greedy / 49.167 sampled**,
versus **143.667 / 70.750** retained, with zero of24 two-wall successes in every
arm. Training takes **21m35.191s**, .57% longer than the control. Greedy gain per
minute rises2.22%, but its paired interval crosses zero; sampled performance falls.
Retain the existing recipe with this optional loss off. The original864/90%
two-wall gate and five-game goal remain unfinished.

The subsequent [matched-arrival diagnostic](2026-11-04-cdp-breakout-prior-continuation-arrivals.md)
finds lower true-ending prior MSE (.911006→.162000), but higher intermediate-life
MSE (.001530→.178824) and29/96 false below-.5 predictions. Prior AUC.791667→.807726
has a paired interval crossing zero. It establishes a prediction tradeoff, not a
causal explanation of gameplay or an accepted improvement.

## Full200k comparison

The [fixed full protocol](../experiments/2026-11-04-cdp-breakout-prior-continuation-200k.md)
adds only `prior_continuation_scale=1` to the retained learning configuration:
12M64/Auto/N8/B16/T16/H15/R16, replay-value1/expert100 and the same learning
rates, labels and guidance. It starts fresh on development seed1009; no bounded
experience or weights are transferred. There are no new parameters, class weights,
terminal replay mixture or multi-step auxiliary targets. Native/runner versions
and qualified guidance wrapper files differ from the historical control, so this
is not an isolated causal estimate.

| Measurement | Retained control | Prior continuation1 |
|---|---:|---:|
| Training actions / updates |200,000 /12,477|200,000 /12,477|
| Emulator frames |799,595|799,588|
| Training time |21m27.910s|21m35.191s|
| Greedy mean return |143.667|147.625|
| Sampled mean return |70.750|49.167|
| Own initial greedy / sampled |2.208 /1.417|2.208 /1.417|
| Greedy gain per training minute |6.590132|6.736457|
| Sampled gain per training minute |3.230038|2.212029|
| Two-wall successes, each mode |0/24|0/24|

Each assessment uses the first three natural episodes in each ofeight lanes,
with guidance, labels and updates off and zero truncations. The primary greedy
return difference is+3.958, paired lane95% interval[−55.338,58.333]. Its gain/min
difference is+.146325[−2.613242,2.672747]. The lower endpoint does not exceed zero,
so the locked improvement rule fails. This does not establish equivalence or harm
in greedy play. Sampled return falls30.51%, difference−21.583[−36.667,−7.292];
sampled gain/min falls31.52%. These are eight-lane,10,000-resample intervals on a
reused development cohort, not training-seed reliability estimates.

The auxiliary objective executes throughout all12,477 updates. Independent
recount reproduces every world total, loss mean.028900, valid fraction.998330 and
5,333 native-reported masked rows out of3,194,112. Sampled row identities are not
logged; the masked count is not an independent reconstruction of those identities.

| Training actions | Auxiliary loss mean | Valid fraction | Masked rows | Terminal rows | Updates without terminal rows |
|---|---:|---:|---:|---:|---:|
|0–50k|.041464|.997565|1,934|1,921|1,652/3,102|
|50k–100k|.027094|.998214|1,429|1,416|1,988/3,125|
|100k–150k|.024262|.998631|1,095|1,095|2,199/3,125|
|150k–200k|.022871|.998906|875|882|2,347/3,125|

Terminal exposure totals5,314 rows (.16637%) across4,291 updates, versus5,376
rows across4,372 updates in the control. Equal row weights and the original
sampler remain active. Loss decline alone does not demonstrate better ending
discrimination, probability calibration or imagined action ranking; matched
forecast quality has not been measured for this learner.

Late expert-label cross entropy is.558399 versus.560514 control, without an
accepted gameplay gain. Actual guidance overrides89,849 actions (44.9245%). Of
7,832 actions at pre-action scores216–431,6,894 are guided and938 native; none
reaches432 and maximum score including tails is408. Guided online returns and
these score bands are not frozen competence or verified wall events. The reviewed
curve retains its online last50-episode mean82.48 separately from frozen scores.
Learning calls take1,150.420s of1,295.191s total; wrapper timers do not isolate
individual GPU kernels.

All389 current/338 retained pins,100k/6,227 recovery,357 matching initial and
invariant frozen tensors,47 pre-update events,799/794 actual-initial semantic
events, finite reports, parameter changes, full training pixel/action/label and
frozen action/outcome/frame replays, videos, guards, coverage and curves pass.
The independent recovery check takes.579s while preserving the original active
process. CPU observer costs remain in wall time; no correction is applied.

Source`ae754547336f835028d0f87ed3651c55e5903808` launches08:03:00 UTC November4,
PID1148345/invocation`5e9d868e14e04ebb8c703bab0177aaa5`. The main service succeeds
08:28:41, original collector08:29:18. Independent3.342s CPU closure completes
08:29:38.010925, verifying55 completion artifacts, all exposure arithmetic and
both frozen comparisons. All three original services succeed. No full-run failure,
repair, cap change or repeated stage occurs; the earlier bounded CPU fixture
rejection remains preserved below.

Evidence remains at `/x/Temp/kindle-prior-continuation-20261104/full`. Native
`07873e7c` and its qualified runtime remain unchanged. No GPU allocation is active
at closure; check live state before work. No extension, second learner, coefficient
sweep or automatic successor is selected.

| Full artifact | SHA256 |
|---|---|
|`declaration.json`|`e4fdaecac5c4dc83a601c6d590cf4c7d347323da1d3fa6f9e5c17ff4021f6654`|
|`comparison.json`|`d61b1c4532f3724e9d05794ead05863364316c454b4f89c28e51ec4257e2a5fd`|
|`completion.json`|`0ab0c5248041b34821a14c06d366ddc1a2f701874470802c7e6f33a8029a033d`|
|`terminal-summary.json`|`0f524a76ebef68577284515300a4373081a5c55324c30b2836601b6bc9090fc0`|
|`closure.json`|`07729cc263dfc89477b2457a8f0f23c18dfcf25805ee17882224c45653e3e570`|
|`candidate/learning-review.json`|`704bbafda6661c777e2666ba803d950006929d2969b62c6e467ab7dc0c6d07ea`|

## Bounded qualification before the full comparison

The bounded test passes8,192 actions/32,728 frames/489 updates in **47.016s**.
It directly supervises continuation on sampled one-step prior states with the
shared native head, real terminal flags, equal row weights and scale1. The option
defaults to off and adds no parameters. The conservative200k projection is
**21m32.945s**, below the30-minute gate. A1,024-action frozen restore takes.892s
with zero updates and no completed natural episodes. At that stage, execution
and early cost were established; gameplay benefit was unmeasured and no full
learner had yet been selected.

The [protocol](../experiments/2026-11-04-cdp-breakout-prior-continuation.md) and all
completed stages remain fixed. Do not repeat them.

## Continuation is used in both targets and loss weights

Native imagined lambda returns use continuation at the arrival state; actor and
critic losses use cumulative continuation including the posterior start. Replay
targets use actual termination flags but, in this recipe, bootstrap from imagined
returns. The imagined-return shader function and CPU lambda-return/weight helpers
are byte-identical between retained447fe50 and terminal-weight3672a21a. This is a
scoped source check, not broad runtime equivalence.

An exploratory CPU summary of all24,954 saved learner reports finds:

| Training actions | Control imagined continuation | Weight10 imagined continuation | Control advantage attenuation | Weight10 advantage attenuation |
|---|---:|---:|---:|---:|
|0–50k|.982457|.961853|.896782|.803722|
|50k–100k|.994997|.987439|.965450|.910465|
|100k–150k|.996035|.993549|.968834|.944760|
|150k–200k|.996546|.995194|.972274|.957095|

Attenuation is sum(weighted absolute normalized advantage) divided by sum(absolute
normalized advantage), not mean survival or a parameter-gradient norm. In the last
window, absolute advantage is.019345 control/.027131 weight10 and weighted absolute
advantage is.018808/.025967. The weighted signal is therefore larger in weight10
despite stronger proportional attenuation. Return scales differ33.744/30.092 and
mean imagined returns16.823/18.157. These data do not support a simple overall loss
of actor signal; they do not exclude a damaging change at particular states.

The runs have different experience and learned states. These descriptive means
are not paired causal estimates, calibration measurements or significance tests.
Analysis takes.697s CPU on one core at nice10, with no model/emulator/GPU work;
both full logs match the previously completed experiment's manifest before/after.
Exact values and source-function hashes remain at
`/x/Temp/kindle-terminal-weight-use-20261104`. No analysis is repeated.

## Selected implementation and qualification

The auxiliary term uses the current transition's deterministic state and sampled
prior categories, before the arrival image enters the posterior. It uses an
independent RNG stream, the same continuation head and aligned real targets.
Reset rows receive zero weight and stay in the ordinary B*T denominator. Gradients
reach dynamics/prior and earlier posterior states. Acting, forecast graphs, the
posterior loss and disabled configuration remain unchanged.

The first CPU fixture fails configuration validation because centered CDP was
paired with truncated BPTT. Its source and13.003s failure are retained. Correcting
only the fixture to batch2/BPTT2 preserves production code and assertions. All
184 Rust library/5 gym CPU tests then pass in13.334s wrapper;18 focused Python
tests pass in.203s. Graph dependency checks exclude the arrival observation and
posterior sample, verify prior/head gradient paths, unchanged parameters/inference
and compilation of ordinary/staged training. Existing CPU microbatch partition
and graph-size checks pass; this bounded GPU run uses B16/micro16. Numerical
parity across other microbatch configurations is not established here.

Strict workspace/binding Clippy and1,064 packaged Python checks pass, with10
GPU-construction tests deselected by the CPU harness. Native
`07873e7c4b19fc7c863fb418dc15932912d1f7fb`, SHA256
`aeecf9e7d04cd27b03a9b8660e2d1531e8ca96d1a4c23c1f2e0ea623a5f1d254`,
is packaged at `/x/Temp/kindle-prior-continuation-20261104/runtime`.

Two guarded GPU fixtures pass without a retry. Masked BCE loss/logit derivatives
agree with F64 at1,17,256,513 rows, including logits±30 and masked terminals:
maximum loss error1.023e−7, derivative error1.716e−9. Masked derivatives are zero.
The complete prior-loss graph has finite nonzero gradients in dynamics, prior
and continuation head, loss.828785, exact invariance to changed arrival image/
posterior sample, and exact zero loss under reset masks. Native inference remains
unchanged; the fixtures are training checks, not gameplay evidence.

## Bounded execution and closure

All314 current/338 retained pins,357 matching initial/invariant frozen tensors,
47 pre-update semantic events,first update376/residual credit.5,489 finite reports,
changed CNN/reward/continuation/actor/value,full training pixel/action/label replay
and guards pass. Disabled frozen compatibility matches128 semantic events and
357 tensors. The guided training replay verifies21 natural episodes and3,788
actual overrides (46.2402%). Guided scores are not unaided competence.

Auxiliary loss mean is.090883; native valid fraction mean.997276/minimum.9765625,
corresponding to341 masked replay rows out of125,184. The loss uses the same
keep-action input whose native producer zeros replay `is_first` boundaries;
the graph and numerical fixtures verify that mask routing. Sampled row identities
were not logged, so the masked-row count is a native report, not an independent
reconstruction of those sampled identities. Terminal exposure is343 rows across
238 updates versus322/225 in the retained bounded check. The other251 updates
have no terminal rows. No class weighting or sampling mixture is active.

Every world total includes scale1 times the auxiliary loss, with maximum
recomputed arithmetic error.0001106 within the fixed relative tolerance. A
separate.018s CPU reader reproduces all489 totals, loss/valid-fraction means,
minimum valid fraction and masked-row count from the immutable reports.

The main service`kindle-breakout-prior-continuation-check-20261104` launches
07:49:57 UTC November4, PID1146962, invocation`d2acf1e299084ff49fb54b32965fb1bd`.
It succeeds07:51:09; original collector succeeds07:51:19. Independent1.881s
CPU closure completes07:51:43.704520, verifying66 final artifacts, all pins,
process identities, exposure, invariance and the cost decision. Both services
are terminal success. No bounded preparation/run/collector failure or repeated
GPU stage occurs; the earlier CPU fixture rejection remains preserved.
Closure status is`complete_with_cpu_fixture_repair`.

The retained bounded reference took47.690s. The slight timing difference is not
an isolated speed effect: native/runner versions and later trajectories differ.
This check uses12M64/Auto/N8/B16/T16/H15/R16,replay-value1/expert100 and retained
learning rates/guidance. Its projection uses exactly25×bounded time×1.10; it is
an early estimate, not a full training measurement.

Root`/x/Temp/kindle-prior-continuation-20261104` is closed through the bounded check;
runtime,wheel,test executable,build cache and all evidence remain retained.
No learner or GPU allocation was active at bounded closure. Retain143.667
greedy/70.750 sampled,all0/24 two-wall. The separately declared full comparison
above follows this completed check; no bounded stage repeats. Original864/90%
two-wall and five-game goal remain unfinished.

| Artifact | SHA256 |
|---|---|
|`build-receipt.json`|`71b8fdec84ba06a1ff8aecdadcfe6e6e71510d31011899532dff8dc9d20cded6`|
|`numerics/review.json`|`5ba54f093c3e0b794bd02ed548418e478fd1b204b696c5e0b8413d49ec276bc2`|
|`bounded/declaration.json`|`320458a3a3b872f8dc51aa95899038a6c2503b0eb21c5144572b3b6e18589bba`|
|`bounded/check/review.json`|`1188c389e2f7844ebc8ca8a7cab4e8661368a97915e460582f92867616dbf095`|
|`bounded/check/completion.json`|`0ae4c323b7790e9f2df8ae5340c793816fd40761931388f6563b854a2aaae1cb`|
|`bounded/check/terminal-review.json`|`78b04720f0173d9dcf7450a50567c5a420a7b0d6c52ca98f39021eed07907759`|
|`bounded/closure.json`|`2e308de92cd89af33d8f87331d2d10cbb40359bd469b079d0ffe9ab38ba497af`|
|`bounded/auxiliary-arithmetic-review.json`|`09a361aba2726778f3c0265a025cca4e085d82ccd10c2b2c44a42b2dd0a25f2a`|
