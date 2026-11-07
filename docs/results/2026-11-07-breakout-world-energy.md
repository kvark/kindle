# Breakout: joint predictive energy improves ending, regresses precise tracking

The [declared distribution-aware comparison](../experiments/2026-11-07-breakout-world-energy.md)
is closed. **The world goal remains unproven.** Retain both models as development
evidence; the new recipe is not a replacement for the coverage control. It costs
86.58% more native learning time, improves ending forecasts substantially, and
regresses short-horizon ball tracking. No actor/critic learns or resumes.

Root `/mnt/data/kindle-breakout-world-energy-20261107`. Native/source161839d9,
SHA256 `8d5552e5bf900e67b632d8e9e685fe9a291a0cc2f2844b9782e9e041f52b934f`,
matching root/runtime and archived source. No collector,old-model rerun or
unchanged extension. No successor or new GPU allocation is selected.

## Fixed comparison

Fresh matched initialization,8,852,675 parameters,RGB64/12M/B8/T32/Auto,8,192
updates. Reuse exactly the coverage data/4-retained/2-new/1-wait/1-terminal plan,
rates4e-4,posterior motion-weighted CE,KL1/.1,free-nat1,mixing.01. Seed41001 is
reused development,not fresh confirmation. All18 submitted actions,sticky.25,
repeat4,native pooled RGB/one native GPU resize. Labels select/audit data only.
No RAM/object/phase labels enter model inputs or loss targets.

Add two independently sampled one-step prior worlds,each RGB decoder mean plus
sampled categorical reward/Bernoulli continuation. Score target distances minus
half sample-pair distance using the joint smoothed Euclidean metric,weighted by
past-frame motion only. Scale100; keep old observed losses unchanged. Reward
coordinates/targets use existing symlog two-hot bins; continuation targets encode
gamma or natural termination. Mask clipped starts/resets,include valid terminals,
no lost-life penalty. Native categorical derivatives remain ST surrogates.
This changes a bundle of predictive RGB/reward/continuation supervision and
sampling; it does not isolate which term causes the observed tradeoff.

Native learning987.968s (16m27.968s) versus529.515s;
loop1338.229s (22m18.229s) versus796.900s,excluding native construction.
Replay2,097,152 arrivals,2,021,344 valid predictive rows;32,850 available/32,849
sampled,10,909 terminal,34,440 positive reward,10,488 natural-reset and75,808
context-reset rows. Other project processes were present; timings are not isolated.

## Reading the image versus forecasting it

On the same128 development origins and fixed draw,observed ball reconstruction
improves108/112→109/112 (96.43%→97.32%). Observed brick balanced error improves
.223954→.207707. Forecast within1 resized pixel:

| Future decisions | Coverage control | Joint energy |
|---|---|---|
|1|94/109 (86.24%)|81/109 (74.31%)|
|4|33/109 (30.28%)|23/109 (21.10%)|
|8|14/107 (13.08%)|4/107 (3.74%)|
|16|3/114 (2.63%)|3/114 (2.63%)|
|32|0/95 (0.00%)|0/95 (0.00%)|

At one step,detected balls fall108→94 among109 supported targets; at32,77→53
among95. Thus the regression includes missing/low-contrast detections,not just
position errors. Detector thresholds/region remain unchanged. Raw RGB panels
are retained; these counts do not establish why contrast or position changes.

Fitting data show the same problem: observed ball159/163→161/163,one-step
153/177→125/177,eight-step34/176→21/176. The regression already occurs on fitting
windows; additional held-out examples alone cannot explain it. This does not
isolate capacity,gradient bias,representation aliasing or optimization as a cause.
No best-checkpoint selection; final8,192 is fixed. Four-draw curves use the same
32 development origins,with fluctuations retained in `development-curves.png`.

## Prospective fixed-action forecasts

Only the new final checkpoint was evaluated against the retained oracle:
history16 primary/32 secondary,eight fixed model draws,84 groups,235.299s,
all204 tensors unchanged. Same oracle RGB/actions/reward/termination arrays for
every plan. Seven groups contain33 distinct native contexts; duplicate contexts,
plans and repeats are not independent episodes. This is reused development.

Primary equal-category mean ball-feature energy over16/32 decisions improves
.169502219→.151784428 (10.45%),**failing the predeclared25% reduction gate**.
The unchanged threshold.127126664 is above the exact empirical-oracle minimum
.004211016; this is a finite-target feasibility check,not a population bound.
The old invalid90% FIRE pair-match gate is not reused.

Primary-history ball-feature energy (lower is better),including predicted
survival/dead mass and detected presence/position:

| Group | Coverage8 | Coverage32 | Energy8 | Energy32 |
|---|---|---|---|---|
|serve|0.577191|0.567080|0.599095|0.410545|
|freeflight|0.020588|0.305965|0.439611|0.202553|
|contact|0.093930|0.256250|0.156387|0.332478|
|brick|0.022748|0.138106|0.049328|0.150285|
|wall|0.064349|0.296554|0.058870|0.168104|
|life_loss|0.004527|0.016517|0.007166|0.032297|
|ending|0.071036|0.158956|0.008714|0.049055|

The ending result is useful: last-life RIGHT mean ending probability
50.211%→97.182%,against100% oracle; its Brier.389513→.002293. Across all four
ending plans,Brier.144566→.006129,passing.10. Predicted mean ending probability
.530977 is near oracle.531250 on this selected group. This is not natural-frequency
calibration or fresh confirmation.

NOOP8 false-ball detections remain0. Every non-ending group's mean32 ending
probability remains below5%(maximum4.103%),passing the declared mean gate.
But assigned ending probabilities increase:65/304 non-ending origin-plan cases
exceed5%,maximum25.990%,versus0/304 and1.907% for coverage. These are assigned
probabilities,not binary false-ending decisions or independent cases.

Reward/brick forecasts are mixed. For example,free-flight8 reward Brier
.402050→.211677 improves,but ending32 reward Brier.000173→.192762 and ending
alive-only brick Brier.003655→.008989 regress. The complete category/horizon,
per-origin/event and secondary-history results remain in `branch-comparison.json`
and `branches/*.json`. No omitted category or favorable-draw selection.

`branch-forecasts.png` shows fixed lane0/draw0 at8/32 for six predetermined
cases,with alive weights and oracle ending marked. It is an illustration,not a
score or representative random sample. In the ending panel,the RGB decoder can
still render after the predicted ending; the scored world carries its dead mass.

## Gradient and evaluation limits

An independent CPU enumeration finds a concrete head-gradient limitation. For
three reward bins[-1,0,1],probabilities[.25,.5,.25],target0,the expected
coordinate-ST gradient is zero while the true expected energy gradient is
[.0624063,-.1248126,.0624063]. Finite differences verify the latter. The existing
native tests correctly verify the implemented surrogate; they do not prove it
is the true categorical objective gradient. This counterexample does not
attribute the full learner's RGB regression or ending improvement to that bias.

A conditional score-function estimator matches the exact gradients in two CPU
fixtures (`score-function-review.json`); no replacement native path or learner
is implemented/selected. The latent RSSM would still use its existing ST path.

The unchanged detector excludes parts of the field; a separate CPU RAM-reference
audit records finite-coordinate balls outside its support,including falling
balls. RAM references are imperfect and enter no model/loss. There are **no
above-top-brick oracle endpoints**,late-game/tunnel/second-wall coverage is
absent,and many long endpoints have no active ball. Good absence/ending scores
do not establish precise surviving-ball flight. Removed-brick memory,full RGB
fidelity,calibrated stochastic futures,untouched tests and fresh confirmation
remain open. No CDP-superiority,reliability or gameplay result follows.

## Evidence and failures

43 CPU world tests(21 GPU tests ignored),41 affected and41 packaged Python checks,
strict workspace/bindings Clippy pass; final-source workspace Clippy also passes.
New513-row native F64 energy values/derivatives and one-update preflight pass.
77 qualification/89 learning/648 frozen input pins verify. Five native guards
and their sealed9-pin audits pass. Exact204 initial tensors/18 initial arrays,
old-objective restore,65 expected first-update changes,68 final changed logical
parameters,8,192 finite13-report rows,204 frozen tensors and18 exact restore
arrays pass. Score/weighted-total/mask identities and the complete plan pass.
No GPU/numerical fault,host recovery or completed-worker retry.

Preserve the first CPU mean-gradient reference failure (exact1/7 versus autodiff
F32 coefficient; repaired at unchanged1e-10 tolerance),wrong Python test filename
and missing-package-path collection failures. Successful repair logs remain;
no GPU ran during repairs. The earlier pointwise prototype's failed final
68-change assertion and separate CPU repair remain in its own closed root and
[protocol](../experiments/2026-11-07-breakout-world-predictive.md#closed-prototype-before-full-learning).
It never ran a full learner. No protocol failure is relabeled as passed.

Exact source/runtime,protocols,queues,reviews,full curves,tensors,exports,panels,
CPU derivations and hashes remain in the external root; `closure-review.json`
links them. Retain both checkpoints as different development capabilities.
No new GPU job or automatic successor is selected. The active world goal remains
open; actor/critic and five-game campaign stay parked.
