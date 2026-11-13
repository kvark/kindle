# Distribution-aware one-step Breakout world training

The world-only goal remains unproven. The [coverage comparison](../results/2026-11-07-breakout-world-coverage.md)
fixes waiting but leaves drifting trajectories, weak ending predictions and brick
memory. The [pointwise predictive prototype](2026-11-07-breakout-world-predictive.md)
was parked before a full learner: independently penalizing each prior sample
against one realized future can favor averaging distinct valid futures. Preserve
its failed final worker assertion and separate CPU repair; do not repeat it.

Select a two-sample joint predictive **energy score** alongside unchanged
posterior reconstruction/KL/reward/continuation losses. Energy scores compare
forecast samples to the observation and to each other; the distributional
motivation follows [Gneiting and Raftery's primary scoring-rule report](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation).
This is an experiment in useful learning, not evidence of calibration or an
unbiased gradient estimator. Native categorical samples use straight-through
softmax derivatives, including the added head samples; their gradients are
surrogates. Existing observed losses are auxiliaries and can favor other behavior.

Root `/mnt/data/kindle-breakout-world-energy-20261107`. No actor/critic or collector.

## Exact objective

At each arrival t, independently sample two categorical prior states, conditioned
on the same true past and incoming action, before observing image_t. Decode each
RGB mean, sample a reward bin from its categorical head, and sample continuation
from Bernoulli(sigmoid(logit)). RGB uncertainty is represented by the latent;
there is no independent per-pixel noise. Reward coordinates are the symlog of
existing raw bin support, not the head's raw-reward expectation. Continuation
coordinates are0/1, consistent with the existing discounted-continuation target.

The target world contains actual RGB, an independent draw from the observed
reward's existing two-hot encoding, and an independent Bernoulli draw with
probability0 for a natural terminal and gamma otherwise. This intentionally
scores those existing encoded head targets, rather than claiming direct physical
survival calibration. All draws use the separate rollout RNG; posterior RNG and
initialization remain unchanged. No RAM, object, phase or policy labels enter
inputs or targets.

For samples X1/X2 and target Y, score
`0.5*(d(X1,Y)+d(X2,Y))-0.5*d(X1,X2)`, where

```
d(A,B) = sqrt(sum(w_rgb*(A_rgb-B_rgb)^2)
                 + reward_scale*(A_reward-B_reward)^2
                 + continuation_scale*(A_cont-B_cont)^2 + 1e-6)
```

Positive detached RGB weights have per-image mean1, using the qualified motion
weight function on **image_t-1 and image_t-2 only**. Starts or previous resets
use uniform weights. The metric must not depend on image_t; target-dependent
weighting would change the distribution being scored. Targets, weights and masks
are detached. Existing posterior RGB CE still uses current-versus-previous
motion weights, unchanged from the control.

The smoothed distance is chosen to avoid a singular derivative at identical
samples. Our mathematical rationale is that subtracting its constant sqrt(eps)
gives a positive mixture of Gaussian-kernel distances:

```
sqrt(r²+eps)-sqrt(eps)
  = 1/(2*sqrt(pi)) * integral_0^infinity
      (1-exp(-t*r²))*exp(-eps*t)*t^(-3/2) dt
```

This is a derivation, not an empirical claim of proper trained behavior. The
constant shifts scores but not their comparison/gradients. The two draws estimate
the forecast-pair term independently; using the same draw twice would lose it.

Mask incoming action-1 (clipped starts and natural resets), include valid
terminal arrivals, and normalize by all B*T rows. No lost-life penalty. Keep
`prediction_length=0`: this is one-step supervision throughout teacher-forced
sequences, not an open-loop suffix. Add reconstruction_scale*energy (100 here)
to the old total. Reward/continuation scales1 are coordinates inside the metric;
the old observed head losses retain their original scales.

Thirteen reports: original nine, then joint energy, mean target distance, mean
sample-pair distance, valid fraction. Verify energy=target_distance-.5*pair and
weighted total separately. No parameters, latent width, inference path or geometry
change. Explicit `pixel_loss="predictive_motion_energy"`, checkpoint v2. Older
readers reject the objective; new readers preserve production v1/v2 objectives.
The archived pointwise prototype remains readable with its matching runtime.

## Bounded qualification

CPU checks independently verify F64 score values/derivatives, reset/all-zero
masks, target/weight detachment, hard samples and softmax surrogate derivatives,
true two-point distribution versus mean collapse, target-free prior state,
gradient connectivity, ordered parameter identity, full graph compilation and
checkpoint objective rejection.43 CPU world tests pass(21 GPU tests ignored),
41 affected Python tests and strict workspace/bindings Clippy pass. Preserve the
first derivative fixture: its reference divided gradients by exact7 rather than
the F32 mean coefficient used by autodiff. Corrected at unchanged1e-10 tolerance.
Also preserve the wrong Python test filename and missing package-path collection
failures, then the successful explicit-runtime check. No GPU ran during repairs.

After CPU checks/build/package, pin source/native/runner/executable/data/plan/
reference bytes. A serial guard runs only the new independent F64 native energy
fixture (513 rows, including masked rows; 180s cap), then one full-size preflight
(900s cap). Existing AMD host guard/Auto; no resets or backend project.

The preflight uses the coverage plan's first batch including waiting/terminal,
compares204 initial tensors and18 arrays, restores the old motion objective,
and matches old observed first-update components at1e-5 relative tolerance.
Verify13 finite reports, score/total identity, exact mask fraction; expect65
changed logical parameters. The three unchanged reward hidden tensors follow
from exactly-zero initial reward output weights, independently verified against
the earlier motion preflight. Do not repeat the prototype's incorrect68 assertion.
Verify204 frozen tensors and18 age1 restore arrays. Qualification does not prove
learning. No full learner is selected until this review completes.

## Subsequent fixed comparison

Only after qualification review, separately declare one fresh matched-init
8,192-update learner on the exact coverage data/4/2/1/1 plan. Seed41001,
8,852,675 parameters,RGB64/12M/B8/T32/full BPTT/Auto,4e-4 rates,
reconstruction100,KL1/.1,free-nat1,mixing.01,reward/continuation1. No actor,
critic, imitation, replay-value or intrinsic training. This is a reused development
initialization, not fresh confirmation. No warm extension or collector.

Verify32,850 available/32,849 sampled arrivals,10,909 terminal,34,440 positive,
10,488 natural-reset,75,808 context-reset and2,021,344 valid one-step rows of
2,097,152 replay rows. Retain0/2048/4096/8192 milestones, fixed evaluation locations,
four-draw curves,256 fit/128 development exports,204 frozen checks and18 exact
restored arrays. Native/whole-loop wall time matters: two added decoder passes
cost compute. Caps3,600s learning/600s restore. No automatic successor.

After learner review, separately declare only this checkpoint's frozen evaluation
against the retained oracle: history16 primary/32 secondary,eight fixed draws,
84 groups,1,800s. No old model rerun or recollection. Report1/4/8/16/32 forecasts,
proper finite-mixture feature energy/event Brier, false alarms, support, brick
memory and full RGB panels. Detector limitations and duplicate origins remain.

Retain intermediate gates: observed ball accuracy>=90%; NOOP8 false detections
<=10%; last-life32 ending Brier<=.10; every non-ending group's mean32 ending
probability<=5%; equal-category mean ball-feature energy over16/32 at least25%
below coverage. Report every category and regression, not favorable clips. FIRE
90% pair matching remains an invalid criterion; use scores/oracle spread.
Untouched tests, late-game/second-wall states, removed-brick memory, stochastic
calibration and fresh learner confirmation remain required by the larger goal.


## Qualification reviewed; fixed learner declared

Native/source161839d9,SHA8d5552e5 with matching root/runtime. Both guarded jobs
pass with sealed audits: new513-row F64 native energy values/derivatives, then
full-size preflight.77 input pins;204 exact initial/frozen tensors,18 initial/
restore arrays,old objective restore and original observed components match.
Exactly65 first-update parameters change,with the independently verified three
hidden reward tensors unchanged; new output weights move.13 reports satisfy
energy/total/mask identities. No numerical/GPU fault or host recovery.

`qualification-review.json` reviewed. `learning-protocol.json` and
`learning-queue.json` separately declare the fixed fresh8,192 updates and restore
only,89 input pins,source/native as above. All exact budgets/settings/counts in
the preceding section apply. Training is active; frozen oracle evaluation has
not yet been declared. No collector,old-model rerun or automatic successor.


## Learner reviewed; frozen evaluation declared

All8,192 finite13-report rows,89 learning pins,exact original204 initial tensors/
18 development arrays,68 changed parameters,204 invariant final tensors and18
restore arrays pass; both guards/sealed audits pass. Native987.968s,loop1,338.229s.
Dense observed ball109/112 versus108/112,one-step81/109 versus94/109,eight-step
4/107 versus14/107. Brick balanced error.207707 versus.223954. No blanket promotion.

A separate exact CPU three-bin audit finds a coordinate-ST gradient limitation:
for reward support[-1,0,1],probabilities[.25,.5,.25],target0,expected ST gradient
is zero while the true score gradient is[.0624063,-.1248126,.0624063]. Finite
differences verify it. This is a conditional-head counterexample,not a causal
explanation of the full learner's regression. An unimplemented score-function
estimator matches the exact conditional gradient in two CPU fixtures; no new
native path or learner is selected.

`branch-protocol.json` separately declares only this final checkpoint's frozen
history16/32,eight-draw,84-group comparison,648 pins. No recollection,old-model
rerun or learning. The original25% energy-reduction gate is unchanged; a CPU
oracle feasibility check gives baseline.169502219,threshold.127126664 and exact
finite-oracle minimum.004211016. This is not a population calibration bound.
A separate RAM-reference detector audit confirms no above-top-brick endpoints
and some finite-coordinate balls outside detector support. Preserve these limits.


## Closed comparison

All declared stages are closed; see the [complete result](../results/2026-11-07-breakout-world-energy.md).
The final frozen comparison improves ending Brier.144566→.006129 and last-life
RIGHT ending probability50.211%→97.182%,but precise tracking regresses and primary
ball energy improves10.45%,failing25%. Non-ending group means pass5% while
individual assigned probabilities worsen. Native training costs86.58% more.
Retain both models as different development capabilities,not a replacement.
All77/89/648 phase pins,five guards/sealed audits,204 frozen tensors and18 restore
arrays pass. Initial CPU failures and original prototype failures remain intact.
No new native correction,learner,collector or GPU successor is selected. The full
world goal remains active/unproven; actor/critic stays parked.


## Qualification reporting correction

The original publication/build receipt's41 affected/41 packaged Python checks
were incorrectly copied from the pointwise prototype. The retained energy logs
contain27/27. Original pinned receipt/protocol/source remain unchanged; the
separate CPU count audit and [result correction](../results/2026-11-07-breakout-world-energy.md#python-verification-count-correction)
preserve the unmet41-count claim. No old stage was rerun or retroactively passed.
