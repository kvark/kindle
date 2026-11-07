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
