# Current actor imitation remains dominant after conditional optimizer attribution

The CPU follow-up to the [current raw-gradient diagnostic](2026-11-03-cdp-breakout-actor-gradients.md)
finds weighted imitation contributions **7,493× /10,928×** the imagined
score-function contribution on two fixed batches. Clipping and preconditioning
do not remove this imbalance in the current-batch estimate. Saved momentum also
contributes substantially to the full estimated update. These are F64 conditional
estimates, not measured native parameter movement or proof of a gameplay cause.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-actor-optimizer.md)
completes on source `71840bbc29a43bb3b016b24480ae66cb37df7328` at **10:34:37.510 UTC
November 3**. Analysis takes **1.027s CPU** (1.088s wrapper); terminal collection
passes at **10:36:51.527**, taking 1.928s. All 13 fixtures, 210 current pins and
158 source pins pass. No native import, GPU work, emulator work, learner update,
failed stage or repeated stage occurs. Evidence is retained at
`/x/Temp/kindle-actor-optimizer-20261103`; analysis and terminal readers are closed.

## What was calculated

Both batches independently start from the same saved 22 behavior parameters
(1,646,097 elements) and their 44 first/second moments at update 12,477. Neither
estimate is applied or fed into the other. The complete source gradient matrices
and checkpoint are reused without another backward pass or capture.

The private analyzer retains the qualified AGC/LaProp arithmetic and pins native
`447fe50` optimizer setup plus Meganeura `423d373` runtime/shaders. Behavior rate
is 8e-5, with warmup complete and actor updates enabled; the base/world rate4e-5
does not set this rate. AGC is .3 with minimum parameter norm .001; LaProp uses
beta1=.9, beta2=.999, epsilon1e-20 and zero weight decay.

For each tensor, clipping and the new variance denominator are calculated from
the **complete total gradient** and held fixed when transforming components.
History plus current contribution reconstructs the complete estimate exactly.
The component estimates are not separate optimizer steps, nor predictions for
changing a loss weight. Norms are not additive percentages.

## Actor results

| Measurement | Batch 0 | Batch 1 |
| --- | ---: | ---: |
| Parameters clipped | 11/11 | 10/11 |
| Median AGC multiplier | .022414 | .065585 |
| Raw total gradient norm | 305.301907 | 251.756398 |
| Clipped total gradient norm | 12.415027 | 11.965164 |
| Full estimated update norm | .019410 | .018230 |
| Saved-history contribution norm | .015340 | .015340 |
| Current-batch contribution norm | .011053 | .010340 |
| Conditional imitation contribution norm | .011053 | .010340 |
| Conditional score-function contribution norm | 1.47517e-6 | 9.46277e-7 |
| Conditional entropy contribution norm | 1.45166e-8 | 1.60396e-8 |
| Imitation / score-function norm | 7,492.750 | 10,927.503 |
| Current / imitation cosine | .999999991 | .999999996 |
| Full / saved-history cosine | .822655 | .823756 |
| Full / current cosine | .614355 | .540962 |
| Saved-history / current cosine | .056807 | −.031207 |
| Full update / weight norm | .046902% | .044051% |

Raw imitation/score ratios were 6,721/10,925. Conditional transformations retain
the large imbalance. The score-function/full-update norm ratios are .007600%
and .005191%; these are norm ratios, not fractions of an additive update budget.
Mixed saved momentum cannot be attributed to individual objectives here.

## Critic results

Four of 11 critic parameters clip in each batch. Conditional imagined-value
contribution norms are .006328/.006550 and replay-value norms .011623/.010242,
with positive cosines .828789/.614571. Full update norms are .022881/.021274;
history contributes .015810 in each independent estimate. Full update/weight
ratios are .047405%/.044077%. These measurements do not establish which critic
objective improves gameplay.

## Verification and limits

All raw per-parameter/group dot matrices reproduce the native saved matrices
with rtol1e-9/atol1e-8. Maximum group component-sum relative L2 is 2.8272e-8;
maximum absolute error is 1.2669e-11. All parameters, moments, gradients and
estimates are finite; saved variances have no zero or subnormal values. Actor
and critic cross-head derivatives remain zero. Inputs and independent starting
history remain unchanged.

Fixtures cover bias correction, saved momentum, cancellation before clipping,
zero gradients, invalid states, input invariance, behavior-rate fallback/warmup,
exact parameter prefixes and actor freezing separately from critic updates.
Native, JAX and PyTorch imports are blocked; zero native import attempts and no
open GPU descriptors are recorded. Five original arithmetic functions are
AST-identical to the earlier qualified CPU implementation.

F64 does not emulate native reduction order, rounding or parameter writes. The
earlier world-optimizer native comparison remains protocol-failed. These are
two reused final-checkpoint sampled-history batches with fresh diagnostic draws,
not original replay minibatches, early-training evidence or independent seeds.
The result does not explain the policy100 regression causally.

Retain policy1/expert100/critic1/1 and the current gameplay control. The selected
next development is a default-off **actor-only imitation decay**, preserving
world-model imitation, with numerical and short-run qualification before any
full learner. The earlier shared decay reduced both objectives on an older
recipe and lowered frozen scores; it does not test this separation. No full
learner, sweep or extension is selected by this diagnostic. The five-game goal
remains unfinished.

Key SHA256 receipts:

- Result: `4f08d5d7c01a6c5e19399af62460784f54d51c56b4e2d77feafce70c2eb96642`.
- Completion: `ff81959f17d39d9c43d73880a6a1ec8932c77b84787bb19f18b6504a7b2187f7`.
- Declaration: `c2c5bc7f149ebde301b85ba1f747fb49b42b38cb53e11ec997a59b2c509e2ace`.
