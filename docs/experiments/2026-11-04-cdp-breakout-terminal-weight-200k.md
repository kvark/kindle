# Fresh Breakout200k comparison with terminal-only continuation weighting

Select one fresh200k-action learner after the [bounded weight10 check](../results/2026-11-04-cdp-breakout-terminal-weight.md)
passes execution, native loss/gradient verification and its30-minute cost gate.
Training8,192 actions/489 updates takes46.934s; the conservative200k projection
is21m00.438s. The short frozen capture completes no natural episodes. This separate
full comparison tests gameplay within the original five-game goal. No success is
inferred from finite losses, changed weights or early timing.

The retained continuation head poorly recognizes true endings, which are rare in
replay. Whole-continuation scale10 regresses gameplay. Terminal-biased replay.25
improves selected ending predictions but worsens intermediate-life predictions
and gameplay; it changes sampling for every objective. Terminal-only weighting
is a distinct test: increase terminal continuation gradients while keeping
continuing-row weights and the existing sampler. Neither the best weight nor
calibrated probabilities are assumed.

## Fixed recipe and source qualification

Keep centeredCDP12M/RGB64/Auto/N8/B16/T16/full-BPTT/micro16/H15/R16,
base/encoder4e-5, dynamics4e-4, behavior8e-5, replay-value1, continuation1,
expert100/unit weights/world gradients, action-effects1, replay100k/context1,
sampled collection/posteriors, actor-unimix.01, full18/sticky.25/repeat4/noops0.
Keep native pooled RGB with one GPU resize, all-state corrective labels and
20% episode guidance/cap1920. Reuse development seed1009. Start fresh; no bounded
weights, replay or experience are transferred.

Set `terminal_continuation_weight=10`: true terminal targets0 receive weight10;
continuing targets gamma receive1, including intermediate life losses. Retain
the ordinary row denominator and continuation scale1. Terminal replay mixture
and open-loop supervision stay off. Other objectives retain their explicit
weights and sampling; shared parameter updates and later experience can change.
The enabled stable-logit BCE differs numerically from the older sigmoid/BCE path,
so this does not isolate only the coefficient or establish probability calibration.

Reuse qualified native source3672a21a, SHA256
`a8b483a3229e1c57c39d8742c36a3dc12edefcb8bc7c705306c9bf32bc85f8e2`, from
`/x/Temp/kindle-terminal-weight-20261104/runtime`, and its bounded runner bytes.
No rebuild or optional runtime mixing. Native and runner differ from retained447fe50.
Packaged teacher/scheduler files are byte-identical to the terminal-mixture package;
reuse its recorded AST/4,001-step full18 episode-guidance qualification by verifying
both source sets and the original receipt. Those wrappers differ from the retained
package; their block/minimal modes remain off. This scoped evidence does not prove
every possible trajectory equivalent. Bounded default-off compatibility matches
128 semantic events/357 invariant tensors; fresh initialization matches357 tensors
and47 pre-update events. Original CPU qualification failures remain preserved.

## Budget, recovery and assessments

Train exactly200,000 aggregate actions/12,477 updates with a1,800-second training
cap. Require first update376 and residual credit.5. Save actual initialization
and100k/6,227 and200k/12,477 checkpoints. The100k observer verifies finite weights
and moments and makes that recovery directory/files read-only. A second CPU
reader verifies it once while preserving the original live learner identity.
Recovery omits replay, games, live beliefs and complete RNG state.

After training and its audit, run final sampled, actual-initial sampled, final
greedy and actual-initial greedy frozen arms, in that order. Each selects the
firstthree natural episodes per each ofeight lanes,24 episodes, seed4000001009,
with guidance, labels and updates off. Keep600k-action/300-second arm caps and
the existing100k-frame episode cap. Require zero truncations and357 invariant
tensors per arm. Preserve cutoffs/failures; do not raise caps, repeat completed
arms, extend training or change episode selection based on outcomes.

Require bounded/retained initial equality, pre-update semantics, both actual-initial
semantic comparisons, finite reports, weighted world/behavior totals and changed
CNN/reward/continuation/actor/value parameters. Derive integer terminal counts
from continuation-target means and report total/50k-window exposure; no mixture
sampling report may appear. Preserve full training pixel/action/label replay,
frozen recorded-action/outcome/frame/boundary replays/videos, coverage, guidance
exposure, curves and host guards. Frozen logs have no live pixel hashes; do not
claim equality to unrecorded images. Stage audits gate subsequent GPU work.
Training replay cap900s, frozen replay500s, main7200s, terminal observer7500s.
CPU observer costs remain included without wall-time correction.

## Locked comparison

Reuse retained closed control evidence: sampled70.750/greedy143.666667, own
initials1.416667/2.208333, all0/24 two-wall,200k/12,477 in1287.910495s. Greedy
return is primary; sampled secondary. Compare absolute return and each arm's
own-initial-subtracted return per training minute. Use paired eight-lane bootstrap,
10,000 resamples/seed0. Favor the candidate only if primary return and gain/min
both rise and the gain/min difference interval's lower endpoint exceeds zero.
Report both modes, all absolute scores and original mean864/90%-two-wall gate.

One reused development seed/cohort, historical control timing, native/runner/
guidance-wrapper differences and changed experience limit causal and reliability
claims. No weight sweep, second learner, unchanged extension or automatic successor
follows. Root `/x/Temp/kindle-terminal-weight-20261104/full`; service
`kindle-breakout-terminal-weight-200k-20261104`. Require helper/comparison checks
and all five actual queue preflights before launch on an idle host. The five-game
goal remains unfinished.
