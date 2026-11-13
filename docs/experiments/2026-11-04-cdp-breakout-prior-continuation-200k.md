# Fresh Breakout200k comparison with one-step prior continuation supervision

Select one fresh200k-action learner after the [bounded prior-continuation check](../results/2026-11-04-cdp-breakout-prior-continuation.md)
passes numerical/gradient checks, execution and the30-minute cost gate. Its8,192
actions/489 updates take47.016s; the conservative200k projection is21m32.945s.
The short frozen capture has no completed natural episodes. This separate full
comparison tests gameplay; finite losses and early cost do not establish benefit.

The observed-versus-predicted ending gap motivates direct continuation supervision
on one-step prior states. The earlier three-step joint feature/reward/continuation
loss regressed and is not repeated. Terminal weighting also failed to improve
gameplay; its probability shift and lower prior ranking do not establish a causal
limitation. This candidate retains equal row weights and the original sampler.

## Fixed recipe and source

Keep centeredCDP12M/RGB64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16,
base/encoder4e-5,dynamics4e-4,behavior8e-5,replay-value1,continuation1,
expert100/unit weights/world gradients,action-effects1,replay100k/context1,
sampled collection/posteriors,actor-unimix.01,full18/sticky.25/repeat4/noops0.
Keep native pooled RGB with one GPU64 resize, all-state corrective labels and
20% episode guidance/cap1920. Reuse development seed1009. Start fresh; no bounded
weights, replay, game state or experience are transferred.

Set `prior_continuation_scale=1`. The shared head receives sampled one-step prior
features and actual targets0 at true endings/gamma otherwise. Reset observations
are masked using the native keep-action input. The ordinary B*T denominator
includes masked zeros. Gradients reach the prior, dynamics and earlier posterior
states; the same-step arrival image/posterior is excluded. No additional parameters,
class weights, terminal replay mixture, output correction or multi-step auxiliary
targets. The posterior objective and other weights remain unchanged; shared
parameter updates and later experience can differ.

Reuse qualified native07873e7c, SHA256
`aeecf9e7d04cd27b03a9b8660e2d1531e8ca96d1a4c23c1f2e0ea623a5f1d254`,
from `/x/Temp/kindle-prior-continuation-20261104/runtime`, and its bounded runner.
No rebuild or runtime mixing. Native/runner differ from retained447fe50.
Teacher/scheduler bytes match the already qualified terminal-mixture/weight
packages; reuse the saved AST/4,001-step full18 episode-guidance qualification
after checking both source sets and its receipt. Their wrappers differ from the
retained package; block/minimal modes remain off. This is scoped qualification,
not broad equivalence. Bounded initialization matches357 tensors/47 pre-update
events; default-off frozen compatibility matches128 events/357 invariant tensors.
The original CPU centered-CDP/BPTT fixture rejection remains preserved.

## Budget, recovery and assessments

Train exactly200,000 aggregate actions/12,477 updates with a1,800-second training
cap. Require first update376 and residual credit.5. Save actual initialization
and100k/6,227 and200k/12,477 checkpoints. The100k observer verifies finite weights
and moments and makes that recovery directory/files read-only. A second CPU
reader verifies it once, preserving the original live process identity. Recovery
omits replay, games, live beliefs and complete RNG state.

After training and its audit, run final sampled, actual-initial sampled, final
greedy and actual-initial greedy frozen arms, in that order. Each selects the first
three natural episodes per each ofeight lanes,24 episodes, seed4000001009, with
guidance, labels and updates off. Keep600k-action/300-second arm caps and the
existing100k-frame episode cap. Require zero truncations and357 invariant tensors
per arm. Preserve failures and cutoffs; no cap increase, repeated completed arm,
training extension or outcome-dependent episode selection.

Require bounded/retained initial equality, pre-update semantics, both actual-initial
semantic comparisons, finite reports, weighted world/behavior totals and changed
CNN/reward/continuation/actor/value. Every world total includes scale1 times the
auxiliary prior loss. Report raw auxiliary mean, valid fraction and native-reported
masked rows overall and in fixed50k-action windows. Sampled row identities are
not logged; do not claim their independent reconstruction. Derive terminal exposure
from continuation-target means; no mixture-sampling report may appear.

Preserve full training pixel/action/label replay, frozen recorded-action/outcome/
frame/boundary replays/videos, coverage, guidance exposure, curves and host guards.
Frozen logs lack live pixel hashes; do not claim equality to unrecorded images.
Stage audits gate subsequent GPU work. Training replay cap900s, frozen replay500s,
main7200s, terminal observer7500s. CPU observer costs remain included in wall time.

## Locked comparison

Reuse retained sampled70.750/greedy143.666667, own initials1.416667/2.208333,
all0/24 two-wall,200k/12,477 in1287.910495s. Greedy is primary; sampled secondary.
Compare absolute return and each arm's own-initial-subtracted return per training
minute using paired eight-lane bootstrap,10,000 resamples/seed0. Favor the candidate
only if primary return and gain/min both rise and the gain/min difference
interval's lower endpoint exceeds zero. Report both modes and the original
mean864/90%-two-wall gate.

One reused development seed/cohort, historical control timing, native/runner/
wrapper differences and changed experience limit causal and reliability claims.
No coefficient sweep, second learner, unchanged extension or automatic successor.
Require helper/comparison checks and all five actual command preflights before
launch on an idle host. Root `/x/Temp/kindle-prior-continuation-20261104/full`;
service`kindle-breakout-prior-continuation-200k-20261104`. The five-game goal remains
unfinished.
