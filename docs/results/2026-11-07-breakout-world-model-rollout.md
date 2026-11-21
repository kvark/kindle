# Direct future-image training improves Breakout fitting forecasts

Direct RGB supervision through prior rollouts improves eight-step fitting ball
accuracy from1/8 to4–5/8, while sixteen-step accuracy remains2/8 on the standard
sticky-action clips. On separate sticky-off fitting clips, four-step forecasts
reach8/8, eight-step4/4 measurable balls, and sixteen-step7–8/8 across four fixed
draws. **This demonstrates short-trajectory fitting, not a qualified Breakout
world model.** No actor or critic trains.

## Change and controls

Continue the [isolated pixel-world investigation](2026-11-07-breakout-world-model-capacity.md).
Keep the8,852,675-parameter world model from the12M RSSM preset, RGB64, Auto,
B8/T32, flat4e-4 rates and reconstruction100. Add direct pixel squared error
from16 prior transitions after16 observed arrivals. The same decoder is used
for observed states and imagined states. Average squared error over batch and
horizon, sum image dimensions, then apply the reconstruction100 coefficient.
The ordinary posterior reconstruction, reward/continuation and KL losses stay.

Future frames are detached loss targets. Only the observed-prefix state and
future actions drive the prior rollout. Native categorical samples use the
existing straight-through estimator; their separate RNG does not consume the
posterior-sampling stream. `prediction_length=0` retains the previous objective.
The setting is stored in the standalone checkpoint metadata. Checkpoints retain
weights, optimizer moments and update count, but not the complete training RNG.

Both new learners are fresh, reuse seed41001 and run exactly2,048 updates on
eight32-arrival contact clips. The standard comparison reuses the exact previous
clips. All204 initial tensors and all four initial prediction traces match the
reconstruction100 control exactly. No collector, old learner or assessment is
restarted. Four diagnostic draws are not independent learners or episodes.

The separate debug learner uses the already-retained sticky-off corpus. It has
different collection seeds, actions and trajectories. This tests whether fitting
failure also occurs without sticky actions; it is not a paired causal estimate
of the effect of sticky actions.

## Fixed final results

Ball accuracy means a detected ball within one resized pixel of the real image's
detected ball. Missing predictions count as failures. The below-brick detector
does not cover upper-wall balls or every near-paddle frame. Its location error
against native pooled-image centroids is checked on38 supported fitting targets:
mean.229,95th percentile.366 and maximum.386 resized pixels.

| Standard sticky.25 fitting clips | RGB100 control | RGB100 + prior16 |
| --- | ---: | ---: |
| Observed-state reconstruction |6/8|6/8|
|1-decision forecast |6/8|6/8|
|4-decision forecast |3/6|4/6|
|8-decision forecast |1/8|5/8 on three draws;4/8 on one |
|16-decision forecast |1–2/8|2/8 on all four draws |
| Training-call time |130.495s|186.404s|

Direct supervision costs42.84% more training-call time in this fitting test.
Its loop including periodic evaluation/checkpoints takes216.029s. The observed
improvement supports training the tested future predictions directly, without
establishing broad reliability, CDP superiority or a gameplay benefit.

| Separate sticky-off fitting clips | RGB100 + prior16, four fixed draws |
| --- | ---: |
| Observed-state reconstruction |3/8 on every draw |
|1-decision forecast |8/8,7/8,7/8,6/8|
|4-decision forecast |8/8 on every draw |
|8-decision forecast |4/4 supported targets on every draw; four targets untested |
|16-decision forecast |8/8,7/8,7/8,7/8|
| Training-call time / loop time |186.962s /218.377s|

The poor observed-state reconstruction beside better future predictions is a
remaining inconsistency. An image decoder omitting the ball does not alone prove
that the latent lacks ball information. The cause is not isolated here. Fitting
eight trajectories can exploit their regularities; it does not establish learned
physics on new histories or action interventions.

## Next requirement

Move to full-corpus fixed-data training with development curves and separately
held-out episode families. Before that run, support natural reset crossings by
masking prior-image losses after the first future reset while keeping the
terminal arrival. Future boundary labels must not reset imagined state or feed
future observations into it. The current capacity path deliberately rejects
such crossings; simply filtering them out would underrepresent game endings.

Then measure full-field state/dynamics, contacts and outgoing motion, brick
changes, reward and life/game-over events, action branches and32-step forecasts.
Retain natural-frequency and challenge-set results separately. Later-game
coverage and fresh-initialization confirmation remain required. The world-model
goal is active and unproven; no full-corpus learner has launched by this report.

## Evidence and validation

Root `/mnt/data/kindle-breakout-world-rollout-20261107`, source`25b3328b`, native
SHA256`547f065f5c2c39e808ff7aaee4035af2f29317fdb2c48093ab40a23380ecc437`.
Thirty CPU world/regression tests pass, with18 GPU tests ignored, alongside53
Python analysis/regression tests and strict workspace/binding Clippy. A new
two-step tiny GPU fixture passes eight updates, reset rejection, weighted-loss
arithmetic, frozen weights/moments and exact checkpoint prediction restore.
The CPU graph check excludes future targets from rollout states and confirms
prior/RSSM/decoder gradient connectivity. Existing operations and estimator are
reused; no new backend primitive is introduced.

All three guarded phases complete successfully with no host recovery. CPU review
checks283 source/native/config/data pins, both complete2,048-report ledgers,
finite204-tensor checkpoints, loss totals within F32 tolerance, matched initial
tensors and initial forecasts. All267 source files are archived and verified
before further edits. No numerical or GPU execution failure occurs in this stage.

Artifacts: `protocol.json`, `debug-declaration.json`, `review.json`,
`source-25b3328b.tar.gz`, per-stage results/curves/checkpoints and fixed forecast
arrays. Visuals: `analysis/eight-step-comparison.png`,
`analysis/capacity-curves.png`. Earlier detector validation remains at
`/mnt/data/kindle-breakout-world-model-20261107/analysis/detector-review.json`.
