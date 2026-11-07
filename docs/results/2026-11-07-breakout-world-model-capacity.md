# Breakout world-only learning exposes a visual-detail and forecasting gap

The standalone native training path works, but **world-model competence is not
proven**. A small fitting test learns the court while largely omitting the ball.
Increasing reconstruction pressure improves observed-state and one-step ball
accuracy from1/8 to6/8; longer imagined trajectories still lose accuracy. These
measurements concern eight reused fitting clips, not held-out physics or play.

## What actually trains

`WorldModelLearner` uses Kindle's RGB CNN, categorical RSSM, RGB decoder, reward
and continuation heads on Meganeura + Blade. Its graph contains no actor/critic
parameters or targets, replay-value, imitation or intrinsic losses. Images and
incoming actions/rewards/terminals are time-major; each clipped context begins
with a reset. Forecasts receive observed history and future actions, with no
future-image argument. Learning and inference remain native.

The AMD tiny fixture completes eight updates. Every encoder/posterior/dynamics/
decoder/reward/continuation parameter family changes. All204 saved tensors,
including optimizer moments, remain exact during frozen predictions; restoring
the trained checkpoint reproduces those predictions exactly. This establishes
execution and checkpoint behavior, not learned game physics. The initial CPU
test-constructor failure and its passing repair remain separate logs. Initial
optimizer-moment materialization was corrected before GPU execution.

## Fixed data and capacity experiment

Eight independent environments collect native max-pooled RGB, followed by one
batched GPU64 resize. Retain native images as well as resized images. A fixed
pixel controller plus random command blocks supplies actions. RAM geometry and
life labels are saved only for analysis and fitting-clip selection; they do not
enter model inputs or losses. No policy or critic learns.

| Corpus | Sticky actions | Aggregate actions | Emulator frames | Recorded game-overs |
| --- | ---: | ---: | ---: | ---: |
| Standard fitting |.25|8,192|32,760|5|
| Separate deterministic debug |0|4,096|16,381|1|
| Fresh standard development |.25|4,096|16,380|3|

Select one32-arrival clip per standard fitting lane, with a paddle-height
vertical reversal after the16-arrival observed prefix. These are contact
candidates, not a validated exact-collision classifier. All clips stay within
their source episode. The first arrival is treated as a clipped-context reset;
its incoming action/reward are excluded. Development/debug data is not used by
these two learners, and no untouched test set is scored.

Both fresh learners use the12M RSSM preset, RGB64, Auto, B8/T32, full BPTT,
flat4e-4 base/encoder/dynamics rates, default dynamics1/representation.1/free-nat1
settings and2,048 updates on the same256 source arrivals. The sole configured
difference is reconstruction loss1→100. All204 initial tensors match exactly.
Training calls take129.214/130.495s; the loops including periodic evaluation and
checkpointing take159.027/162.429s. These timings exclude setup and initial
evaluation. This is a capacity check, not a general recipe or matched gameplay
comparison.

## Physical measurements

The conservative image detector measures the red ball below the brick region
at64x64. A missing/ambiguous prediction counts as a failure, rather than being
discarded from coordinate error. Targets support8 examples at observed/1/8/16
steps and6 at4 steps; the other two near-contact targets are unsupported by this
detector. Upper-wall ball positions are outside its scope. Four fixed categorical
draws are reported individually; they are not independent learners or episodes.

| Ball within1 resized pixel | Reconstruction1 | Reconstruction100 |
| --- | ---: | ---: |
| Observed-state reconstruction |1/8|6/8|
|1-decision prior forecast |1/8|6/8|
|4-decision prior forecast |0/6|3/6|
|8-decision prior forecast |0/8|1/8|
|16-decision prior forecast |0/8|1/8 on three draws;2/8 on one |

The earlier fully coupled RGB checkpoint is also inspected on these same clips,
using scalar re-observation under the new compatible runtime. All399 tensors
remain unchanged and there are zero updates. Its observed-state and all queried
prior images have zero accepted ball detections, despite visually reproducing
much of the court and brick layout. The rendered images were inspected. This
does not revise its original negative gameplay comparison or qualify all
properties of its latent representation.

The controlled fitting comparison supports an effect of reconstruction pressure
on visual precision in this setup. It does not establish that pixels outperform
CDP or that an accurate observed state yields an accurate imagined trajectory.
Both fitting models remain inadequate. Reward-event/termination competence,
brick transitions, action counterfactuals,32-step forecasts, later walls and
held-out generalization are still unproven.

## Next focused change

Train RGB predictions after the observed prefix ends, providing direct image
error through a bounded prior rollout. The current standard RSSM objective
supervises posterior reconstructions and matches priors through KL; it does not
directly optimize the tested future pixel trajectories. This next change is
selected for implementation, not yet trained or qualified by this report.
Do not restart the completed capacity runs or collector.

## Retained evidence

Root: `/mnt/data/kindle-breakout-world-model-20261107`. Source`9c1dcf89`, native
SHA256`f93bd82123f7dc3b20e9e6adf0beb353b2046b17ec835bf47e629f3fc8e6a4d3`.
All five guarded GPU children finish successfully, with no host recovery.
CPU review verifies271 source/native/config pins,24 corpus files, both complete
2,048-report ledgers, finite checkpoints, matched initial tensors and weighted
loss totals within declared F32 tolerance. It archives and verifies all266
source files before later edits; historical source bytes remain recoverable.

Key artifacts: `protocol.json`, `reconstruction100-declaration.json`,
`capacity-review.json`, `source-9c1dcf89.tar.gz`, per-stage `result.json`,
`eval-2048.json`, `forecast-2048.npz`, `learning.jsonl`, initial/final checkpoints,
`analysis/ball-comparison.png` and `analysis/contact-rollouts.gif`.
