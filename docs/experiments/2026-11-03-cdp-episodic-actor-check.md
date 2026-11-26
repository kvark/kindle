# Paired bounded Breakout check for completed-return actor learning

The optional implementation at e3b02163 passes163+5 Rust CPU tests,1006 packaged
Python tests,strict workspace/binding Clippy and12 native F64 loss/gradient
cases. The native fixture preserves actor/critic parameters and passes221 input
pins and host guards. Use its existing isolated runtime without rebuilding.

Select three sequential phases: legacy frozen compatibility, a fresh bonus-off
control, then a fresh bonus-off completed-return actor candidate. No overlap.
First restore retained200k policy1/replay-value1 and take1024 greedy actions
using its original evaluation seed. Require zero updates,all357 unchanged
tensors and exact128 original greedy semantic events before any training phase.

Each new arm trains8192 actions under150s, then restores for1024 greedy frozen
actions under60s. Require489 updates,first update376 actions,finite reports,
changed CNN/reward/actor/value parameters,finite checkpoint arrays and invariant
frozen arrays. Use seed1009/evaluation4000001009,12M64/Auto/N8/B16/T16/fullBPTT/
micro16/H15/R16/context1,encoder4e-5/behavior8e-5/dynamics4e-4,policy1,critic1/1,
constant expert100 with world gradients,all-state corrective labels and the
original teacher. Preserve20% episode overrides/cap1920,lambda.95/horizon333,
unimix.01/entropy.0003,replay100k/warmup1000,full18/sticky.25/repeat4/noops0,
raw rewards/native max-pooled RGB/one GPU64 resize. All other optional losses,
crop and imitation decay stay off. Neither arm reuses previous experience.

Both arms disable the model-derived disagreement bonus. The control omits
`replay_actor_scale`; the candidate uses100. Require identical initial arrays
between these arms and matching retained initial arrays after removing only
exploration-ensemble tensors. No new actor parameters are introduced. Derive
the expected reduced schema from those named omissions before launch. Compare
all common pre-update semantic transitions; no later equality is required.

For the candidate, require nonzero completed and positive replay-row counts,
finite nonnegative raw loss,weight sum bounded by positive rows,and exact loss
coefficient accounting. Rows without a completed true terminal stay masked;
no positive targets are required before one exists. Existing native fixtures
cover stream/reset/timeout/eviction/prefix alignment. Replay each full guided
training trajectory once on CPU, verifying pixels/actions/labels and recorded
override counts. Preserve native/runner identity,host guards and ROM pins.

Measure both training times and their relative difference. Require conservative
candidate projection `seconds * 200000 / 8192 * 1.10 <= 1800` before considering
a full learner. Short frozen actions need not finish an episode and do not
measure competence. These checks do not select a full run,seed confirmation,
coefficient sweep or cap increase. A future loss comparison needs the bonus-off
control because both differ from the retained action-effects1 recipe.

Pin commands,source/runtime,expected schema,original retained evidence and
protocol before execution. Validate each completed phase before starting the
next; preserve failures and never repeat a successful stage. Root
`/x/Temp/kindle-episodic-actor-20261103/check`. Keep all earlier checkpoints,
results and readers unchanged. Check live state before launch. The five-game
goal remains unfinished.
