# Fresh200k Breakout enumerated one-step actor comparison

Select one fresh learner after the [bounded check](../results/2026-11-06-cdp-actor-expected-value.md#bounded-learning-cost-check-complete)
passes native correctness, learning, restore, replay, cost and memory gates.
The8,192-action run takes62.640s versus retained47.690s,31.350% more observed time.
Its padded200k projection is1682.233s, with9.702GiB conservative full-replay
headroom. The full comparison tests whether frozen learning gain offsets this
extra cost. No completed learner or bounded experience is reused.

Use source `f2b80fe596ae553bbea3cc7a3c638ea55529dc22`, native SHA256
`961f79583c1e2384f5dd2daba56fc2357c1a8dc280abd6bb8aef4d35fbf8b069`,
with the exact bounded runner/runtime/teacher. Set `actor_expected_value=true`.
Replace sampled-action policy loss with a probability-weighted expectation of
detached one-step values over all18 actions at replay posterior starts. Values
include reward, existing action-effects bonus and discounted continuation×critic;
one shared latent draw is reused across actions. Entropy, imitation, critic and
ordinary imagination remain. No extra coefficient or learned parameter.
Horizon and start distribution change with the estimator; this does not isolate
categorical-gradient bias or reproduce continuous-control SuperMimic.

Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16,32×16, encoder/base4e-5,
dynamics4e-4, behavior8e-5, critic1/1, expert100/unit weights/world gradients/
all-state corrective labels,20% episode guidance/cap1920, sampled collection/
posteriors, latent/actor mixing.01 and replay100k. Actor256; coordinates, residual,
dynamics-gradient and completed-return actor options off. Full18/sticky.25/
repeat4/noops0, native pooled RGB and one GPU64 resize. Only the actor flag changes
the effective recipe; replay-value bootstrap records its existing imagined
default explicitly. Source hashes differ from retained447fe50; scoped checks
do not isolate every runtime difference. No rebuild or backend work.

Start fresh at seed1009/evaluation4000001009, with no transferred weights or
bounded experience. These are reused development choices. Train200,000 aggregate
actions/12,477 updates within1800s; retain100k/6,227 recovery,357 finite tensors
and read-only recovery files. All357 initial tensors and47 pre-update events
must match bounded and retained. Actual-initial sampled/greedy semantic traces
must also match retained: this option does not change inference. Audit exact
config, finite reports, both weighted totals, actor sharing and component times.
Require the expected-value metric and positive snapshot timing in every report;
the metric is already included in policy_loss and must not be added twice.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm. Guidance/labels/learning off;
357 invariant tensors, full action/outcome/frame/boundary replays and videos.
Keep600k-action,100k-frame-episode and300s wall caps per arm. No cap increase,
stage repeat or episode substitution. Frozen logs lack live image hashes;
replay cannot prove unrecorded image equality. Use actual own-initial controls.

Greedy is primary. Favor the candidate only if raw greedy return and own-initial
gain per training minute improve over143.667/6.590132, and the10,000-resample
paired eight-lane gain/min95% interval has a positive lower bound (bootstrap
seed0). Report sampled secondary scores and paired raw differences. Original
task gate stays mean864 and90% two-wall. Reused lanes do not prove seed reliability.

Require complete training pixel/action/label/override replay, guards, curves,
exposure/score-range accounting,100k recovery observation and terminal collection.
CPU observer time stays in measured training wall time. The original CPU fixture
compile failure remains preserved; corrected CPU, native and bounded stages pass.
Preserve new failures. A separately justified CPU repair may complete missing
evidence without repeating GPU work or relabeling a failed protocol.

Root `/x/Temp/kindle-actor-expected-value-20261106/full`; service
`kindle-breakout-actor-expected-value-200k-20261106`. One GPU allocation at a time
with the existing host guard. Stop on real faults or numerical errors; no driver
reset/reload, extension, second learner, sweep or automatic successor. Main
controller7200s, terminal collector7500s, training replay900s and frozen replay500s
caps. An observer timeout is not a stopped job: inspect original identity.
Five-game goal remains unfinished; keep the retained default pending evidence.
