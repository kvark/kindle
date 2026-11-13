# Fresh 200k Breakout RGB96 comparison

Select one fresh learner to test whether increased visual spatial capacity earns
enough frozen learning to offset its cost. Existing ball/motion readouts and
the unresolved Breakout gap motivate this hypothesis; they do not establish a
perceptual cause. The [bounded check](../results/2026-11-06-cdp-rgb96.md#bounded-check-complete)
passes: 8,192 actions/489 updates take66.690s versus47.690s retained,39.842% more
time. The padded200k estimate29m51s passes the previously declared35-minute cap;
conservative full-replay headroom3.757GiB passes2GiB. This is a separately selected
learning test. No bounded experience transfers and no completed stage repeats.

Use native source1279a415, SHA256
7704d25c9df5d37c48d5223f2d12c1d5d3c0a812c1f68cf393c87a637d010ab0,
with the exact bounded runtime/runner/teacher and qualified reference-repair
receipt. Its two test-only repairs do not change the production library.
Set observation_kind=rgb96. One GPU resize consumes native pooled RGB; no CPU
downscale/upscale, coordinates, crop, frame pair, object labels or emulator state.
CNN spatial cells grow16→36 and embedding1024→2304. CNN weights and RSSM/
actor/critic widths stay fixed, while posterior input, both CDP predictor layers
and four disagreement output heads grow. There are24,862,993 trainable parameters
versus16,334,353:8,528,640 more. Retain357 tensor names with42 changed shapes
including optimizer moments. This tests geometry, capacity and cost together;
it cannot isolate resolution or establish a perceptual explanation.

Keep centered CDP12M/Auto/N8/B16/T16/H15/R16,32×16, base/encoder4e-5,
dynamics4e-4, behavior8e-5, critic1/1, expert100/unit weights/world gradients/
all-state corrective labels,20% episode guidance/cap1920, sampled collection/
posteriors, actor/latent mixing.01 and replay100k. Actor256 and all optional actor
objectives stay off. Full18/sticky.25/repeat4/noops0 remain. Only observation_kind
changes the effective config; replay-value bootstrap explicitly records its
existing imagined default. Native/runner hashes differ from retained447fe50;
scoped qualification does not isolate every runtime difference.

Start fresh at seed1009/evaluation4000001009 with no transferred weights.
These are reused development choices. Train200,000 aggregate actions/12,477
updates within2100s; retain100k/6,227 recovery with read-only files. First update
is at376 actions; final debt.5. Require all357 initial tensors and47 pre-update
events to match this candidate's bounded run. RGB64 initial tensors/actions need
not match: compare against the candidate's own actual-initial controls. Audit
config,42 changed tensor shapes, finite reports, both weighted loss totals, actor
sharing, changed encoder/reward/continuation/actor/critic and component times.
Direct-gradient and expected-value actor metrics must remain absent.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm. Guidance/labels/learning off;
357 invariant tensors and exact96px restore geometry. Keep600k-action,
100k-frame-episode and300s wall caps per arm. Retain full action/outcome/frame/
boundary replays and videos. Frozen logs lack live image hashes; do not claim
unrecorded image equality. No cap increase, episode substitution or stage repeat.

Greedy is primary. Favor the candidate only if raw greedy return and own-initial
gain per training minute improve over143.667/6.590132, and the10,000-resample
paired eight-lane gain/min95% interval has a positive lower bound (bootstrap
seed0). Report sampled secondary scores and paired raw differences. Original
gate stays mean864 and90% two-wall. Reused lanes do not prove seed reliability.

Require complete training pixel/action/label/override replay, guards, curves,
exposure/score-range accounting,100k recovery observation and terminal collection.
CPU observer time stays in measured training wall time. Original Python and two
native fixture failures remain failed; separately corrected tests and bounded
stages remain closed. Preserve new failures. A separately justified CPU repair
may complete missing evidence without repeating GPU work or relabeling a failed
protocol. Preflight the changed helpers and all five commands without native/
emulator construction, including wrong geometry and corrupted loss rejection.

Root /x/Temp/kindle-rgb96-20261106/full; service
kindle-breakout-rgb96-200k-20261106. One GPU allocation at a time under the
existing host guard. Stop on real faults or numerical errors; no driver
reset/reload, extension, second learner, sweep or automatic successor.
Controller7200s, terminal collector7500s, training replay900s and frozen
replay500s caps. An observer timeout is not a stopped job: inspect original
identity. Retain RGB64 pending evidence; the five-game goal remains unfinished.
