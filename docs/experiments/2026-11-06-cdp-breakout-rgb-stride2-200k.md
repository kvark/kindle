# Fresh200k Breakout stride-two encoder comparison

Select one fresh learner after the [bounded check](../results/2026-11-06-cdp-rgb-stride2.md#completed-bounded-cost-check)
passes numerical, checkpoint, replay, cost and memory gates. The8,192-action run
takes41.940s versus saved47.690s,12.056% less wall time. Its padded200k projection
is1,126.327s with11.667GiB conservative full-replay headroom. This supports one
fixed-budget test of useful frozen learning per training minute; it does not
establish gameplay benefit. No completed learner or check experience is reused.

Use production source `22c262ea2a9b1eb2d221b227f1e2f022908de1ef`, native SHA256
`cde123e1aa0d68f61cd6111fe392d605d2ae6864de39b4349a139c82301f6b3e`,
and the exact bounded runner/runtime/teacher. Set only `rgb_stride2=true`.
Four5×5 stride-two convolutions replace encoder stride-one convolution/max-pool;
normalization, activation, channel counts and output shapes stay fixed. All
parameter names/shapes/counts stay16,334,353; replay, latent and actor/critic
dimensions are unchanged. Sampling and nonlinear reduction change, so initial
behavior and learning can differ despite identical weights. Decoder stays fixed.

Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16,32×16, encoder/base4e-5,
dynamics4e-4, behavior8e-5, critic1/1, expert100/unit weights/world gradients/
all-state corrective labels,20% episode guidance/cap1920, sampled collection/
posteriors, latent/actor mixing.01 and replay100k. Actor256; coordinates, residual,
dynamics-gradient, expected-value and completed-return actor options off.
Full18/sticky.25/repeat4/noops0, native max-pooled RGB and one GPU64 resize.
Replay-value bootstrap records its existing imagined default explicitly.
Native/runner differ from retained447fe50; scoped checks do not isolate every
runtime difference. No rebuild, backend change or neighboring-worktree work.

Start fresh at seed1009/evaluation4000001009, without transferred weights or
bounded experience. These are reused development choices. Train200,000 aggregate
actions/12,477 updates within1,500s. Retain100k/6,227 recovery with357 finite
tensors and read-only recovery files. All357 initial tensors must equal bounded
and retained. All47 pre-update events must match the same-stride bounded run;
retained initial actions may differ. Describe those differences and use actual
own-initial frozen controls. Audit exact config, finite reports, both weighted
loss totals, actor sharing and component times. Optional actor objectives stay off.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm. Guidance/labels/learning off;
357 invariant tensors, full action/outcome/frame/boundary replay and videos.
Keep600k-action,100k-frame-episode and300s wall caps per arm. No cap increase,
stage repeat or episode substitution. Frozen logs lack live image hashes;
replay cannot prove unrecorded image equality.

Greedy is primary. Favor the candidate only if raw greedy return and own-initial
gain per training minute improve over143.667/6.590132, and the10,000-resample
paired eight-lane gain/min95% interval has a positive lower bound (bootstrap
seed0). Report sampled secondary scores and paired raw differences. Original
task gate stays mean864 and90% two-wall. Reused lanes do not establish seed
reliability; passing this development comparison would need fresh confirmation.

Require complete training pixel/action/label/override replay, host guards, curves,
exposure/score-range accounting,100k recovery observation and terminal collection.
CPU observer time stays in measured training wall time. Preserve the initial CPU
fixture compile failure and separate repair; native, compatibility and bounded
stages passed once. Preserve new failures. A separately justified CPU repair may
complete missing evidence without repeating GPU work or relabeling a failed protocol.

Root `/x/Temp/kindle-rgb-stride2-20261106/full`; service
`kindle-breakout-rgb-stride2-200k-20261106`. One GPU allocation at a time with
the existing host guard and live device/process identity checks. Stop on real
faults/nonfinite results; no reset/reload/reboot, extension, second learner,
sweep or automatic successor. Main controller7200s, terminal collector7500s,
training replay900s and frozen replay500s caps. An observer timeout is not a
stopped job: inspect original identity. Five-game goal remains unfinished;
keep the retained default pending evidence.
