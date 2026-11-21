# Fresh200k Breakout64×8 categorical grouping comparison

Select one fresh learner after the [bounded check](../results/2026-11-06-cdp-latent-shape.md)
passes cost, memory,357 initial tensor equality, learning, replay and restore.
The8,192-action run takes46.655s versus retained47.690s; its padded200k projection
is1,252.951s, with12,049,403,904 bytes of conservative headroom after subtracting
full replay. Short historical timing with changed experience is not a speedup
claim. This comparison tests useful frozen learning per training minute.

Use native source536012d5, SHA256
`2c9c6e6f40bbcaf33b0c982d09e3a130057894acbe2115f54acd77d0d0a2e8bd`,
with the exact bounded runner/runtime/teacher. Set `latent_shape={variables:64,
classes:8}` instead of32×16. Both have512 flattened entries and16,334,353 trainable
parameters. Sampling, active one-hot entries and KL sums change together; this
is not an isolated capacity/noise test or a remedy established by earlier probes.

Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16, encoder/base4e-5, dynamics4e-4,
behavior8e-5, critic1/1, expert100/unit weights/world gradients/all-state corrective
labels,20% episode guidance/cap1920, sampled collection/posteriors, latent/actor
mixing.01 and replay100k. Actor width256; residual/coordinates/actor dynamics off.
Full18 actions/sticky.25/repeat4/noops0, native pooled RGB and one GPU64 resize.
Only grouping changes the recipe; replay-value bootstrap explicitly records its
existing imagined default. Native/runner/teacher-module hashes differ from
retained447fe50; scoped qualification does not isolate every runtime difference.

Start fresh at seed1009/evaluation4000001009, transferring neither checkpoint
weights nor bounded experience. These are reused development choices. Train
200,000 aggregate actions/12,477 updates within1,800s. Keep100k/6,227 recovery,
357 finite tensors and read-only recovery files. All357 initial tensors and47
pre-update semantic events must match the new bounded run. All357 initial
tensors must also match retained, but retained initial action traces and frozen
scores may differ under changed grouping; describe those differences without
requiring equality. Audit config, finite reports, both weighted losses, actor
sharing and component times. No new loss, rebuild or backend change.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first3 natural episodes per8 lanes=24 per arm. Guidance/labels/learning off;
357 invariant tensors, complete action/outcome/frame/boundary replays and videos.
Keep600k-action,100k-frame-episode and300s wall caps per arm. No cap increases,
stage repeats or episode substitution. Frozen logs lack live image hashes;
replay cannot prove unrecorded image equality. Use own-initial controls.

Greedy is primary. Favor the candidate only if raw greedy return and own-initial
gain per training minute improve over143.667/6.590132, and the10,000-resample paired
eight-lane gain/min95% interval has a positive lower bound (bootstrap seed0).
Report sampled secondary scores and paired raw differences. The original task
gate remains mean864 and90% two-wall. Reused lanes do not establish seed reliability.

Require complete training pixel/action/label/override replay, guards, curves,
exposure/score-range accounting,100k recovery observation and terminal collection.
CPU observer time remains in actual training wall time. Native qualification's
original test readback failure remains failed; separate test repair completed
missing evidence without changing production native. Bounded stages pass with
no failures/repeats. Preserve any new failure; a separately declared CPU-only
repair may complete evidence without rerunning GPU stages or relabeling failure.

Root `/x/Temp/kindle-latent-shape-20261106/full`; service
`kindle-breakout-latent-shape-200k-20261106`. One GPU allocation at a time with
existing host guard; stop on real faults or numerical errors. No automatic reset,
reload, extension, second learner, sweep or successor. An observer timeout is
not a stopped job: inspect original process/invocation. Five-game goal unfinished.
