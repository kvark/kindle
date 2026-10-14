# Reproduce the convolution group in its actual learner allocation

The [intermediate study](../results/2026-10-14-middle-convolution.md) finds a
4.163ms isolated dX/dW sum versus an8.085ms mixed interval in a trace that
fails its overhead gate. The grouping control is negative. Isolated tests also
set `no_alias`, which makes all buffers host-visible; the learner uses device-only
intermediates. That is a placement difference, not evidence it causes latency.
Artifacts: `runs/captured-convolution-group-20261014.FeH2MbcF`.

Keep image candidatee1a2407c's graphs, shaders, arithmetic, precision, weights
and original allocation. Add diagnostic access to existing buffer pieces and
cached group ranges; no allocation-policy, kernel or production change. One
original fixed-batch update prepares the same learner as prior diagnostics;
then disable optimizers/clipping and retain exact292 before/after tensors.

Measure the fixed world graph once normally and once as prefix/target/suffix,
16 warmup+64 measured calls each. The target is the original five-dispatch
group3974..3979: addition, two partial reductions, dX and dW partials. Preserve
all producers and dependencies; no masking. This three-section fixed-session
experiment is not a retry of the failed49-section full-learner trace. Require
exact full outputs and report its own3% perturbation gate; if it fails, reject
ordinary component attribution without repeating either timing.

Execute the prefix and snapshot all seven target inputs before alias reuse,
including raw BF16 bits. Record logical sizes, physical allocation/offset and
requested placement/runtime host-visibility policy, not queried Vulkan memory
types; verify disjoint input ranges. Save exact target outputs before
the suffix. Replay the same cached target using these original bindings, restoring
every input from GPU snapshots before each call. Then substitute bounded synthetic
inputs in the same bindings, matching the prior primitive's dy/weight/input
sequence. Each gets16 warmup+64 measured calls. Time restoration separately;
this deliberately changes cache history and is not ordinary learner throughput.
Require exact captured-input target outputs and finite synthetic outputs. Restore
all captured bytes, execute the full graph and require exact full outputs and
all292 saved tensors. Export raw snapshots for independent offline analysis.

This distinguishes omitted operations, real values and original allocation
context without a kernel/grouping/dtype/shape sweep. It does not alone isolate
memory-placement causality or guarantee a GPU instruction explanation. Preserve
the prior failed overhead/auxiliary-F32 controls; no JAX retry or Atari work.

One300-second guarded GPU process,330-second persistent service; Restart=no,
KillMode=control-group, no CPU quota, expected5080 and >=2GiB sampled Vulkan
headroom, allocation warnings retained. CPU preparation/review1CPU/2GiB/zero
swap. Review every failure, no blind retries. Upstream99e3de27/e7c3f7b rechecked
unchanged at21:46 UTC; no new relevant fix. No production repin or goal closure.
