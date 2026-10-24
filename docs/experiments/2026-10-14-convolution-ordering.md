# Separate ordering from explicit storage visibility at the independent boundary

The [single-boundary intervention](../results/2026-10-14-convolution-boundary.md)
reduces the real group7.387→4.168ms and complete RGB12M/B32 updates5.29%, with
exact reports/states. One ordinary storage barrier changes both execution and
memory-access scopes. This follow-up asks whether explicitly including storage
accesses is necessary for that local benefit; it is not a barrier-API sweep or
a retry of the negative fully dependent-chain study.

Artifacts: `runs/convolution-ordering-20261014.T8A1WKAK`. Preserve the same
captured292-tensor state, seven inputs, original allocation mapping, all4,023
dispatches/pipeline keys and dtype/arithmetic. Add one isolated Vulkan diagnostic
method calling the same `vkCmdPipelineBarrier`, compute→compute, but with zero
memory/buffer/image barriers. The [Vulkan contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier.html)
retains the execution scopes while empty barrier arrays specify empty access
scopes. Use this only between the proven-independent auxiliary/convolution
halves; keep all original data-dependency and restoration barriers unchanged.
Do not replace a real memory dependency with an execution-only dependency.

Replay three arms in the same session: intact group, one ordinary storage
boundary, one execution-only boundary. The latter two use identical diagnostic
recording and one cached pass; clear the command cache when changing barrier
semantics because its key is only the dispatch range. Every call restores all
seven raw inputs, separately timed. Require exact five outputs after each arm,
unchanged before/after292 tensors, original descriptors/pipelines and requested
allocation regions. First a logged-validation smoke with one warmup/two calls
per arm; then a fresh validation-disabled timing process with16 warmup/64
measured calls per arm. Nine plus240 optimizer-free group replays, zero learner
updates/game actions. Retain all samples and every failure; no GPU retry.

The ordinary boundary must reproduce at least30% improvement over intact for
interpretation. Execution-only must retain at least30% improvement and be no
more than5% slower than the ordinary boundary to establish that explicit
storage access scopes are unnecessary for this benefit. Otherwise the scope
distinction remains meaningful/unresolved; no relaxation or variant sweep.
This observes API-level semantics, not cache counters: a driver may perform
cache operations even for an execution-only dependency. No new whole-update
speed claim, general production policy, JAX capture or Atari restart follows
automatically. Use the result to design the general scheduling intervention.

Two300-second host-guarded jobs, separate330-second persistent services,
Restart=no/KillMode=control-group, no GPU CPU quota. Expected5080 and >=2GiB
sampled Vulkan headroom; record allocation warnings. CPU preparation/review
1CPU/2GiB/zero swap. Review smoke before timing. No compiler/kernel changes,
new device features, reset/recovery or separate NVML polling.
