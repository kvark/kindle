# Current B32 command-buffer execution control

Declared October15,2026 at09:39UTC before new source or GPU work, after the
[complete native trace](../results/2026-10-15-whole-update-attribution.md).
Its47.699ms GPU-pass union rules out small helpers/host work as the main
remaining54.113/26.632ms gap. Earlier command-buffer causal evidence used B8;
this is one current-workload execution-path intervention, not a flag sweep.

Run the retained pinned upstream RGB12M/B32/T16/BF16 fixture, model, data,
seed, precision, library versions and runner twice, serially: ordinary
command buffers, then `--xla_gpu_enable_command_buffer=`. Append that flag
after upstream setup, as in the qualified B8 diagnostic. Do not change model,
arithmetic, autodiff, algorithm policy or native code. Keep16 warmup plus256
measured updates/arm,544 synthetic updates/zero game actions. No capture,
extra seed, timing repeat or optimization follows automatically.

Before updates, require expected command-buffer presence/absence, original
batch/precision/work preflights and initial288 tensors. Afterward compare all
272 returned metric reports, initial/final288 parameter/optimizer tensors,
reachable heavy operation contracts including convolution algorithms, and
compiled fusion bodies. Compare the enabled arm with its retained B32 control.
Any numerical/computational discrepancy stops a graph-only interpretation
for review, not tolerance relaxation or an automatic rerun. The benchmark
drains final deferred metrics outside timing; retain and report that tail.
Measure original completed-core and inclusive caller boundaries separately.

Use the existing host-only guard in a persistent systemd user service,
Restart=no,KillMode=control-group;900s/worker,1,920s/service. No concurrent
compilation/analysis or external NVML polling. Normal JAX/CUDA initialization
is permitted. Expected5080/580.178.04/boot
`3e89d55c-a9e5-472f-a18a-06508c5bafa7`,>=2GiB sampled Vulkan estimated headroom.
Record standalone allocation warnings; API/numerical/hard/deadline faults
stop the job for review. No recovery or blind retries. CPU preparation/review
uses1CPU/2GiB/no swap. Seal the wrapper, original runner and dependency closure.

The paired difference includes all effects of command-buffer execution and
host management; it is not automatically a pure device dependency cost.
It cannot be added to convolution bypass differences as an independent
native speedup. Exact outputs do not imply identical machine code or learning
quality. No Atari restart, production promotion or CUDA learner.
