# One cost-attribution capture after the T64 preflight

T64's equal-position cost falls only4.6%, below the fixed10% gate, and sampled
headroom is barely above2GiB. Keep T16. Before choosing the next finite learning
allocation, attribute the dominant imagination cost with one existing profiler
invocation. This is a diagnostic, not a new optimization candidate or sweep.

Use the already qualified release test binary5100da4e from the October9 refresh,
Meganeura31026833/Blade56f0565. Restore the retained12M Breakout1009 checkpoint
only into a disposable synthetic fixture, seed701. The existing
`profile_fixed_batch_checkpoint` test executes one ordinary update, then
profiles posterior, imagination, slow value, world gradient and behavior
gradient on fixed inputs. It verifies exact saved tensors before/after the
optimizer-free profiling passes. No game actions, replay resume or competence
claim. Preserve the source checkpoint unchanged and account for the one extra
synthetic update and repeated diagnostic passes.

`MEGANEURA_GPU_TIMING=1` is confined to this process. Per-dispatch timestamps
perturb scheduling, so use dispatch/shape/arithmetic attribution and label all
instrumented times. They are not ordinary update latency or SM utilization.
World/behavior profiles omit optimizer/clipping/accumulation; do not claim they
cover every learner operation. No compilation or concurrent heavy work.

One serial host guard,300s process limit,10min persistent service,
Restart=no/KillMode=control-group, ordinary GPU-worker CPU allocation,
RTX5080/580.178.04 and>=2GiB sampled Vulkan estimated headroom. Allocation
warnings are retained; API/numerical/hard-fault/deadline failures stop for review.
No separate NVML polling, recovery, blind retries or gate changes. CPU report
preparation/audit uses one CPU,2GiB,zero swap.

Artifacts: `runs/cdp-imagination-cost-20261009.c1GzuW6V`. Seal binary/source
checkpoint hashes before launch. Review before selecting any change or longer
learning budget. All five quality targets and the RGB budget question stay open.
