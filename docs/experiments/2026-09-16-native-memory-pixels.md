# N6 native gameplay and memory budgets without NVML

Status: **all ten same-driver N6 phases pass**, including exact repeated Boxing
state/reports/traces, frozen restores, Freeway overrides, native memory-budget
coverage and the matched throughput non-regression gate. The independent
retained-file audit also passes. This completes this new non-NVML pixel/runtime
comparison; it does not repair historical gates or establish a speedup,
adoption, driver/NVML causal fix or new game competence.
Keep main ce80 and Pong held. This continues the completed
[current-upstream hardware, full-state and native profiling work](2026-09-16-current-upstream-timing.md).

## Small native API, unchanged learning

The opt-in `gpu_memory_budget` getter exposes the existing shared Blade context's
memory statistics through the Rust core and Python `Agent`/`VectorAgent`.
It creates no context, monitor, GPU worker or CPU execution fallback. Without
`--min-gpu-budget-headroom-mib`, the Atari runner makes no memory-budget calls
and creates no memory sidecar. Enabled runs retain the original action/reward
log schema and write a separate flushed `.gpu-memory.jsonl` ledger.

Samples follow construction, initial observations, each act/observe/learn,
individual reset batches, checkpoints and completion. Every row retains the
native process ID, action/update counters, estimated usage/budget and query
duration; the first also records the actual native GPU. Unsupported budgets,
invalid integer values or headroom below the requested minimum stop the runner.
No NVML function or `nvidia-smi` command is involved.

Vulkan defines usage and budget as dynamic process estimates. Their difference
is **estimated budget headroom**, not globally physically free VRAM or a peak
measurement. The new gate is explicitly declared as such; it does not convert
old NVML readings into current health or change historical acceptance results.
See the [Vulkan memory-budget definition](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html).

| Role | Kindle | Meganeura | Blade | Native SHA256 prefix |
| --- | --- | --- | --- | --- |
| ce80 control | `5ff7b55` | `ce80e9cd` | published 0.9.0 | `55899347` |
| Current candidate | `e2a1d53` | `589d73ab` | `2accfeee` | `234d43ac` |

Both branches carry identical memory/profiler changes on their respective bases.
Both Python profilers now match main's no-NVML implementation. Native learning
math, precision, cooperative policy, configuration and backend pins are unchanged.
The source branches are `exp/ce80-pixels-20260916` and
`exp/current-pixels-20260916`; neither is merged.

## Completed builds and preserved failures

The private [native builds](../../runs/current-pixel-package-v2-20260916.swlgNZ/preparation.md)
use one CPU, 2 GiB and zero swap. Control passes **78 Rust tests**, candidate
**86**; 22/23 GPU tests respectively remain ignored in these CPU invocations.
Formatting, release Clippy and fresh wheel builds pass in both arms.

Each original build stops at command fourteen, before Python tests, because its
identity reader expects a depfile named for the Cargo package `kindle-py` rather
than the compiled `_native` library. Preserve both original `complete: false`
results and absent original Python-test/identity outputs. Neither build is retried.
The still-earlier gHOVn2 writer stopped on an absent role directory before any
build or GPU work. Preserve that original failure too.

The separate [package completion](../../runs/current-pixel-package-check-20260916.wIjRCa/completion.md)
checks the actual `_native` depfile and unchanged native → Kindle → Meganeura
compiler chain, shared Blade edge, fresh build window, and source/wheel/import
identity. **All 562 Python tests pass per package**, including the opt-in/default
memory behavior and no-NVML profiler check. All original commands remain visible;
the completion does not relabel their failed assertions as successes. Its parent
and candidate closures bind 2,602 and 2,604 pins. Only `check.py audit ROLE` is
reusable; builds and test writers are terminal.

## New same-driver pixel declaration

[cDLhXG](../../runs/current-driver-pixels-20260916.cDLhXG/declaration.md) binds
**3,141 inputs**, fifteen passing reader fixtures and fresh upstream reads of
Meganeura `986f49a` / Blade `92553493`. It re-audits the prior complete GPU
profile/state/hardware and both packages before declaring ten separate phases.
There is no automatic queue or retry. The guard owns the direct Python process
hosting the Rust extension; synchronous ALE stepping spawns no GPU descendants.

Retain N6, 12M/B16/T64/full-BPTT64/world-microbatch16/R256/F32, learning rate
4e-5, warmup1000, AGC0.3, reconstruction0/future-prediction0.25. Each training
arm gets 3,840 actual actions/610 updates, seed7301, followed by final-checkpoint
restore for 768 sampled frozen actions/zero updates, seed8301. Order is Boxing
control/candidate/candidate/control, then candidate Freeway with .25/hold64
exploration during training only.

Require exact all 241 tensors/146 moments, metadata except declared backend
identity, non-timing reports and action/reward/reset/episode traces across the
Boxing repetitions. Check complete save/restore and Freeway override causality.
The warm 2,304–3,840 window retains 1,536 actions/384 updates: both candidate/
control ratios must be at least .98; a speedup claim requires both above 1.005.
All phases have the same post-stage Vulkan headroom gate of at least 2 GiB,
complete sample coverage, host guard and 1,800-second bound. No NVML telemetry
is inferred from these measurements.

This is explicitly a same-driver comparison on 580.178.04. The older 595-to-580
exact checkpoint-anchor failure remains failed; it is neither silently dropped
from its original declaration nor repaired by this new protocol.

The first control pair completes with native PIDs 75227/77773, exit zero and
reaped, with no recorded kernel fault. Training completes all 610 updates;
frozen restore consumes the exact final 241-tensor checkpoint and performs zero
updates. All 1,924/387 post-stage memory samples pass. The minimum headroom is
3,446,210,560 bytes (3.21 GiB), and the training memory queries take 85.5 ms total,
maximum 0.451 ms. These are native query costs, not full instrumentation overhead.

Its warm control window takes 190.374 s for 1,536 actions/384 updates:
**8.068 actions/s, 0.5379× aggregate and 0.08965× per stream**. Learning consumes
139.443 s (73.2%); observation 49.762 s (26.1%). Native learner reports average
363.011 ms/update, including world training 174.136 ms, imagination 86.983 ms,
posterior 59.979 ms and world synchronization 15.374 ms. Removing NVML has not
made this control super-real-time;
the measured workload still supports optimizing native learning/perception.

The first candidate pair also passes, PIDs 78334/80541, exit zero and reaped.
All 241 tensors/146 moments, non-timing reports and training/frozen traces are
exactly equal to the new control, with no relaxed numerical tolerance. The two
candidate phases verify all 28 complete session initializations and 450,310
trace records. Memory coverage passes for 1,924/387 samples, with the same
3.21 GiB minimum training headroom. Neither phase records a kernel fault.
The candidate's 191.308 s warm window yields 8.029 actions/s, **0.995114×**
control throughput: inside the regression limit, but not a speedup.

## Complete group and next decision

All four fresh Boxing training runs match exactly in all 241 tensors/146
moments, metadata except declared backend identity, non-timing reports and
gameplay traces. All four sampled frozen runs also match exactly and consume
their own final checkpoint bytes with zero updates. No tolerance is relaxed.

| Order | Role | Train / frozen PID | Warm window | Actions/s |
| --- | --- | --- | --- | --- |
| A | ce80 | 75227 / 77773 | 190.374 s | 8.0683 |
| B | Current | 78334 / 80541 | 191.308 s | 8.0289 |
| B | Current, repeat | 81064 / 83122 | 191.358 s | 8.0268 |
| A | ce80, repeat | 83662 / 85544 | 190.269 s | 8.0728 |

Both candidate/control ratios pass the unchanged .98 regression floor:
**0.995114 and 0.994309**. Neither passes the >1.005 speedup gate. This backend
refresh is essentially throughput-neutral, not a solution to sub-real-time
learning. Candidate aggregate real time is 0.5353×/0.5351×, about 0.0892× per
stream; these clocks include learning at the original R256.

Freeway's train/frozen pair, PIDs **86083/88263**, passes the same complete
save/restore and budget checks. Training verifies **896** persistent override
actions, per stream `[0,128,192,128,192,256]`; frozen evaluation is unassisted,
sampled and has zero updates. This short integration window is not a competence
test or a new Freeway win.

The [independent completion](../../runs/current-pixel-completion-20260916.vN4gvf/capture/result.json)
re-audits all ten raw phases in their actual package environments, then checks
fresh host kernel evidence. Its eleven commands complete under one CPU, 2 GiB
and zero swap. **3,496 input/evidence pins** reverify. Across the group:

- **19,200 training actions / 3,050 updates**, plus **3,840 frozen actions**.
- All ten native children exit zero and are reaped; **9,528 host checks** pass,
  maximum gap **0.289699 s**, with no recorded kernel fault or NVML call.
- All **84 candidate initialization sessions / 1,350,930 trace records** pass,
  including physical allocations/binds, immediate Shared zeroing, constant
  uploads and checked waits.
- All **11,555 native memory samples** cover their declared stages/counters.
  Minimum estimated budget headroom is **3.21 GiB**. Total query time is
  **0.490 s** across all ten phases, not a measurement of total logging overhead.

Preserve cDLhXG's preparation, all ten invocations, both package targets and the
completed vN4gvf writer. Never rerun a native phase, package builder or capture.
The complete retained-file check, requiring no GPU execution or queries, is:

```bash
python/.venv/bin/python -B runs/current-pixel-completion-20260916.vN4gvf/audit.py audit
```

The new same-driver gates pass; the older cross-driver anchor and original NVML
memory gates are not relabeled as passing. Main remains ce80 and Pong stays held.

The unchanged small-batch block-matmul source is staged separately at
`dee38b2` (`exp/current-block-matmul-20260916`). Its `networks.rs` is byte-identical
to the original `7b190f88` candidate; current backend, Python runner and learning
settings stay fixed. It is not built or GPU-qualified, and no compilation ran
alongside this matched timing experiment. Next, build its source-matched native
fixtures in a fresh private target and run block component/world correctness,
complete state and matched N6 throughput before any new Pong campaign. Reuse
these completed controls where explicitly declared; do not repeat this group
or introduce a CPU fallback, actor/learner separation or NVML monitoring.
