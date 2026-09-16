# Block-matmul on the qualified current backend

Status: qualified. All fourteen new GPU tests, six complete-state canaries and
ten N6 windows pass. Throughput improves 27.3% with exact learning state;
Freeway integration and the complete raw audit also pass. No adoption.
NVML stays disabled.

## Change and fixed inputs

Kindle `dee38b2` carries the unchanged block graph from `7b190f88` onto
`e2a1d53`, whose [same-driver N6 comparison](2026-09-16-native-memory-pixels.md)
is complete. Both arms use Meganeura `589d73ab` and Blade `2accfeee`, including
upstream `986f49a` / `92553493`. Fresh checks still find those upstream tips.

Only `networks.rs` changes native behavior: batches 2–16 use upstream block
products; batch-one and larger cooperative products retain their serial graph.
Parameters, precision, optimizer, learning recipe and causal histories stay fixed.
This targets the many splits/concatenations in the
[measured world graph](2026-09-16-current-upstream-timing.md), not CPU offload.

## Build and native correctness

The [fresh Rust preparation](../../runs/current-block-fixtures-20260916.t8SQkG/declaration.md)
passes all 28 commands, 89 Kindle and five backend CPU tests, formatting and
release Clippy. Its final reader fails because Cargo's test depfile uses relative
source paths. Preserve that failed result; no build was repeated. The separate
[fixture reader](../../runs/current-block-runtime-20260916.Cx3Xei/fixtures.py)
verifies the unchanged binaries, relative/absolute depfiles, source manifests,
actual compiler edges, shared Blade and build timestamps. Public canaries also
match their hashed executable.

The [GPU declaration](../../runs/current-block-runtime-20260916.Cx3Xei/declaration.md)
binds 3,928 inputs and eight passing reader tests. It permits fourteen individual
native invocations, with explicit review between them. Seven unchanged backend
requirements reuse their complete current-source raw results; the completed
backend group is not restarted. Each new test uses the existing host-only guard,
actual native device/driver checks and complete initialization traces on the
RTX 5080 / driver 580.178.04. No NVML, automatic successor or recovery.

| Component | Result |
| --- | --- |
| Upstream block products, composed loss and all gradients against F64 | Pass: 12 sessions, 4,068 initialization records; direct child 91107 exits zero and is reaped |
| Kindle production blocks against serial products | Pass: loss, output and all three gradients bit-identical for batches 6/16 and both production shapes; 8 sessions, 6,461 records; child 91603 exits zero and is reaped |
| Production B16/T64 world model | Pass: all nine loss comparisons, 62 parameters / 51 nonzero gradients; worst relative L2 0.000745610 against the unchanged .003 limit; child 91740 exits zero and is reaped |
| Remaining Kindle runtime requirements | All eleven pass: device copies, complete checkpoint round-trips, live/vector learning and override causality, composed continuation and causal/batched LeVJEPA parity |

All fourteen guards pass, with 234 complete native initializations and 580,871
trace records. Every direct child exits zero and is reaped; no kernel fault is
recorded. N4/N6/N8 dense LeVJEPA batched/serial maximum error is zero; that is
correctness evidence, not combined learner-memory qualification for N8. GPU utilization,
recovery action and physically free memory are unmeasured, not inferred healthy.
These component tests do not satisfy memory, full-state or speedup gates.

The production test's two world plans shrink from 65,200/55,247 to
45,872/35,919 dispatches. That is 19,328 fewer in each graph, with unchanged
numerical acceptance. Dispatch count is not elapsed-time speedup; measure that
separately in complete-state canaries and warmed N6 playing plus training.

## Complete state and synthetic timing

The separate [complete-state group](../../runs/current-block-state-20260916.zc0Aig/declaration.md)
binds 4,274 inputs and six reader fixtures. All six native runs pass: updates 1
and 8 match exactly across both implementations, including all **241 tensors /
146 moments**, metadata, normalizers and non-timing reports. Both eight-update
repeats and both retained same-driver current-backend control anchors are exact.
All children exit zero and are reaped; 66 complete initializations / 780,729
trace records verify with no recorded fault or NVML. Preserve these terminal runs.

Untraced synthetic median timings over all eight updates:

| Stage | Parent A | Block B | Block B repeat | Parent A repeat |
| --- | ---: | ---: | ---: | ---: |
| Total update | 363.539 ms | 261.102 ms | 258.532 ms | 364.397 ms |
| World training | 176.325 ms | 97.999 ms | 96.665 ms | 177.479 ms |
| Posterior inference | 59.096 ms | 34.961 ms | 34.906 ms | 59.240 ms |

These are 28–29% shorter synthetic learner updates, not Atari end-to-end speed.
No compiler or other GPU workload ran alongside them. Native Vulkan memory usage
after construction is 6,449,397,760 bytes for parent and 2,222,063,616 for block;
after updates it is 6,449,397,760 / 2,356,281,344 bytes. Every declared snapshot
has ≥2 GiB estimated budget headroom. These are snapshots, not peak or physically
free memory; N6 combined perception is checked separately below.

## N6 playing plus training

The [source-matched Python package](../../runs/current-block-package-20260916.NaQ1zs/declaration.md)
is complete: native `886bae68`, 4,445 input pins, all 562 Python tests, Python
formatting/release Clippy and actual compiler/source/wheel/import identity pass.
It reuses the unchanged source's completed Rust checks and has its own fresh
private target. No old package or source is overwritten.

The [fixed N6 AB/BA pixel/restore/override/budget comparison](../../runs/current-block-pixels-20260916.M7whE0/declaration.md)
binds 4,487 inputs and 17 passing reader checks. Both arms use the same current
backend, N6/R256/B16/T64,
3,840 training actions / 610 updates, then 768 frozen actions; Freeway keeps
the existing .25/hold64 integration and unassisted restore.

Both Boxing orders now pass: every one of the 241 tensors / 146 optimizer moments,
610 non-timing reports, training and frozen action/reward/reset/episode traces
matches exactly across all four pairs. Both fresh serial controls also reproduce
the retained same-driver current-backend N6 pair exactly. Checkpoint container
hashes differ; tensor values, names, shapes and complete metadata are compared,
not just container bytes.

Warmed windows each contain 1,536 actual actions / 384 updates, with zero debt:

| Run order | Window seconds | Actions/s | Aggregate real time | Per-stream real time |
| --- | ---: | ---: | ---: | ---: |
| Serial A | 191.317 | 8.0286 | 0.53524× | 0.08921× |
| Block B | 150.317 | 10.2184 | 0.68123× | 0.11354× |
| Block B repeat | 149.939 | 10.2441 | 0.68294× | 0.11382× |
| Serial A repeat | 190.927 | 8.0450 | 0.53633× | 0.08939× |

The two speed ratios are **1.272755 / 1.273359**, both above the original
1.005 speedup gate. This is 27.3% higher throughput, or about 21.4% shorter
end-to-end windows—not real-time training yet. No compilation or competing GPU
work runs alongside these windows.

In the first matched window, mean learner time drops from 365.713 to 259.113
ms/update. World training falls from 174.470 to 92.585 ms and posterior inference
from 59.913 to 35.726 ms. Imagination stays at 87.214 / 86.668 ms; observation
stays at 49.810 / 49.723 seconds per complete measured window. World training
and imagination are now similarly expensive. These are native stage wall times,
not GPU occupancy or utilization. The fixed R256 workload still needs roughly
137 ms/update to reach aggregate 1× if other stage costs stay unchanged.

Both block training windows retain at least 7,695,106,048 bytes (7.17 GiB)
estimated Vulkan budget headroom, versus 3,446,210,560 bytes (3.21 GiB) for both
serial windows. This remains a sampled budget estimate, not physically free or
peak VRAM. NVML stays disabled.

## Complete integration and retained audit

Freeway's .25/hold64 training and unassisted frozen restore both pass. Its
executed override ledger is `[0, 128, 192, 128, 192, 256]` across six streams;
the restored evaluation performs zero updates and no overrides. This tests
integration, not a new Freeway competence result.

The ten completed windows contain 19,200 training actions / 3,050 updates and
3,840 frozen actions. All 140 complete initializations / 2,053,988 trace records,
11,555 native memory samples and 8,318 host checks verify. Maximum host-check
gap is 0.293 seconds; every direct child exits zero and is reaped, with no
recorded fault or NVML call. Total native budget-query time is 0.482 seconds
across the entire group. Utilization remains unmeasured.

The separate [completion reader](../../runs/current-block-completion-20260916.oVOgnU/declaration.md)
reverifies 4,843 pins, all ten raw results, the package/compiler identity and
hardware/state input chains, plus fresh host-only kernel/boot/driver evidence.
Both original speedup gates and both retained current-backend serial anchors
pass unchanged. Preserve this completed writer and every prior invocation;
only its retained-file audit may be reused:

```bash
python/.venv/bin/python -B runs/current-block-completion-20260916.oVOgnU/audit.py audit
```

## Next decision

The source-matched `dee38b2` / native `886bae68` bundle is now runtime-qualified.
Source integration and an explicit matched learning declaration are next;
main remains ce80 and the original Pong queue/hold stays terminal. Reuse the
qualified package with its matching adapters; do not relabel a rebuild as the
same tested artifact. A new matched learning declaration must use one package;
do not silently mix the old Pong root 1009 with new-backend roots as a three-root
reliability result. No game budget, seed, competence threshold or historical
result changes.
