# Block-matmul on the qualified current backend

Status: all fourteen new GPU tests and six complete-state canaries pass. N6
Atari throughput remains unmeasured. No adoption. NVML stays disabled.

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
free memory; N6 combined perception still needs its own checks.

## Next decision

The [source-matched Python package](../../runs/current-block-package-20260916.NaQ1zs/declaration.md)
is complete: native `886bae68`, 4,445 input pins, all 562 Python tests, Python
formatting/release Clippy and actual compiler/source/wheel/import identity pass.
It reuses the unchanged source's completed Rust checks and has its own fresh
private target. No old package or source is overwritten.

The [fixed N6 AB/BA pixel/restore/override/budget comparison](../../runs/current-block-pixels-20260916.M7whE0/declaration.md)
now binds 4,487 inputs and 17 passing reader checks. Its first serial training
window is active. Both arms use the same current backend, N6/R256/B16/T64,
3,840 training actions / 610 updates, then 768 frozen actions; Freeway keeps
the existing .25/hold64 integration and unassisted restore. Both fresh serial
controls must also reproduce the completed current-backend N6 anchor exactly.
Keep both warmed throughput ratios ≥0.98; a speedup requires both >1.005.
Main remains ce80 and Pong roots 2017/3019 stay held. No game budget, seed,
competence threshold or historical result changes.
