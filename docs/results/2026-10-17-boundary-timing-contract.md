# Measure existing execution boundaries without splitting passes

October17,2026. [Evidence](2026-10-17-boundary-timing-contract.json) ·
[Backend plan](../backend_training_performance_plan.md).

**A separate cached-chunk timestamp observer is feasible on CPU.** It can retain
the learner's original compute passes, barriers, submissions and host waits.
This is a checked recording/lifetime contract, not an implemented or qualified
GPU observer. Keep isolated`e45b9c43`: **RGB43.197ms versus inclusive JAX27.223ms,
1.59×**. No backend build, GPU call, learner update or game action occurs here.
Production and Atari remain unchanged.

## Why the existing profiler is insufficient

The retained complete trace covers all272 updates,6,775 intervals and1,088 stage
intersections. Its native58aff452 world chunks still match the current compiled
world exactly: two posterior chunks, then three backward chunks. But pass timing
reuses a primary encoder's query slots and therefore adds three intermediate
host waits. Its27.935ms world/6.976ms imagination intervals remain valid for that
instrumented run, not current ordinary timings. The original2,107ns crossing
and preflight/reader failures remain; no timestamps were discarded or recollected.

Per-dispatch profiling also opens extra compute passes. The proposed observer
instead places timestamps at existing in-pass barriers and chunk endpoints.
It does not revive the failed sparse-pass timing method.

## Complete native membership

| World program | Dispatches | Original groups | Cached chunks | Timestamp slots | Convolution intervals / products |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB |3,751|1,112|5|1,137|17 /23|
| Centered CDP |4,088|1,117|5|1,130|8 /11|

The original RGB recording has1,107 within-chunk group barriers plus20 existing
within-group convolution-class barriers. Four other group edges fall at chunk
boundaries, retaining the original primary/secondary synchronization. CDP has
1,112+8 within-chunk barriers. Both marked command streams reduce exactly to
their originals when query commands alone are removed.

Independent convolution products sharing an interval stay together: **17 RGB
intervals are not23 isolated kernel timings**. The remaining1,115 RGB/1,117 CDP
intervals contain the other model work. No favorable subset is selected.

Non-world runtime groups were not retained in the plan exports. Their dispatch
counts bound capacity, but are not invented group partitions. All four cached
programs need at most4,700 RGB/5,119 CDP query slots; this is a capacity bound,
not GPU memory use. An implementation must export and reconcile the actual
imagination, slow-value and behavior groups before interpreting their timings.

## Query ownership and measurement limits

Each immutable cached secondary owns a separate pool, reset by its recorded GPU
commands. Query results are harvested only after an **existing successful wait**
covering that execution, before replay or destruction. The observer uses64-bit
values plus availability, without adding a query wait or accepting partials.
Vulkan orders resets relative to query operations, but availability alone can
still describe a prior replay; owner/generation tracking is required.
([Reset semantics](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdResetQueryPool.html),
[result availability](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html).)

Source review finds a concrete implementation requirement: current Session wait
paths discard errors, while the timing helper returns`None` both when timing is
off and when its wait times out. A diagnostic must check the existing wait's
actual success; neither`None` nor an available old query is completion proof.
This does not diagnose a historical GPU fault or request another wait.

Keep ordinary context/pass timing disabled. That context does not enable
calibrated timestamps, so this proposal reports **intra-secondary elapsed
intervals**, not host-aligned overlap or a global device timeline. Check the
actual timestamp period/valid bits at initialization and reject ambiguous wraps.

The fixed`ALL_COMMANDS` writes have execution dependencies of their own. Keeping
original barriers and submissions does **not** prove zero instrumentation cost.
These are completion-frontier intervals, not pure kernel latency, SM utilization,
DRAM traffic or a directly subtractable JAX gap budget.
([Timestamp semantics](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html).)

## Verification and next step

Ten complete cached-world command projections,2,176 modeled executions over272
modeled updates, three duration fixtures and22 negative checks pass. The
independent reviewer matches all original within-group boundaries and29 source
seals. Review finishes**02:43:42UTC**; no worker remains. The original CPU audit
confuses20 shape contracts with18 shader enum variants and fails before plan
loading. Its source/log remain; the separate reader checks the full named set.

Next implement this small observer in an isolated copy, including actual group/
command exports and explicit completion/generation checks. CPU/source checks
must pass before a separate bounded numerical/timing allocation. Fixed gates:
exact plans/pipelines/groups/original command signatures and numerical states,
reports and targets;24-update qualification, then one ordinary/marked RGB pair
with16 warmups+256 measured updates,<=5% control drift and<=5% overhead. Preserve
all samples/tails and stop on failure; no marker-density search or retime.

The observer is not an optimization or production change. Convolution feeding,
recurrence and the full root-cause/parity goal remain open. All rejected kernel,
UBO and normalization searches stay closed.

Artifacts: `runs/boundary-timing-contract-20261017.Uhh6GIP0`.
