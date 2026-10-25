# Conditional read spans lose active lanes, not local-memory capacity

October16,2026. [Declaration](../experiments/2026-10-16-control-lowering.md)
· [Compact evidence](2026-10-16-control-lowering.json).

The rejected read-span implementation performs **23.08% fewer thread-level
instructions but18.01% more warp instructions**. Its average active threads per
issued warp instruction drops **31.997→20.857**. Both arms have **zero measured
local-memory traffic**, with a passing nonzero local-array control.

This identifies the failed implementation's loss of active lanes; it does not
assign exact driver instructions or directly count divergent branches. It also
rules out treating that new problem as the qualified baseline's existing gap:
the baseline already keeps almost32 threads active per issued warp instruction.
Its excess convolution feeding/address work versus JAX remains the target.

Keep isolated58aff452: **RGB46.199ms versus inclusive JAX27.223ms,1.70×**.
No new ordinary timing, backend build, production change or Atari restart.

## Complete unchanged-family observation

All20 contracts/23 occurrences retain exact binaries, shader sources, fixtures,
bindings, transforms, reductions and outputs. The observer bridge adds eight
metrics to the retained50; the SDK and capture logic are unchanged.

| Main-kernel observation | Control | Rejected candidate |
| --- | ---: | ---: |
| Warp instructions |10.296B|12.150B|
| Active thread instructions |329.435B|253.417B|
| Predicated-on thread instructions |267.182B|212.682B|
| Active threads / warp instruction |31.997|20.857|
| Predicated-on threads / warp instruction |25.950|17.504|
| CBU instructions, any predicate on |25.593M|350.212M|
| CBU instructions, all predicates off |0|0|
| Local reads / writes |0 /0|0 /0|

All nine changed contracts lose active-thread coverage; candidate means range
14.933–24.795. Every old warp/pipe/tensor instruction anchor reproduces exactly.
These are instruction-mask measurements, **not SM utilization, occupancy or
latency shares**. Predicated-on instructions include bookkeeping, not only useful
tensor arithmetic. Small counter-ratio deviations around32 remain unrounded in
the data; no clamping or threshold adjustment. Observed predicate sums reconcile,
but exact additivity was not used as a collection gate.

The result is consistent with extra masked-path execution after introducing
common-span/alignment branches. It does not prove which source branch diverges,
whether a helper is inlined, or where the driver places individual instructions.
NVIDIA distinguishes active versus predicated-on thread instructions and direct
divergent-branch observations in its [counter documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).
The complete local SDK catalog contains1,639 metric base names; its only branch
metrics describe branch-resolving stalls, not direct divergence or calls.

The prior **18.809→28.187ms,49.86% regression** remains the ordinary timing
result; this observation does not retime or reopen that rejection. No favorable
subset, integration or full-update comparison follows.

## What the emitted code proves—and does not

These are call-expanded static K-loop sites, including both sides of branches,
not native instruction counts:

| Typed pass | Branch sites, old→new | Call sites, old→new | Global load sites, old→new |
| --- | ---: | ---: | --- |
| Forward |69→77|10→48|16×2B→32×2B +4×8B|
| Input gradient |85→95|26→56|16×2B→32×2B +4×8B|
| Weight gradient |75→90|10→64|16×2B→32×2B +4×8B|

New scalar readers and two-path vector-return helpers coexist with the old
fallback. FunctionControl is zero, which proves neither inlining nor retained
calls. Constant division/modulo IR sites do not prove native division cost.
Fourteen packed occurrences remain byte-identical.

Two concrete CPU proofs prevent misleading explanations:

- Across all four typed dW contracts, every one of1,310,720 aligned reduction
  positions has a full aligned span. Split boundaries and channel terms preserve
  that alignment. The new fallback's unsigned division helpers are therefore
  unreachable for this workload, despite remaining in emitted IR. This is not
  proof the driver removes them or their effect on compilation.
- Existing dX signed-remainder helpers receive a constant divisor1. An independent
  interpreter checks the actual helper instructions for65,564 values across64
  call sites; all return zero, with eight negative mutations rejected. These are
  not newly expensive software divisions. General odd/boundary shapes retain
  their original fallback requirements.

For an actual valid forward span,32 of128 invocations are aligned. Two legal
subgroup assignments give four versus zero mixed alignment groups. This rejects
inferring actual divergence from local IDs; no particular mapping is measured.

## Verification and retained failures

41 guards pass with40 recorded allocation warnings. All2,050 counter passes and
6,902 metric values reevaluate offline.516,574,592 retained output comparisons
and258,287,296 fresh bit-exact pairs pass, including changed-input/restoration
checks. The positive control reads8.464MB/writes12.593MB locally while the
register and empty controls stay zero; new thread counters are live.

There are2,822 standalone operation calls, zero learner updates/actions or
ordinary timing samples. Last GPU-guard event is**18:55:17UTC**; bounded aggregate
review finishes**18:55:32UTC**.361 collection/aggregate source seals pass; no
study worker remains. Independent raw-counter/membership/output publication
review checks365 combined source seals and finishes19:12:09UTC. Capture durations
are not ordinary performance results.

The first CPU audit conflates signed constant-one remainder helpers with unsigned
division helpers and fails. Signedness/call-argument classification repairs it;
the independent interpreter then checks the actual operations. The first bridge
comparison fails on one added trailing newline; the reviewed comparison ignores
only terminal newlines while requiring an otherwise exact body outside the metric
list. Original scripts/logs remain. Neither repair repeats GPU work or relaxes
numerical gates.

## Next decision

Do not optimize the baseline for a loss of active lanes it does not have. Start
from its qualified, guarded scalar-load staging and CPU-prove **common-base/carry
address arithmetic** for the derivatives, sharing redundant index/cursor work
without a per-lane alignment fast/slow path. Preserve every load mask, scalar
load body, K order, precision, compensation and barrier. Validate general small
and strided cases as well as the complete current family before any GPU trial.

No new vector-storage/layout/width/geometry sweep, selection of timed shapes,
compiler feature or automatic GPU allocation. This is not a parity promise:
library feeding/reuse and the non-convolution boundaries remain open, as does
the full root-cause/performance goal.

Artifacts: `runs/control-lowering-20261016.S6Tu7DVO`.
