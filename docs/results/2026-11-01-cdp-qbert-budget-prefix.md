# Qbert's fresh 300k run reproduces the recorded 200k training prefix

Every selected training event through200,000 actions matches the retained200k
control after removing explicit timing fields. All357 arrays in the new200k
recovery are elementwise equal to the control's final checkpoint, including
optimizer moments. The faster teacher and longer declared endpoint therefore
preserve this observed training prefix on the reused10007 learner.

| Compared evidence through200k | Control | Candidate | Exact matches |
|---|---:|---:|---:|
| Vector transitions |25,000|25,000|25,000|
| Non-timing learner reports |18,715|18,715|18,715|
| Completed episode records |118|118|118|
| Emulator frames |799,841|799,841|same per-stream counters|
| Checkpoint arrays |357|357|357 elementwise equal|

There are no differing, missing or extra event identities. Transitions retain
actions, pixel hashes, labels, overrides, rewards, frame counters and boundaries;
reports retain all non-timing learning values. Every compared array is finite,
with unchanged shape/dtype andzero changed elements or maximum absolute error.
Only event wall timestamps, episode elapsed time and the explicit learner timing
dictionary are excluded. No frozen gameplay outcome is produced by this check.

This narrows the longer-run comparison: the extra100k begins from checkpoint
values equal to the old200k endpoint. Experience and updates still increase
together. Hidden beliefs, gradients and RNG state between updates were not
directly compared; this is not independent seed reliability or universal proof
of teacher equivalence. Final/actual-initial frozen assessments remain required.

## Checkpoint identity and accounting limits

At200k, metadata differs only in tensor-file hashes, while all357 saved array
values agree. Serialized file identity and bitwise float identity are not claimed.
At100k, only the old metadata remains; its file hashes differ from the new copy.
There are no original100k tensor files available for a numerical comparison.
Matching recorded events does not replace those missing checkpoint values.

The two prefixes cover200k actions and799,841 emulator frames. The initial
summary incorrectly added cumulative frame counters, producing9,998,052,118.
A separate.902s CPU correction verifies each increment and the final per-stream
counters against the control's completed review. It verifies the exact original
prefix byte hashes and changes no event comparison or array metric. Six counter
fixtures pass. The original incorrect auxiliary field remains in its report;
`frame-count-correction.json` and `final-closure.json` supersede that field.

## Preserved CPU reporting failure and completed repair

The original analysis fails at18:03:55 UTC November1 after5.920s wrapper time.
Bare tensor names collapse357 arrays into324 dictionary entries because22 names
occur in multiple checkpoint groups. Its final count assertion rejects the
report; no original review exists. The original source,declaration and15 input
pins,start receipt and failed execution are unchanged.

The [qualified repair](../experiments/2026-11-01-cdp-qbert-budget-prefix-repair.md)
changes one accumulator key to include its checkpoint group. The exact source
difference, a synthetic repeated-name fixture and all357 schema identities pass
in.115s (.134s wrapper). The14 unchanged numerical/event fixtures are reused.
All25 repair inputs and524 active learner inputs pass before/after the repaired
analysis. It completes18:06:05 UTC/sourceac42edb in4.883s (4.921s wrapper), with
the original PID826399/invocation6947a8ed4d734296a1cff8e1af8113b6 unchanged.

The frame-count correction completes18:08:25 in.902s (.925s wrapper), and final
closure completes18:09:12. Closure is `complete_with_cpu_reporting_repairs`.
CPU inputs are reread for the failed-analysis repair; no successful comparison,
GPU stage, native inference or learning update is repeated. Preserve all CPU
co-host time without correcting training wall time. The learner continues.

Root `/x/Temp/kindle-qbert-budget-prefix-20261101`; repaired numerical report
under `group-key-repair`. The [original protocol](../experiments/2026-11-01-cdp-qbert-budget-prefix.md)
and both original/repair declarations remain fixed. Complete;do not rerun.

| Artifact | SHA256 |
|---|---|
| Original failed execution |`6f48600bb092e36b6e95496cd6c12d9fa3de6662f31da7448429879b08304099`|
| Repaired comparison |`4276235e0cbc3856aad27e3ba7473ffa81df60dc93f610788b0c946e2bc4ff8d`|
| Frame correction |`a57cbb5074ef9bf6704c7ada3366ab1dce8f8b37845c7872d3101d33a11e7de7`|
| Final closure |`9ab0d98a0b09c081fc91f4759a53afb85efbc6c6433a94b7b5bf167c51de2a6c`|
