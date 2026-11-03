# Replay16k passes bounded execution after a CPU audit repair

The fresh check completes **24,576 actions / 98,167 frames / 1,513 updates in
158.358s**, including **515 reports with replay saturated at16,384 frames**.
The conservative200k projection is **23m37.590s**, below the30-minute limit.
A1,024-action frozen check takes.903s with zero updates and no natural episodes.
Execution and early cost are established; gameplay benefit remains unmeasured.
No full learner, capacity sweep, extension or automatic promotion is selected.
The five-game goal remains unfinished.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-replay16k.md) changes
only shared replay capacity100,000→16,384 from the retained lambda0.95 control.
Native447fe50/SHA35ad0db, exact runnercf4ea238 and teacher remain unchanged;
there is no source implementation change or build. Keep12M64/Auto/N8/B16/T16/
H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100, sampled
collection/posteriors, full18/sticky.25 and the existing guidance schedule.
Seed1009 and evaluation4000001009 are reused development choices. No prior
weights, replay, belief, games, RNG or check experience transfer occurs.

| Bounded stage | Actions | Raw frames | Updates | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Fresh training | 24,576 | 98,167 | 1,513 | 158.358 |
| Frozen greedy | 1,024 | 4,096 | 0 | .903 |

All **378 original / 408 repair pins**,357 control-matching initial and
invariant frozen tensors,47 pre-update events,finite supervised reports,
changed CNN/reward/actor/value weights,full guided pixel/action/label replay
and host guards pass in the corrected audit. The pre-eviction16,000 actions
match **2,078 semantic events / 2,000 vector transitions** against the retained
control. Final replay length is16,384. Actual overrides number12,760 (51.921%);
guided returns and this short frozen capture are not competence measurements.

## Original failure and separate CPU closure

Main976085/invocationaf346c90f72b4c26bbee0fa9333738bf launches04:51:39.595 UTC
November3/source1e277a5. Both guarded GPU stages finish successfully. The new
long-prefix CPU comparison then incorrectly includes episode wall-clock time,
and the original main fails **04:54:31.441**. The original collector976088/
invocationc96a1d9b2ced4483b1ed81f4aa0b7615 fails **04:54:39.720** on that state.
Neither original service is restarted or relabeled as passing. The original
audit has no review/completion artifact. A separate partial-line status-read
error also remains recorded; it did not affect either service.

Saved-prefix diagnosis finds exactly40 differing `elapsed_seconds` fields,
first at action1,224. Every other recorded semantic field matches. The repair
excludes that wall-clock field along with `unix_time`; actions,rewards,pixel
hashes,labels,overrides and boundaries remain compared. Eight tests of the
actual prefix reader verify timing tolerance, semantic differences and the
fixed16,000-action boundary. Exact original sources and service journals remain
pinned. Native, runner, learning and evaluation are unchanged.

The corrected audit writes only under
`/x/Temp/kindle-breakout-replay16k-20261103/cpu-audit-repair`. It performs the
previously unrun guided replay in12.906s and completes **04:59:11.269 UTC** in
**17.639s**, after2.741s CPU preparation. Closure is
`complete_with_cpu_audit_repair`; original audit/collector failure remains.
No GPU stage, successful replay or original collector repeats. The original
success-only terminal reader was prepared but never run; do not run it now.

The exact native source confirms cached recurrent context and sampled refresh
from a pre-update posterior. This does not measure cache age or establish a
stale-state defect. Smaller replay changes recency,diversity,reuse and coverage
together; its effect on frozen gameplay requires a separate full comparison.
The original90%-two-wall gate and retained70.750/143.667 scores are unchanged.

Both the native-check and repair roots are closed. No Kindle allocation is
active at closure; check live state before new work. Do not repeat completed
qualification, GPU stages, replay, repair or collectors. Keep the failed
original services and all receipts intact.

| Artifact | SHA256 |
| --- | --- |
| Original declaration | `c70548faa4622c1154babe39a7c8a6fe99bb96a39d2322dcdc957a283cef3178` |
| Repair declaration | `038260a0cf0e1aa8a14c38415172453c1c81ed699d5d40d48af0f43a57f8a184` |
| Prefix diagnosis | `63e3d246a0c75c7521a7cf7f201f32f1c3192c7f7ddd1a79847f7822d64cef11` |
| Corrected review | `de05ac47f67f206aae213b4cad0b270742621cb15f77b5f1007ccbb26c4f2cdf` |
| Guided replay | `ff3aee7a70d0567e1b8463cedd8477107e32b42ee56df0c39d9d8b2c62a1a04b` |
| Corrected closure | `fa2de49f7a177449919f934d28808c294d3bb44cf147cc11ad41b2104e0808de` |
