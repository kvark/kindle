# Longer Qbert training sequences pass the bounded execution and cost check

The fresh Qbert B8/T32 comparison is **active**, starting14:31:43 UTC October29
under source `07ce3dc`, service `kindle-qbert-context32-200k-20261029`, PID307065/
invocation `7e5b79d7f445434187c3f3221a6117e4`. Startup passes all229 pins, exact
config and357 finite initial tensors matching both control and bounded check.
First updates504/520/536/552/568/584 and86 finite supervised reports in the
1880-action inspected prefix match the schedule. Progress,100k recovery and
postprocessing observers are active. Budget200k/12469 updates,45min training
cap and four frozen arms stay fixed. No task result is available yet. Check
actual service state; never restart because observation times out.

Fresh 12M48/Auto/N8/B8/T32/H15/R16 completes **8,192 actions / 32,756 emulator
frames / 481 updates in 72.797s**. The subsequent 1,024-action frozen restore
takes .788s with 4,096 frames and zero updates. No natural frozen episodes
complete, so this check establishes execution and early cost, not task competence.

All 41 declared inputs remain unchanged. All 357 initial tensors match the
retained B16/T16 control; all 357 frozen tensors are finite and unchanged.
Config/source/counter checks, changed CNN/reward/actor/value weights, aligned
actor/world expert losses, complete training pixel/action/corrective-label replay
and both host guards pass. First update504 and final481 updates match the
independently derived schedule. All481 reports have nonzero supervision.
Actual overrides are4,391/8,192 (53.601%), not the20% episode selection rate.

The conservative200k projection is **1,955.004s (32m35s)**, using measured time
times200000/8192 times1.10. This passes the declared45min ceiling but is not a
guaranteed full-run duration. B16/T16→B8/T32 preserves256 replay positions per
update andR16 while changing temporal gradient length and sequence diversity.
Behavior-target rows rise240→248; later warmup produces eight fewer optimizer
updates at a fixed action budget. Capacity, native447fe50, runner, teacher and
supervision stay fixed. No isolated memory effect is claimed.

Service `kindle-qbert-context32-check-20261029` starts14:26:12 UTC October29,
source `c1ca09b`, and succeeds14:28:05 after1m53.315s including CPU replay.
Terminal collection passes14:28:28. No fault, failed stage, GPU repeat, cap
change or recovery occurs. Evidence remains outside Git under
`/x/Temp/kindle-qbert-context32-20261029/native-check`.

- Declaration: `4c81d61ba7d955d35a25d6d8ca4b2f22a1575e29ea7f401ed75b45ac18d71811`
- Review: `5e749eb457194c03366db1465fea88c04c367ea1e0834215826195da3ea666cd`
- Completion: `d28ff70c8c2d053fb058e861b34d998e3d8d5ae6caa29598027a6d3702cf0f2e`

See the [bounded protocol](../experiments/2026-10-29-cdp-qbert-context32.md).
After completion, one fresh200k life is separately selected in the
[full protocol](../experiments/2026-10-29-cdp-qbert-context32-200k.md). No bounded
weights or experience are reused. The five-game goal remains unfinished.
