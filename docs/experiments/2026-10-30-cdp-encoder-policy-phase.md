# Phase-conditioned actions on the improved Breakout policy's own trajectories

The [fixed-state diagnostic](../results/2026-10-30-cdp-encoder-representation.md)
finds better state readouts and teacher agreement after the encoder-rate change,
but uses the old control's trajectories. Select one CPU-only replay of the
completed encoder2e-5 learner's final-greedy recording to measure the behavior
it actually produced. No native agent, policy evaluation, training or GPU work.

Reuse the exact phase definitions, teacher, action grouping and replay routines
from `/x/Temp/kindle-breakout-frozen-phase-20261029/analyze.py`. Replay all28,808
recorded actions, keeping the first3 natural episodes per8 lanes=24 as the
selected cohort. Preserve extra episodes and partial tails in saved rows.
Require exact agreement with the retained episode/task/partial-tail replay,
frame/reset counts and source identities. Ordinary frozen logs contain no live
pixel hashes; newly saved replay-input hashes do not establish live-image parity.

Report all phases, all selected rows, per-lane results, exact teacher agreement,
horizontal-command agreement and complete confusion counts. Reuse the old
control's saved rows and aggregates after reproducing its metrics exactly on
CPU; do not repeat its emulator replay. Do not select phases or thresholds from
the new result. The two policies encounter different states and episode lengths:
this is not a paired-state comparison, a causal decomposition or independent
learner confirmation. Pixel-teacher labels are estimates, not optimal actions.

CPU preflight checks all source/completion identities, exact AST equality of
reused functions/classes and reproduction of retained control aggregates. Run
on one low-priority CPU core under90s, with native-module imports forbidden.
Verify no GPU device descriptors, all declared inputs and the active Qbert
learner's235 pins and original PID/invocation before and after. Retain co-host
time as a caveat without adjusting the Qbert training wall time. Stop and
preserve any failed analysis; do not disturb or restart the active learner.

Root `/x/Temp/kindle-encoder-policy-phase-20261030`. No loss, architecture,
policy replacement or successor GPU allocation is selected by this diagnostic.

CPU preflight passes in1.855s:194 input pins,exact AST equality for all five
reused functions/classes,and exact reproduction of the old control's aggregate
and eight per-lane metrics from20,960 saved rows. Original21 phase/173 learner
pins and all235 active Qbert inputs remain unchanged,with the same live process.
No native/emulator import occurs during preflight. Its CPU co-host time remains
recorded without a training-time correction.
