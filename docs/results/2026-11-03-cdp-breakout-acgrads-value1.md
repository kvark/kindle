# Breakout actor/critic representation gradients with replay-value1

The existing optional actor/critic representation-gradient path passes its
bounded execution and early-cost check. This establishes that the losses run
within budget; it does not establish gameplay improvement. The separately
selected [fresh comparison](../experiments/2026-11-03-cdp-breakout-acgrads-value1-200k.md)
will assess that against the retained 70.750 sampled / 143.667 greedy control.

Only `actor_critic_gradient` changes false→true. Initial imagined policy and
value losses now also train the representation through frozen behavior heads
and stopped targets, normalized by H15. Later imagined states remain detached.
Reward, continuation, replay-value1 and expert100 representation gradients were
already enabled. This is an additional supervised path, not a missing-loss fix.
The older unaided 4M gain and Pong regression do not predict this supervised
12M outcome. Native447fe50, runnercf4ea238 and the teacher remain byte-identical.

The [bounded protocol](../experiments/2026-11-03-cdp-breakout-acgrads-value1.md)
completes 8,192 actions / 32,727 frames / 489 updates in **47.296557 seconds**.
Its conservative 200k projection is **1,270.171 seconds (21m10s)**, below the
30-minute gate. The 1,024-action / 4,096-frame frozen check takes 0.879672s,
with zero updates and no completed natural episodes. Neither short score is
an assessment of competence.

All 379 pins, 357 matched initial and invariant frozen tensors, 47 pre-update
semantic events, finite reports, changed CNN/reward/actor/value tensors,
full pixel/action/label replay and host guards pass. All 489 reports contain
nonzero initial policy loss and positive initial value loss; their sums are
12.504847 and 212.903364. Those observations verify active losses, not their
derivatives or causal contribution. Actual guided overrides number 3,839.

CPU preparation passes 14 loss-validation cases, five actual semantic-reader
cases, both commands, 18 config cases, update cadence and the existing tensor
schema without native construction. Main PID989880 / invocation
`ae616af4686c4be2abd4bd985f26433c` succeeds at 07:21:24.245 UTC November 3;
the original collector succeeds at 07:21:30.568. The independent terminal
reader completes once at 07:24:10.295 in 1.197s. No failure, GPU repeat,
collector repeat, native edit or build occurs. Both original services are
terminal success; check live state before any later allocation.

Evidence root: `/x/Temp/kindle-breakout-acgrads-value1-20261103`.
Source `cdd97446e2a8d21da5c0d8228245acbfd51df9d9`; declaration
`c5b7f5795a682c0d364ad64a9d80420b4ad8c78a398c1c77a373f15026f9d9ce`;
review `2125dfca8b80d55fef4c76f513b28cbd01225c72714adc8258152dcdc6d727ad`;
completion `c612e6aee7c4c8293a04dc4addc467764480342565e576573b009b8afd6bfcac`.
The bounded stages and reader are closed; do not repeat them. The full
comparison needs its own declaration, CPU qualification and fresh experience.

Retain the 12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/behavior8e-5/
dynamics4e-4/replay-value1/expert100/replay100k control pending frozen results.
Reused development seeds limit reliability claims; policy and value gradients
change together. The original 90% two-wall gate and five-game goal remain unmet.

## Fresh comparison active

The separately selected full learner launches at 07:27:43.538 UTC November 3,
source286c54b, PID990350 / invocation `9aa43ad5b82b4f25a97817f22593d39d`, with
428 pins and original recovery/progress/postprocessing/collection observers.
Startup passes 357 matched initial tensors, 47 paired semantic events and 270
finite reports through 4,696 actions. The 200k/12,477-update training cap is
30 minutes; 100k recovery and all four frozen assessments remain required.
The full CPU harness and terminal/coverage readers are qualified. Two pre-launch
CPU failures are preserved and corrected; no GPU stage repeats. See the
[full protocol](../experiments/2026-11-03-cdp-breakout-acgrads-value1-200k.md)
for exact qualification and identities. No full frozen result is available.
