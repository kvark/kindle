# Breakout actor/critic representation gradients with replay-value1

The completed [fresh comparison](../experiments/2026-11-03-cdp-breakout-acgrads-value1-200k.md)
lowers frozen Breakout scores to **54.958 sampled / 117.833 greedy**, versus
the retained **70.750 / 143.667** control, in **21m56.980s** (+2.26% training
time). Both final arms have zero of 24 two-wall successes. Retain the control
with actor/critic representation gradients off. No extension, coefficient
sweep, confirmation learner or automatic successor is selected.

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
comparison subsequently receives its own declaration, CPU qualification and fresh experience.

Retain the 12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/behavior8e-5/
dynamics4e-4/replay-value1/expert100/replay100k control after the full result below.
Reused development seeds limit reliability claims; policy and value gradients
change together. The original 90% two-wall gate and five-game goal remain unmet.

## Completed fixed-budget comparison

| Measure | Retained control | Actor/critic representation gradients |
| --- | ---: | ---: |
| Sampled final mean | 70.750 | 54.958 (−22.32%) |
| Greedy final mean | 143.667 | 117.833 (−17.98%) |
| Actual-initial sampled / greedy | 1.417 / 2.208 | 1.417 / 2.208 |
| Two-wall successes, each final arm | 0/24 | 0/24 |
| Training time | 21m27.910s | 21m56.980s (+2.26%) |
| Initial-subtracted sampled score/min | 3.230038 | 2.439292 (−24.48%) |
| Initial-subtracted greedy score/min | 6.590132 | 5.267732 (−20.07%) |
| Final 50k-action replay expert CE | .560514 | .550341 |
| Actual guided-action fraction | 43.0585% | 44.8745% |

The learner completes 200,000 actions / 799,582 frames / 12,477 updates.
All 12,477 reports contain nonzero initial policy loss and positive initial
value loss; their sums are 35.780350 and 1,441.671402. The additional losses
are active, but finite loss reports do not independently validate derivatives
or explain the regression. Policy and value representation gradients change
together. The slightly lower replay expert CE does not outweigh worse frozen
gameplay. Online last-50 mean64.28 includes 89,749 teacher overrides across
training and is not unaided competence.

Fixed descriptive coverage reconciles all 285,024 training/frozen actions and
1,139,517 frames with 48 diagnostic and 428 full input pins. Native training
actions at raw scores216–431 fall from1,262 to828 (.414% of all actions), with
none432+ in either learner. Five of24 selected greedy episodes reach216,
none432; sampled reaches neither. These are score bins, not verified wall
transitions or a causal diagnosis of the architecture. The retained teacher
and raw reward/boundary semantics are unchanged.

The source remains286c54b, original PID990350 / invocation
`9aa43ad5b82b4f25a97817f22593d39d`. All 428 pins, 357 matching initial tensors,
47 pre-update events, 800/795 actual-initial sampled/greedy semantic events,
357 invariant frozen tensors per arm, full pixel/action/label and frozen
replays/videos, host guards, curves, exposure and comparison pass. Four frozen
arms total85,024 actions /339,935 frames /65.659s with zero updates/truncations.

Original recovery succeeds07:38:49.967 UTC November3; independent confirmation
passes once07:39:20.788 in1.284s CPU, retaining206,067,161 bytes at100k/6,227
updates. Main/postprocessing/collection succeed **07:54:00.102 /07:54:18.504 /
07:54:52.283**. All five original services are terminal success. Independent
terminal and fixed coverage readers complete once **07:55:21.963 /07:55:30.341**
in1.333s/4.276s CPU. No full-run failure, collector repeat or GPU-stage repeat
occurs. Two earlier full CPU preparation failures (collector date and parser
fixture) remain preserved; both were corrected before launch. Keep startup,
reader and recovery CPU co-host timings without wall-time correction.

All stages and readers are closed. No Kindle allocation remains active at
closure; check live state before work. Do not restart this learner or repeat
its observers, evaluations, replays or readers. Reused seed1009 and evaluation
cohort4000001009 limit reliability; this result does not establish that all
actor/critic representation supervision is harmful. The original90% two-wall
gate remains unchanged, and Breakout and the five-game goal remain unfinished.

| Artifact | SHA256 |
| --- | --- |
| Full declaration | `416bccea8c93522005c74e4a6c0666d5154ea2f8541dcbc7af5646dad12bb185` |
| Learning review | `312f7245d8c5dddbcd9a010a912025013ff43dcbac63770fef13f4510e57454b` |
| Original completion | `b029cb5ed67c1b78ca30a847fc7bf828e57a4d8403b810e76469e6b74c6d75ad` |
| Independent terminal summary | `074d990ef592ee20abbe88a204ec046154fcd50b6402ffb0872cc4ebb2a25f7a` |
| Fixed coverage | `f8b8ac82526d079b7f6a25dc3166ce655ce5899f64f7a15595266cf710faacc0` |
