# Stronger Breakout encoder improves state readouts, with mixed action evidence

The retained encoder4e-5 checkpoint has more accurate ball, paddle and motion
readouts than encoder2e-5 on the same saved control trajectories. Near-paddle
teacher agreement also rises, but probability assigned to the teacher action
falls slightly and opposite-direction choices increase. This supports better
accessible state information without identifying a single cause of the remaining
control gap. No replacement head, new loss or learner is selected.

The separate [full learner comparison](2026-11-01-cdp-breakout-encoder4e5.md)
establishes frozen sampled25.292→56.292 and greedy42.625→104.667 in21m28s versus
22m27s. This diagnostic adds no training or new gameplay assessment. All frozen
stages preserve357 tensors and zero updates; all473 input pins and host guards
pass. Main succeeds03:44:02 UTC November1/sourcec17d723, with terminal collection
at03:44:41. No GPU stage repeats or failures occur.

## State precision improves on matched rows

The table reports the fixed combined-feature kernel readout's mean absolute
error. Positions use original source pixels; motion uses source pixels per
recorded action. Each target uses exactly shared held-out rows across models
and the pixel reference where available.

| Target | Shared rows | Encoder6e-6 | Encoder2e-5 | Encoder4e-5 | Pixel reference |
|---|---:|---:|---:|---:|---:|
| Ball x |555|12.053|7.554|6.028|.059|
| Ball y |555|10.211|6.515|4.192|.140|
| Paddle x |601|6.768|4.342|3.790|2.843|
| Horizontal motion |540|2.953|2.517|2.146|—|
| Vertical motion |540|2.422|1.628|1.301|—|
| Teacher response target x |557|17.642|13.867|11.702|—|

The improvement is also present in fixed linear readouts: recurrent ball x/y
error falls9.052/6.661→7.820/4.648, while CNN error falls10.420/10.875→7.875/7.504.
Combined-kernel ball x/y errors improve on both held-out lanes. On near-descending
rows, teacher response-target error falls10.082→8.743 for the combined kernel
and10.454→9.494 for the recurrent linear readout.

The pixel reference still recovers much more precise ball position. Its color
and crop priors, two missed reference balls and four extra detections remain
recorded. A readout's residual error does not prove information absent from the
learned features, and better decoding does not establish how the policy uses it.

## Aggregate agreement improves, but directional decisions remain uneven

Each saved actor is evaluated on its own scalar-reobserved recurrent states.
The CPU F64 probability calculation matches the native actor in the separate
128-action check. These are teacher-agreement measurements, not gameplay scores.

| Actor metric | Encoder6e-6 | Encoder2e-5 | Encoder4e-5 |
|---|---:|---:|---:|
| Overall agreement,601 rows |54.08%|60.90%|62.56%|
| Near-descending agreement,136 rows |50.00%|62.50%|66.91%|
| Overall teacher-action probability |.478|.561|.572|
| Near-descending teacher-action probability |.455|.582|.573|
| Overall teacher cross-entropy |.975|.894|.808|
| Near-descending teacher cross-entropy |1.063|.828|.867|
| Near-descending opposite-direction choices,89 directional labels |19|7|11|
| Near-descending NOOP choices,89 directional labels |23|20|16|

Compared with2e-5, near-descending NOOP-label agreement rises23/47→30/47,
while directional-label agreement falls62/89→61/89. The aggregate gain therefore
does not demonstrate better directional control near the paddle. FIRE agreement
stays41/44. Overall agreement improves on both held-out lanes, but the phase and
class breakdowns prevent treating that average as a uniform improvement.

All six fixed linear/kernel action readouts still trail the learned actor's
62.56% overall and66.91% near-descending agreement. The best replacement in each
case is the recurrent linear readout at57.40% and61.03%. There is no supporting
result for replacing the actor with one of these heads.

Teacher disagreement is not optimal-action error. The earlier
[local interventions](2026-10-30-cdp-encoder-representation.md#local-action-interventions-limit-the-teacher-disagreement-diagnosis)
already show that some confident policy disagreements outperform the pixel
teacher locally. This diagnostic does not repeat those interventions or measure
the4e-5 policy's own-state coverage.

## Fixed inputs and completed validation

Reuse the original6e-6 control's unaided final-greedy corpus:20,960 recorded
actions/83,799 frames, with teacher labels added offline. All input pixels,
actions, rewards, reset boundaries and frame counts replay exactly. Preserve
954/509/601 train/validation/test rows on lanes0–3/4–5/6–7, including136
near-descending test rows. Normalization uses training rows, kernel bandwidth
comes from training rows and ridge selection uses validation rows. Reuse the
original6e-6 and2e-5 captures, predictions and fits without repeating GPU stages
or refitting their readouts. The three-way comparison aligns all2,064 rows and
reproduces original control/pixel metrics exactly.

The explicit comparison profile allows only source encoder6e-6 versus restored
encoder4e-5; all other configuration fields must match. Three parsers, six
combined-profile rejection cases and seven actual-configuration cases pass.
The81 scoped CPU tests plus two independent regression fixtures and5.237s
preflight perform zero native/emulator imports. Exact historical Git bytes
resolve4 original representation,5 actor and5 previous encoder source pins;
the original declarations remain unchanged. Prior55/289/358/227 input sets pass.
Native447fe50 and the qualified F64 actor functions/constants stay unchanged.

Feature compatibility takes3.295s; native actor qualification2.922s; full capture
21.858s. The actor check passes128/128 greedy choices,128 readback checks and16
exact prior feature rows. Maximum probability error is8.515e-7 against the fixed
1e-4 tolerance. Every stage preserves all357 checkpoint tensors and zero updates.
CPU state/actor/comparison analyses take3.333/1.539/.245s; all prediction hashes
and three host guards pass. Whole service time is42.143s.

Scalar posterior draws differ from original vector acting. The reused control
states and development learner seed do not measure the stronger policy's own
trajectory distribution or independent reliability. The full learner remains
0/24 two-wall successes in both evaluation modes; the five-game goal is unfinished.
No learner or GPU allocation is active at closure. Check live state before work.

Root`/x/Temp/kindle-encoder4e5-representation-20261101`, same-named service.
The [fixed protocol](../experiments/2026-11-01-cdp-encoder4e5-representation.md),
three frozen captures, predictions, actor probabilities, analyses, exact source
resolutions and original terminal journal are retained.

- Declaration`05bf76a60f4e1c5221b8f91dae106c8cf2307b5d7a2c90f01711e6bebed4598b`.
- Review`9800c472544ff4e22f67f11db1fbf2dd0859e07fc3ba6424431374e46462d32d`.
- Comparison`8674c0d5a20191f79c2d3ad7bb43d77c3b97efd1432a84a7c4abc0c11c251211`.
- Completion`b4646de533388a873c02a61384fc13fb87a5be75330cd3e6e92b567256830f2d`.

## Own-policy trajectories improve productivity and show a smaller agreement gain

The [CPU follow-up](../experiments/2026-11-01-cdp-encoder4e5-policy-phase.md)
replays the4e-5 policy's37,016 saved greedy actions/148,038 frames. All episode/
task outcomes, partial tails, frame/reset counts and source identities reproduce
exactly. The first3 natural episodes per8 lanes select31,161 actions, versus
24,858 in the saved2e-5 cohort. Prior control rows and aggregates are reused
without another emulator replay. This measures each policy's own encountered
states; it does not pair identical inputs or establish independent reliability.

The selected episodes last25.36% longer: mean1035.75→1298.375 actions. Total
return rises1023→2512, equivalent to41.154→80.614 points per1000 actions, a95.88%
increase. Median episode score rises38.5→76.5; ranges are16–103 and14–385.
These describe the already-recorded greedy42.625→104.667 gain, with0/24 two-wall
successes in both cohorts. No new gameplay assessment is performed.

| Teacher-defined phase | Encoder2e-5 rows | Exact / horizontal agreement | Encoder4e-5 rows | Exact / horizontal agreement |
|---|---:|---:|---:|---:|
| All |24858|54.97% /55.33%|31161|59.04% /59.32%|
| Ball unseen |1421|88.46% /89.94%|2288|91.87% /93.27%|
| Velocity unavailable |602|49.34% /51.83%|884|55.77% /56.90%|
| Far descending |6705|50.44% /50.48%|8048|52.40% /52.40%|
| Near descending |5496|57.59% /57.91%|6582|59.86% /60.12%|
| Ascending |10613|52.30% /52.61%|13334|57.22% /57.44%|
| Stationary or ambiguous |21|57.14% /61.90%|25|64.00% /64.00%|

Near-descending exact agreement improves on seven of eight lanes; lane4 falls
59.49%→58.99%. Among directional teacher labels, opposite-direction choices
fall476/3434 (13.86%)→429/4006 (10.71%), while literal NOOP choices rise798/3434
(23.24%)→1147/4006 (28.63%). This differs from the fixed-control-state scalar
actor comparison above, where opposite-direction counts rise. The states,
sampling and posterior draws differ, so neither comparison isolates a mechanism.
Full confusion counts and per-lane results remain retained.

## Fixed windows locate lost lives without assigning a causal action error

The replay additionally records ALE lives immediately before and after each
action. All24 selected episodes lose five lives:120 loss events, including24
terminal losses. Each loss has a teacher-detected near-descending ball within
the fixed32-action window. The last such row is0–4 actions before81 losses and
5–8 actions before39; none requires the older age bins. This establishes recent
pixel-detector visibility, not that the native representation tracks the ball.

At the last near-descending row,86/120 actions match the teacher. The34
disagreements comprise10 FIRE commands where the teacher says RIGHT,12 NOOP
commands where it says LEFT/RIGHT,two opposite-direction commands,and10
directional commands where it says NOOP. The teacher's projected response-target
distance from the current paddle has median8.046 source pixels at these rows.
The target and phase remain estimates; a current target gap does not establish
whether a bounce is avoidable or which earlier command caused the loss.

These observations motivate examining the approach-to-paddle sequence and the
time available for correction. Last-action imitation alone cannot classify the
observed losses: most last near-descending actions already agree with this
imperfect teacher. No counterfactual action or teacher takeover is tested here.
The old control has no per-action life counts, so these windows describe only
the4e-5 candidate. Windows may overlap and rows are temporally correlated.

The subsequent [teacher-takeover intervention](2026-11-01-cdp-breakout-recoverability.md)
tests these failures directly: early control avoids108/120 losses at their
recorded endpoint, versus26/120 at the last near-descending state. This narrows
the timing question while leaving the precise architectural cause unresolved.

The23.375s CPU replay (24.583s controller wall) completes04:04:01 UTC November1,
source3012999. Eight pure tests,265 analysis pins,all276 active Qbert inputs,
originalPID736977/invocation48d20e8959f04654b6f5ff8932328d8d and all replay checks
pass. The four original phase/metric/probe definitions are unchanged; the replay
function changes only its probe factory to the life-counting subclass. No native
module or GPU descriptor is opened, and no model forward pass or update occurs.
CPU preflight2.883s, tests and replay co-host with Qbert; retain those timing
caveats without correction. No failure, replay repeat or new learner is selected.

Evidence root`/x/Temp/kindle-encoder4e5-policy-phase-20261101` retains all rows,
120 indexed windows, per-lane/confusion results, test/source records and the
separate post-hoc arithmetic in`derived-summary.json`.

- Review`328868ff8004a1a26882e893f56ce73d2be99315ae8b2e6ed1d4ba032c7e0efe`.
- Life windows`a859c4760e9a57f091f3b390160a35886e8c59810d0b78cc8e04b58eceafd5bd`.
- Completion`5cb945814f38aad4d75c6cf4a3cc64dd76cbbb6ca34bc98f35f4999b03b2a45c`.
