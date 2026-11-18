# Without sticky actions the same agents lose a quarter to a half fewer balls and still stop at the first wall

Result of [the sticky-check declaration](../experiments/2026-11-18-breakout-sticky-check.md).
Root `/x/Temp/kindle-breakout-nosticky-20261118`, `queue-19.json` (16
cells, all complete, no GPU fault), guards `guards-19`, unit
`kindle-breakout-nosticky-20261118`, 16:57 to 17:37 UTC November 18.
Evaluation only: `L22` and `O22` at 1.6M, held-out `H50` at 1.2M and the
four-action `M22` at 1.6M, each evaluated as its sticky cells were with
`--sticky-actions 0`. CPU readings in `analysis/`, readers in
`scratch-scripts/`.

**Result.** Mixed, on both paddle widths: the sticky draw is neither the
direct cause of the lost balls nor irrelevant. In sampled play without it
the four learners lose 24% to 45% fewer balls on the wide paddle and 23%
to 49% fewer on the narrow one, and their medians rise to 331 to 409.
Half to three quarters of the misses remain with no noise to blame, the
agents still move into contact as often, and only 2 of 96 distinct
sampled games pass the first wall. The gap to the gate is mostly the
policy's own.

## Two protocol facts found in the run

- **The two cohorts are the same games.** The cohort seed only feeds the
  emulator's sticky draw, so without it cohorts 4000777001 and 4000022003
  replay identical games (the logs differ in the header only). Each
  learner has 48 distinct games, 24 sampled and 24 greedy, not 96. Rates
  are unaffected; counts in the pooled readers are doubled.
- **Greedy play stalls.** Without the draw, 1 to 4 of each learner's 24
  distinct greedy games run to the 100,000-frame limit in an endless
  volley. Those volleys count as returned balls and inflate the pooled
  narrow-paddle rate (`L22`: 7,730 pooled narrow descents at .960). I
  therefore read the tests on **sampled play** (no truncated game, one
  copy of the cohort) against the sticky sampled cells, and give the
  pooled figures as declared beside them.

## Declared tests

Sampled play: sticky (48 games) to no sticky (24 distinct games).

| | `L22` | `O22` | `H50` | `M22` |
|---|---|---|---|---|
| 1. Wide paddle returned | .951 to .973 | .924 to .942 | .920 to .952 | .915 to .949 |
| Wide, lost balls per descent | .049 to .027 | .076 to .058 | .080 to .048 | .085 to .051 |
| 2. Narrow paddle returned | .821 to .895 | .787 to .892 | .756 to .811 | .788 to .864 |
| Narrow, fastest balls | .764 to .884 | .732 to .893 | .711 to .792 | .690 to .855 |
| 3. Median score | 369.5 to 409 | 250 to 331 | 226.5 to 349.5 | 222.5 to 365 |
| 3. Games past 432 (no sticky, of 24) | 2 (best 733) | 0 | 0 | 0 |
| 4. Wide descents with a move among the last two actions | .734 to .699 | .773 to .762 | .813 to .822 | .765 to .740 |

Pooled as declared (both modes, duplicated cohorts), no sticky: wide
.972 / .950 / .956 / .947, narrow .960 / .887 / .944 / .928, pooled
medians 409 / 357 / 360 / 353.5 against 380.5 / 317 / 260 / 241.5.

- **Test 1 (wide):** between. No learner reaches .975 (`L22` .973 sampled,
  .972 pooled) and every learner rises by more than .015 (.018 to .034).
- **Test 2 (narrow):** between on sampled play: no learner reaches .90 and
  every learner rises by more than .03 (.055 to .105). By the letter the
  pooled figure passes the direct-cause bar on three learners; that is
  the stalled volleys, and I do not read it as a pass.
- **Test 4 (habit):** unchanged, as expected.

Predictions: wide .955 to .98 (three learners landed just under, .942 to
.952), narrow .84 to .92 (one under), medians higher by 30 to 100 (two
rose by more, 123 and 142). The direct-cause bar on the wide paddle was
not met; I had put it at one in three.

## What remains without the draw

- A paddle within 5 px of the landing three actions early that holds
  still returns .97 to 1.00 on the wide paddle; one that moves again
  returns .945 to .984. The agents still move again from there in 47%
  to 76% of the cases.
- Lost balls still miss by a median 8 to 12 px from the paddle's centre,
  with the paddle moving toward the landing in .64 to .89 of the cases.
- Scores stop at the first wall regardless: sampled medians 331 to 409,
  and 16, 6, 6 and 5 of 24 games end at 400 or more with only two past
  432. The planner under sticky actions has a median of 437.

## Post hoc: the decision near contact needs more position precision than the state shows

From existing dumps of `O22` and `M22` at 1.6M (own play under sticky
actions) and the imagined-contact dumps of `O22` and `L22`; not declared.

- **The right action flips at about 8 px** (`imagined_in_place.py`,
  emulator clones). One to two actions before contact on the wide paddle,
  staying loses .03 with the paddle within 4 px of the landing, .06 at 4
  to 8 px, .33 at 8 to 14 px and .50 beyond; one move toward the landing
  loses .18, .11, .13 and .10.
- **The policy does not make that distinction**
  (`policy_at_contact.py`). One action before contact it puts .55 /
  .35 of its mass on still / toward with the paddle within 5 px, .57 /
  .29 at 5 to 12 px and .32 / .65 beyond 12 px (`M22`; `O22` .48 / .40,
  .43 / .50, .22 / .76).
- **Imagination pays for holding still** on average: on `O22` the imagined
  return of staying exceeds that of an uncorrected tap by 7% to 18% of
  the return range with the paddle in place, and its imagined losses
  there are close to the emulator's (.048 against .030 for staying).
  At 8 to 14 px it expects staying to lose .14 where the emulator loses
  .33 (`L22` .06 against .24): the blur sits at the paddle's edge.
- **The state reads positions to about 3.5 px per object**
  (`state_precision.py`, `state_precision_mlp.py`, `O22`, 64,000 rows,
  held out by lane). A linear readout with 48,000 training rows gives
  the paddle to 3.5 px mean (within 6 px .85), the ball's x to 3.8 px
  and landing minus paddle one to four actions out to 6.2 to 6.6 px
  (within 4 px .39 to .41). A two-layer MLP readout is no better (4.3,
  4.9 and 6.9 to 8.1 px); the same MLP on the emulator's exact
  coordinates reaches 3.3 px mean, 2.0 median, against its own
  straight-line label.
- Against a simple perception reading: the paddle's movement follows the
  true landing (.63 to .89) rather than the linear readout's belief
  ([minimal-action report](2026-11-18-breakout-minimal-actions.md)). A
  readout cannot show that precision is absent, only that two readout
  families do not find it.

Reading (mine): the action that decides a ball depends on an offset of
about 8 px; the policy acts as if it resolved about 12 px; the state's
readable precision for that offset is 6 to 7 px. Whether the limit is
the representation or the actor's use of it is not settled by these
readings.

## Limits

- 24 distinct sampled games per learner without sticky actions; one
  development seed for three learners, one held-out seed.
- Agents trained under sticky actions and evaluated without them; the
  gains are what the draw costs these policies, not what training
  without it would give. The gate is under sticky actions.
- The post hoc readings use one checkpoint each of two learners.

## Next

Not launched. The user chose the actor's training signal on the
open-loop recipe as the next run. The readings above add a second
candidate, position precision in the representation, which the runner
cannot test without native work (image sizes stop at 96x96, and the
96x96 recipe already uses 16.5 of 20.9 GB). The choice between them goes
back to the user with this report.
