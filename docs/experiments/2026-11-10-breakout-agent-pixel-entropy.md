# Pixel-world Breakout agent with a lower actor entropy scale

The [action-temperature diagnostic](../results/2026-11-10-breakout-action-temperature.md)
showed that the pixel-world agent's sampled deficit against CDP is a flat
action distribution, not a worse ranking: at temperature .25 its 400k
checkpoint scores 198.8 against CDP's 113.9 on cohort 1009, while at
temperature 1 it trails 83.6 to 92.8. The retained protocol evaluates at
temperature 1, so the fix has to be in training. The actor entropy
regularizer (upstream's 3e-4) is the direct knob; actor unimix (.01) is the
other and stays unchanged here.

## Recipe

The [matched pixel world 64x64](2026-11-10-breakout-agent-world96-matched.md)
200k command (seed 1009; retained recipe minus CDP, plus motion cross-entropy
and the 8-step prior image rollout, bonus kept) with one change each:
`--actor-entropy 1e-4` (`e1e4/`) and `--actor-entropy 3e-5` (`e3e5/`). The
runner flag is new (sets `actor_entropy`, a native config field that already
existed); it is a fresh-training option. Root
`/x/Temp/kindle-breakout-pixel-entropy-20261110`, `queue.json`, unit
`kindle-breakout-pixel-entropy-20261110.service`.

## Evaluation

Frozen evaluations of each 200k checkpoint on cohorts 4000001009 and
4000022003, sampled and greedy, three natural episodes per environment. The
actual-initial checkpoint is the seed 1009 initialization already evaluated
(sampled 1.286 / greedy 2.071 on cohort 1009; 1.241 / 2.214 on cohort
22003). References at 200k, greedy / sampled: pixel world entropy 3e-4
156.840 / 37.680 (cohort 1009) and 151.080 / 40.286 (cohort 22003); retained
CDP 143.667 / 70.750 and 94.625 / 70.500.

## Reading

One seed, two entropy points. A sampled score at or above CDP's with the
greedy score held would say the pixel objective's gain transfers to the
retained sampled protocol; a sampled gain with a greedy loss would say the
entropy bonus was doing useful exploration work; no sampled change would
point at unimix or at the critic instead. Policy entropy and online returns
are reported alongside.

## Addition (November 11): actor unimix 0

The entropy scale did not move the actor's distribution. The other explicit
policy knob is actor unimix (.01 in the retained recipe: 1% uniform mixed
into the policy at training and sampling time). One run: the matched pixel
world 64x64 200k command (seed 1009, entropy 3e-4) with `--actor-unimix 0`,
frozen evaluations on cohorts 4000001009 and 4000022003, sampled and greedy.
Root `/x/Temp/kindle-breakout-pixel-unimix0-20261111`, `queue.json`, unit
`kindle-breakout-pixel-unimix0-20261111.service`. Read against the unimix
.01 run's 37.680 / 156.840 and 40.286 / 151.080 (sampled / greedy).
