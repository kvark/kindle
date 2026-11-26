# Pixel-world Breakout agent with a lower actor entropy scale

[Protocol](../experiments/2026-11-10-breakout-agent-pixel-entropy.md). Root
`/x/Temp/kindle-breakout-pixel-entropy-20261110`; outputs `e1e4/` and
`e3e5/`, guards `guards/`, unit `kindle-breakout-pixel-entropy-20261110.service`
(launched 21:41 UTC, invocation `863c12018544450fae1b366221995f73`). Source
`964f71fb`, Meganeura main `a0eff600`. Each command is the matched pixel
world 64x64 200k command (seed 1009) with `--actor-entropy` added.

## Entropy 1e-4

Training 21:41 to 22:14 UTC: 200,000 actions, 12,477 updates, 1,964.2 s
wall; 313 online episodes with guided play, mean 48.6. Final policy entropy
.89 (weighted .86) against the 3e-4 run's .87, return scale 17.8 against
37.5, imagined return mean 18.6; world losses unchanged (reconstruction
2,790, prior rollout 2,876).

## Entropy 3e-5

Training 22:15 to 22:48 UTC: 200,000 actions, 12,477 updates, 1,969.2 s
wall; 307 online episodes with guided play, mean 48.7. Final policy entropy
.88 (weighted .85), return scale 20.9, imagined return mean 10.7; world losses
unchanged (reconstruction 2,823, prior rollout 2,920).

## Actor unimix 0 (November 11)

Training 07:01 to 07:33 UTC (root `/x/Temp/kindle-breakout-pixel-unimix0-20261111`,
unit `kindle-breakout-pixel-unimix0-20261111.service`): 200,000 actions,
12,477 updates, 1,953.3 s wall; 307 online episodes with guided play, mean
50.5. Final policy entropy .80 (weighted .78), return scale 24.0, imagined
return mean 20.0.

## Frozen evaluations (200k, three natural episodes per environment, zero updates)

| Entropy scale | Cohort | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|---|
| 3e-4 (reference) | 4000001009 | sampled | 25 | 37.680 | 38.542 | 33 | 67 | 0 | 0/25 |
| 3e-4 (reference) | 4000001009 | greedy | 25 | 156.840 | 162.708 | 117 | 394 | 15 | 0/25 |
| 3e-4 (reference) | 4000022003 | sampled | 28 | 40.286 | 42.833 | 33 | 84 | 0 | 0/28 |
| 3e-4 (reference) | 4000022003 | greedy | 25 | 151.080 | 154.167 | 111 | 383 | 15 | 0/25 |
| 1e-4 | 4000001009 | sampled | 27 | 34.222 | 35.708 | 31 | 88 | 0 | 0/27 |
| 1e-4 | 4000001009 | greedy | 27 | 93.926 | 97.333 | 82 | 292 | 8 | 0/27 |
| 1e-4 | 4000022003 | sampled | 25 | 33.720 | 34.583 | 29 | 74 | 0 | 0/25 |
| 1e-4 | 4000022003 | greedy | 25 | 85.800 | 86.083 | 79 | 264 | 7 | 0/25 |
| 3e-5 | 4000001009 | sampled | 24 | 31.375 | 31.375 | 28 | 55 | 0 | 0/24 |
| 3e-5 | 4000001009 | greedy | 24 | 129.125 | 129.125 | 83 | 342 | 9 | 0/24 |
| 3e-5 | 4000022003 | sampled | 26 | 29.462 | 29.958 | 26.5 | 90 | 0 | 0/26 |
| 3e-5 | 4000022003 | greedy | 24 | 110.917 | 110.917 | 92 | 400 | 11 | 0/24 |
| 3e-4, unimix 0 | 4000001009 | sampled | 27 | 37.704 | 39.667 | 31 | 104 | 1 | 0/27 |
| 3e-4, unimix 0 | 4000001009 | greedy | 24 | 87.042 | 87.042 | 73.5 | 289 | 7 | 0/24 |
| 3e-4, unimix 0 | 4000022003 | sampled | 27 | 32.741 | 35.042 | 29 | 68 | 0 | 0/27 |
| 3e-4, unimix 0 | 4000022003 | greedy | 24 | 104.458 | 104.458 | 87 | 313 | 7 | 0/24 |

## Reading

Negative, for both knobs. Cutting the entropy scale three- or tenfold
leaves the actor's final policy entropy where it was (.87, .89, .88 nats
across 3e-4, 1e-4 and 3e-5) and the sampled scores unchanged or slightly
lower (31 to 34 against 38 to 40), while the greedy scores fall (94 to 129
against 151 to 157). Removing the 1% actor unimix lowers entropy only to .80
and leaves sampled scores at 33 to 38 with greedy at 87 and 104. That every
single-knob variant scores 87 to 129 greedy where the baseline scored 151
to 157 also says the baseline's greedy point may sit at the high end of
seed variation; the three variants are one seed each and cannot separate a
knob effect from that. The
entropy bonus is therefore not what keeps the pixel-world actor's
distribution flat; something else pins its entropy near .88 nats for four
actions, and the temperature result (198.8 at .25) says the ranking under
that flat distribution is good. Candidates not tested here: actor unimix
(.01 mixes 1% uniform but does not explain .88 nats), the return
normalization (return scale 18 to 38 against the CDP learner's values,
which shrink the advantages that sharpen the policy), and the imitation
term (scale 100 on guided rows, which pulls the actor toward the teacher's
distribution). One seed per point; the greedy drops are within the seed
variation seen elsewhere today (CDP 143.7 against 94.6 between cohorts, 94.6
against 75.3 between seeds) and are not read as an entropy effect.
