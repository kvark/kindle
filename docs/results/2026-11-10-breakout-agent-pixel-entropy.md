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

RUN_E3E5

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
EVAL_E3E5

## Reading

READING
