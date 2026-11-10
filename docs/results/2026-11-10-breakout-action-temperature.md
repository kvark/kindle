# Diagnostic: action temperature on the 400k pixel-world and CDP checkpoints

[Protocol](../experiments/2026-11-10-breakout-action-temperature.md). Root
`/x/Temp/kindle-breakout-temperature-20261110`, outputs `out/`, guards
`guards/`; ran 21:35 to 21:37 UTC under
`kindle-breakout-temperature-20261110.service`. Frozen evaluations on cohort
4000001009, three natural episodes per environment, zero updates, no stalled
episodes; temperature 1 and greedy rows are the recorded 400k scores.

| Checkpoint (seed 1009, 400k) | Temperature | Episodes | Mean | Natural-24 | Median | Max | At or above 100 |
|---|---|---|---|---|---|---|---|
| Pixel world 64x64 | 1 (sampled) | 26 | 83.615 | 87.667 | 77 | 238 | 6 |
| Pixel world 64x64 | .5 | 26 | 162.500 | 164.875 | 105.5 | 404 | 13 |
| Pixel world 64x64 | .25 | 24 | **198.833** | 198.833 | 160 | 404 | 17 |
| Pixel world 64x64 | greedy | 25 | 185.160 | 189.958 | 160 | 404 | 19 |
| Retained CDP | 1 (sampled) | 24 | 92.792 | | | | |
| Retained CDP | .5 | 25 | 106.360 | 110.042 | 81 | 324 | 9 |
| Retained CDP | .25 | 25 | 113.880 | 115.042 | 99 | 365 | 12 |
| Retained CDP | greedy | 24 | 173.708 | | | | |

## Reading

The pixel world's sampled deficit is a distribution-sharpness effect, not a
worse action ranking. Halving the temperature lifts it from 83.6 to 162.5 and
quartering to 198.8, above its own greedy score, while the same temperatures
move CDP only from 92.8 to 106.4 and 113.9. At every temperature below 1 the
pixel world leads CDP by 50 to 85 points. The pixel-world actor therefore
spreads probability over actions it ranks correctly; the entropy bonus
(3e-4, upstream's default) or the 1% actor unimix keeps its policy flatter
than the retained CDP actor's. This is one seed on one cohort and an
evaluation-time knob, not the retained sampled protocol; a training-time
fix (lower entropy scale) is the natural next experiment, since the
retained protocol evaluates at temperature 1.
