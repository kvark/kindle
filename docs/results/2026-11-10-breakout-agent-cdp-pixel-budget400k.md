# CDP plus pixel auxiliary: original-cohort scores and a 400k budget point

[Protocol](../experiments/2026-11-10-breakout-agent-cdp-pixel-budget400k.md). Root
`/x/Temp/kindle-breakout-cdp-pixel-400k-20261110`; `eval200k/` holds the
original-cohort evaluations of the existing 200k checkpoints, `candidate/`
the 400k life; guards `guards/`, unit
`kindle-breakout-cdp-pixel-400k-20261110.service` (launched 19:13 UTC,
invocation `426caf90519046c1b5d63a5dd0a86067`). Source `e1dc13e5`, Meganeura
main `a0eff600`.

## 200k checkpoints on the original cohort 4000001009

Three natural episodes per environment, zero updates; no episode stalled.

| Learner (seed 1009, 200k) | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|
| Retained CDP (reference) | sampled | 24 | 70.750 | | | | | 0/24 |
| Retained CDP (reference) | greedy | 24 | 143.667 | | | | | 0/24 |
| CDP + pixel scale 1 | sampled | 26 | 66.538 | 68.292 | 45.5 | 311 | 4 | 0/26 |
| CDP + pixel scale 1 | greedy | 25 | 103.040 | 103.500 | 91 | 319 | 7 | 0/25 |
| Pixel world 64x64 | sampled | 25 | 37.680 | 38.542 | 33 | 67 | 0 | 0/25 |
| Pixel world 64x64 | greedy | 25 | 156.840 | 162.708 | 117 | 394 | 15 | 0/25 |

The retained CDP learner was selected and repeatedly assessed on this
cohort; the pixel-world learners were never selected on either cohort.

Seed 1009 across both cohorts (4000001009 / 4000022003), greedy and sampled:

| Learner | Greedy | Sampled |
|---|---|---|
| Retained CDP | 143.667 / 94.625 | 70.750 / 70.500 |
| CDP + pixel scale 1 | 103.040 / 104.036 | 66.538 / 61.692 |
| Pixel world 64x64 | 156.840 / 151.080 | 37.680 / 40.286 |

## 400k life

RUN400K

EVAL400K

## Reading (200k part)

On its own selection cohort the retained CDP learner keeps the sampled lead
(70.750 against 66.538) and a wide greedy lead (143.667 against 103.040) over
the combined learner, while the pure pixel world again wins greedy (156.840)
and loses sampled (37.680). The combined learner's advantage over CDP is
therefore cohort- and seed-dependent: it wins both modes on seed 22003 /
cohort 22003 and on seed 1009 / cohort 22003 in greedy, and loses both on
seed 1009 / cohort 1009. What is consistent across all three
seed-and-cohort pairs is that the pure pixel world beats CDP in greedy
(156.8, 151.1, 99.3 against 143.7, 94.6, 75.3) and loses in sampled, and that
both pixel-world learners vary far less between cohorts (within 4 points
greedy and 5 sampled) than the CDP learner (49 points greedy between its
selection cohort and the new one). Which learner is "best" depends on the
mode and the cohort; none passes the two-wall gate.

READING
