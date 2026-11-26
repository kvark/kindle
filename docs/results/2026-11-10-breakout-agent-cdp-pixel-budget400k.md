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

Training 19:16 to 20:21 UTC: 400,000 actions (1,599,371 frames), 24,977
updates, 3,929.2 s wall (65m29s against the retained CDP 400k life's 43m37s),
101.8 actions/s, learn stage 3,636 s, VRAM 9.6 GiB. The 200k recovery
checkpoint reproduces the 200k life bit for bit (update 12,477: CDP loss .056,
reconstruction 2,763.07, prior rollout 2,819.74, as in `s1/`). At 400k the
CDP loss is .053, reconstruction 2,662, prior rollout 2,737, exploration .03,
reward .03, KL 18.9; imagined return mean 54.1, return scale 129.3, policy
entropy .61. 404 online episodes with guided play, mean 75.3 (50k buckets:
25.0, 56.1, 83.5, 89.7, 150.6, 91.0, 89.0, 132.4).

### 400k frozen evaluations

| Learner (seed 1009, 400k) | Cohort | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|---|
| CDP + pixel scale 1 | 4000001009 | sampled | 24 | 85.583 | 85.583 | 69.5 | 301 | 7 | 0/24 |
| CDP + pixel scale 1 | 4000001009 | greedy | 27 | 135.593 | 120.833 | 106 | 341 | 16 | 0/27 |
| CDP + pixel scale 1 | 4000022003 | sampled | 25 | 102.840 | 104.583 | 84 | 366 | 7 | 0/25 |
| CDP + pixel scale 1 | 4000022003 | greedy | 27 | 134.519 | 145.417 | 96 | 383 | 13 | 0/27 |
| Retained CDP (reference) | 4000001009 | sampled | 24 | 92.792 | | | | | 0/24 |
| Retained CDP (reference) | 4000001009 | greedy | 24 | 173.708 | | | | | 0/24 |
| Retained CDP (evaluated here) | 4000022003 | sampled | 24 | 120.625 | 120.625 | 90 | 365 | 10 | 0/24 |
| Retained CDP (evaluated here) | 4000022003 | greedy | 24 | 150.750 | 150.750 | 114 | 384 | 15 | 0/24 |
| Pixel world 64x64 | 4000001009 | sampled | 26 | 83.615 | 87.667 | 77 | 238 | 6 | 0/26 |
| Pixel world 64x64 | 4000001009 | greedy | 25 | **185.160** | 189.958 | 160 | 404 | 19 | 0/25 |
| Pixel world 64x64 | 4000022003 | sampled | 24 | 100.917 | 100.917 | 66 | 359 | 7 | 0/24 |
| Pixel world 64x64 | 4000022003 | greedy | 24 | **158.625** | 158.625 | 148 | 349 | 19 | 0/24 |

The pixel world's 400k life (root `/x/Temp/kindle-breakout-pixel64-400k-20261110`,
20:25 to 21:30 UTC) took 3,902.7 s (65m03s), 24,977 updates, 102.5 actions/s,
VRAM 9.5 GiB; its 200k recovery checkpoint reproduces the matched 64x64 life
bit for bit (update 12,477: reconstruction 2,735.49, prior rollout 2,827.57).
At 400k: reconstruction 2,756, prior rollout 2,811, exploration .02, reward
.04, KL 17.1; imagined return mean 16.1, return scale 23.6, policy entropy
.77; 467 online episodes, guided mean 67.5 (50k buckets 33.9, 52.9, 93.6,
65.9, 64.1, 92.0, 96.4, 116.8). Greedy 400k sorted on cohort 1009: 46, 52,
70, 76, 92, 94, 118, 119, 124, 126, 144, 158, 160, 167, 168, 168, 182, 186,
246, 298, 321, 346, 370, 394, 404.

Actual-initial controls of the 400k life on cohort 4000001009: sampled 1.286
(28 episodes), greedy 2.071 (28); identical weights to the 200k life's
initial checkpoint. Budget scaling of the combined learner, 200k to 400k:
cohort 1009 sampled 66.5 to 85.6 and greedy 103.0 to 135.6; cohort 22003
sampled 61.7 to 102.8 and greedy 104.0 to 134.5. Retained CDP, 200k to 400k:
cohort 1009 70.8 to 92.8 and 143.7 to 173.7; cohort 22003 70.5 to 120.6 and
94.6 to 150.8.

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

At 400k the retained CDP learner leads the combined learner on both cohorts
and in both modes (92.8 / 173.7 and 120.6 / 150.8 against 85.6 / 135.6 and
102.8 / 134.5), and does so in two-thirds of the training time. The
combined learner's 200k lead on cohort 22003 does not survive the second
200k: CDP gains more from the extra budget (sampled 70.5 to 120.6 on that
cohort) than the combined learner (61.7 to 102.8). The pixel auxiliary at
scale 1 therefore does not improve the retained CDP recipe at either fixed
budget once both cohorts are counted; it narrows cohort variation and
raises the 200k greedy score on the new cohort, nothing more.

The pure pixel world's greedy edge survives the 400k budget on both
cohorts (185.2 against 173.7 and 158.6 against 150.8), narrower than at 200k
(7% and 5% against 9% and 60%), and its sampled deficit persists (83.6
against 92.8 and 100.9 against 120.6). Across every seed, cohort and budget
tried today, five pairs in all, the pixel objective beats centered CDP in
greedy play and loses in sampled play, at 1.5 times the training time per
action. No learner reached a two-wall episode; the pixel world's 404 is the
closest. The remaining question is the sampled mode: the pixel-world
learners have the sharper greedy policies and the wider gap between their
greedy and sampled scores, so their policy distributions are flatter where
it matters; whether actor unimix, the entropy scale or the imitation weight
closes that gap is a separate experiment, not a conclusion here.
