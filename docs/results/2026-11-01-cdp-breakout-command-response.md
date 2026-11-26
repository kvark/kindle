# Command-dependent teacher gains shrink on the fresh cohort

The fresh comparison raises teacher mean return **250.875 → 273.125 (+8.87%)**,
with improvements in **five of eight lane means**, but misses the predeclared
10% mean-gain requirement. Both arms still have zero of 24 two-wall successes.
Retain the current teacher; no candidate integration, parameter retry, further
cohort or learner is selected. The development gain remains valid for its reused
cohort, without establishing the desired fresh-cohort improvement or Kindle
learning. The five-game goal remains unfinished.

| Cohort | Control mean | Candidate mean | Gain | Improved lanes | Fixed screen |
| --- | ---: | ---: | ---: | ---: | --- |
| Development, seed4000001009 | 225.083 | 295.333 | 31.21% | 6/8 | Pass |
| Fresh, seed4000017029 | 250.875 | 273.125 | 8.87% | 5/8 | Fail |

Each mean uses the first three natural episodes in each of eight lanes, with
no truncations. Every arm has zero two-wall successes. These are CPU reference
teachers using game-specific pixels and fixed response models, not trained
Kindle policies. A fresh cohort does not establish broad seed reliability.

## Prediction fit

The fixed 36-coefficient residual model passes its declared prediction screen.
The gain is small: overall held-out MAE falls .21% and RMSE falls 3.47%; near
descending balls, MAE falls 1.01% and RMSE falls 3.82%. This does not establish
better gameplay or a causal explanation of missed contacts.

| Subset | Rows | MAE, control → candidate | RMSE, control → candidate |
| --- | ---: | ---: | ---: |
| Training interiors | 19,146 | 1.862086 → 1.873721 | 3.107795 → 3.011625 |
| Held-out all | 13,102 | 1.976923 → 1.972683 | 3.287560 → 3.173417 |
| Held-out near descending | 2,955 | 1.948802 → 1.929132 | 3.283574 → 3.157979 |
| Held-out stationary edge | 849 | 2.586030 → 2.343512 | 4.287527 → 3.961072 |

Five of eight held-out lanes have lower MAE; all eight have lower RMSE. Training
MAE increases. All nine command-pair groups have finite solutions and local
recurrence radius below .201. This local check does not prove stability under
command switching. All 37,448 original rows, targets and splits reproduce,
as do the original and six-feature model's aggregate metrics. Exact coefficients,
group sizes, residuals, roots and every prediction remain in the evidence.

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-command-response.md)
uses nine current/previous-command groups with four normalized residual features
each, one F64 ridge fit per group and coefficient 1 throughout. The same 19,146
interior rows train the model; the previously examined episode-2 holdout supplies
13,102 rows. Detector, causal history, edge resets and position clipping remain
unchanged. Earlier rejected fits remain rejected; there is no refit or sweep.

Four CPU fixture groups pass; preflight takes .865s. Fit/analysis takes .470s,
the controller .614s, completing **19:15:51 UTC on November 1**, source8de8a12.
All 16 input hashes remain unchanged. No emulator/native import, GPU work,
learner update, failure or repeated fit occurs. The fit is complete; do not
rerun it.

## Development gameplay improves; the original task gate still fails

The [separate bounded CPU gameplay comparison](../experiments/2026-11-01-cdp-breakout-command-gameplay.md)
raises mean return **225.083 → 295.333 (+31.21%)**, improving six of eight lane
means. Both arms have zero of 24 two-wall successes. Each arm selects three
natural episodes per lane without truncations. The candidate passes the fixed
10%/five-lane development screen, but this reused cohort informed its fit.

| Lane | Retained teacher | Candidate |
| --- | ---: | ---: |
| 0 | 190.333 | 386.000 |
| 1 | 96.667 | 239.333 |
| 2 | 184.667 | 318.667 |
| 3 | 221.000 | 304.000 |
| 4 | 345.000 | 265.667 |
| 5 | 363.667 | 241.667 |
| 6 | 99.000 | 270.000 |
| 7 | 300.333 | 337.333 |

Candidate collection completes 51,752 actions / 206,985 frames in 30.749s.
The 41.522s replay reproduces all actions, debug observations, live input/output
pixel hashes, rewards, boundaries and episode outcomes. Four policy fixture
groups and .454s preflight pass, including all 37,448 saved one-step predictions,
scalar/vector rollout agreement and original-model embedding. All 32 inputs
remain unchanged; the controller completes in 72.478s at **19:21:02 UTC on
November 1**, source2a1c9c4. No failure, stage repeat, control rerun, native/GPU
work or learner update occurs. The development comparison is complete.

## Fresh paired comparison is complete and misses its screen

The [fixed fresh protocol](../experiments/2026-11-01-cdp-breakout-command-fresh.md)
keeps both policy implementations and all coefficients byte-for-byte unchanged.
The shared runner changes only the declared seed fields from the original
header. Its .238s preflight passes 39 inputs, exact policy identity, common
runner, seed-only header changes and episode resets; selection pins 41 inputs.

| Fresh lane | Control | Candidate |
| --- | ---: | ---: |
| 0 | 192.333 | 343.667 |
| 1 | 216.667 | 230.000 |
| 2 | 311.667 | 293.667 |
| 3 | 269.667 | 214.333 |
| 4 | 216.667 | 263.333 |
| 5 | 238.000 | 280.667 |
| 6 | 380.667 | 326.333 |
| 7 | 181.333 | 233.000 |

Control collection completes 55,280 actions / 221,090 frames in 32.304s;
candidate collection completes 64,560 actions / 258,195 frames in 35.377s.
Both full replays recompute every decision/debug record and match all live
input/output pixels, rewards, frames, boundaries and outcomes. Both videos
remain retained. Arm wrappers take 66.163s and 78.181s respectively. All 41
inputs remain unchanged; the controller completes in **144.403s at 19:26:13 UTC
on November 1**, sourcef24cd1a. No failure, refit, cap change, repeated stage,
native agent, RAM access, GPU work or learner update occurs.

The fresh screen required at least 10% mean gain, five improved lanes and no
fewer two-wall successes. The candidate satisfies the latter two conditions
but misses the mean requirement; the criterion is not changed after observing
the outcome. This does not prove the candidate has no benefit. It leaves a
smaller positive point estimate without sufficient gain under this decision
rule. Keep the original teacher and preserve both cohorts separately.

Fresh evidence: `/x/Temp/kindle-breakout-command-fresh-20261101`. All stages are
terminal; no native integration, new learner or automatic successor is selected.
Do not rerun completed fits, comparisons or replays.

Evidence: `/x/Temp/kindle-breakout-command-response-20261101`.
The five-game goal remains unfinished.

- Declaration: `fe210b4ad9a80c4b7d0b71894561307a411850266d3d7bbe35d891e456afb64f`.
- Fit review: `7318e2a760a2fefaecbad91353693062b7d3072fe4dae9be9e5a369ac1998ea2`.
- Predictions: `84ca28b05bc7ec46f7ff145c567b5440c3997148026fbcd6d291183f6805b258`.
- Development comparison: `cddaba41b43982e88e5fc43052bd760abd77f1f78f2d9cdfbebad0b4a8659c37`.
- Development completion: `ae308a2cc1ae61655fc01060d40383595e7fc212e91351a5dc43a7b46cd4f581`.
- Fresh declaration: `e9fa4008ef366a15b27f40ebd35f3ada14e57a39db996c0c09ddf347495512ac`.
- Fresh comparison: `44be7f775571ba21a5d0470e568e8b529402fd5e256807ffb7df486a0cd08353`.
- Fresh completion: `1322f125949061b8b860acfdb7ee7f16215dae4e34fda7c9ecedd10a33bfb72e`.
