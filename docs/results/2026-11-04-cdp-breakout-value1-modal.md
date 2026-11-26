# Frozen modal posteriors raise the observed Breakout mean, without passing the selection rule

The retained trained checkpoint scores **175.708 with modal posterior states
versus 143.667 with sampled states**, keeping actions greedy in both arms.
The observed increase is 22.30%, but the paired lane-bootstrap return difference
is +32.042 with 95% interval **[−25.750, 78.959]**. Both have zero of 24 two-wall
successes. The preset improvement rule fails because the interval includes zero.
Retain sampled posteriors; this does not establish a reliable modal benefit or
show that posterior sampling is harmless.

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-value1-modal.md) tests
trained inference only. No learner, loss change or initial-model GPU arm runs.
Original training remains 200,000 actions / 12,477 updates / 21m27.910s.
There is no paired learning-gain/min estimate without an accepted modal initial.
The original mean864/90%-two-wall gate and five-game goal remain unfinished.

## Fixed comparison

| Posterior mode, greedy actions | Mean, first 3 natural episodes/lane | Two-wall | Evaluation actions | Executed frames | Evaluation time |
|---|---:|---:|---:|---:|---:|
| Sampled control | 143.667 | 0/24 | 43,424 | 173,660 | 27.616s |
| Modal candidate | 175.708 | 0/24 | 44,544 | 178,139 | 28.995s |

Each arm completes 25 natural episodes; the first three per lane select 24.
The extra episode and all unfinished tails remain recorded. There are zero
truncations and zero learning updates. Total evaluation is 87,968
actions / 351,799 frames / 56.610s, excluding
construction, CPU replay and auditing. Construction takes 2.839s
and 2.948s respectively. Timing is reported separately
from the unchanged training cost; it is not a backend speed comparison.

| Lane | Sampled episode returns | Modal episode returns | Sampled mean | Modal mean | Difference |
|---|---|---|---:|---:|---:|
| 0 | 41, 137, 273 | 63, 223, 123 | 150.333 | 136.333 | -14.000 |
| 1 | 126, 114, 148 | 365, 57, 58 | 129.333 | 160.000 | +30.667 |
| 2 | 68, 87, 24 | 307, 116, 88 | 59.667 | 170.333 | +110.667 |
| 3 | 138, 153, 26 | 322, 232, 65 | 105.667 | 206.333 | +100.667 |
| 4 | 344, 206, 166 | 92, 91, 122 | 238.667 | 101.667 | -137.000 |
| 5 | 330, 71, 117 | 329, 229, 19 | 172.667 | 192.333 | +19.667 |
| 6 | 124, 161, 54 | 214, 97, 161 | 113.000 | 157.333 | +44.333 |
| 7 | 79, 199, 262 | 281, 184, 379 | 180.000 | 281.333 | +101.333 |

The decision uses 10,000 paired eight-lane bootstrap resamples, seed0. Six lanes
improve and two regress, including a −137-point lane. The mean exceeds the
10% threshold and two-wall successes do not decrease, but the positive-interval
requirement fails. These are reused development lanes on one retained learner,
not independent training seeds or evidence for a global posterior default.

## Compatibility and frozen-state evidence

Both arms restore the same retained replay-value1 final checkpoint from
`/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009/final`.
Its metadata SHA256 is `fa2fab6babe83d9a6714e3d93b6f77b5d8cbaef400d3bc7721b8a337b0c12be4`.
The recipe stays single-image12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5,
dynamics4e-4, behavior8e-5, replay-value1, expert100. Evaluation uses full18,
sticky.25, repeat4, noops0, native pooled RGB with one GPU64 resize and
seed4000001009. No guidance or expert labels enter evaluation.

Qualified nativeb76f91c/SHA7555bf9 and runnerfac1ff7 are reused without rebuilding.
They differ from the learner's native447fe50/runnercf4ea238. The sampled control
reproduces all **5,478 semantic action/episode/reset events**,
all final counters/outcomes/tails and the retained 143.667 score before the modal
arm starts. Only finite wall-time fields and source/path identity are excluded.
This establishes compatibility for this checkpoint/cohort, not every policy.

All **357 tensors per arm** remain finite and bitwise unchanged: world280,
behavior66 and slow-value11. Configuration, training counters, return statistics
and remaining checkpoint metadata stay unchanged; only the environment counter
advances with frozen actions and serialized file hashes may differ. Both full
recorded-action/reward/frame/reset/task replays and lane0 videos pass. Ordinary
frozen logs lack live pixel hashes, so no live-image replay equality is claimed.

All **431 current / 69 qualification / 338 retained pins** pass, with exact saved
runner and replayer bytes resolving their historical canonical paths. These
sets overlap. Both host guards pass without recovery or unfinished children.
Across 33,050 native memory snapshots,
minimum headroom is 14.526GiB, above the fixed 2GiB requirement.

CPU command qualification takes 3.784s with no native import,
agent or emulator construction. It checks both actual commands, rejects two
invalid frozen commands and 19 malformed audit fixtures, and checks six decision
fixtures. The earlier successful 3.884s preflight is retained; before launch,
queue acceptance was adjusted so an otherwise auditable capped arm could be
replayed before protocol rejection. The final check passes; caps and decision
rules never change. No GPU qualification or completed stage repeats.

The 357 saved initial tensor values match those from the older Breakout model
on CPU. Encoder rate, behavior rate and replay-value scale differ in metadata;
this provenance check supplies no accepted modal initial score. The [older
four-arm assessment](2026-10-29-cdp-greedy-posterior.md#breakout-trained-score-rises-slightly-the-paired-assessment-fails-its-protocol)
remains `closed_protocol_failed` because five initial-modal episodes timed out.
Its original failures and CPU closure remain unchanged; none of its GPU stages
is restarted.

## Original closure and retained evidence

Root: `/x/Temp/kindle-breakout-value1-modal-20261104`.
Selection source: `2afd0dc563a9d90a845c91c1482bfee20307ff57`.
Original main: `kindle-breakout-value1-modal-eval-20261104`, PID1094832,
invocation `e1d8c3006ec94c1ca8a7498712765526`. It launches 01:04:49.931
and completes comparison 01:06:38.808 UTC November4. Original
terminal collection completes 01:06:43.697 in 3.562s;
both services finish successfully. All 25 completion artifacts verify.
A separate read-only summary takes 1.534s and also verifies native
memory snapshots; it reruns no experiment or replay stage.

Key artifacts are `declaration.json`, both `*-review.json` and `*-replay.json`,
both videos and frozen exports, `comparison.json`, `completion-review.json`,
`terminal-journal.jsonl` and `closure-summary.json`. Completion SHA256:
`028c52f47397802a0fa278f69c851ff2b7f84b3396ffc7dca34a2a01725100f2`.
Comparison SHA256: `dc15faf97c64d2dbf096cc2d6db36ffdf58daf3a9e1cbb50e2a2073b44c3dda4`.
The original observer identity and terminal journal remain retained.

No GPU allocation remains active at closure. No retry, cap increase, initial
assessment, inference sweep, learner or automatic successor is selected.
Changing posterior states changes the inputs of a policy trained with sampled
states; this comparison does not isolate perceptual precision or the mechanism
of the observed return difference.
