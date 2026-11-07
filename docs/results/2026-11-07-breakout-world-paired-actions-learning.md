# Paired-data Breakout world learner: bounded run active

The [paired-data protocol](../experiments/2026-11-07-breakout-world-paired-actions.md)
now has one fresh 8,192-update learner active. This page reports its declared
allocation and passed preparation; **no final learned improvement is established**.
The [collected branches](2026-11-07-breakout-world-paired-actions-data.md) and
[action-gap audit](2026-11-07-breakout-world-action-gap.md) motivate the experiment.
Actor/critic, CDP comparisons and the five-game campaign stay parked.

## Intervention and exact comparison

Reuse native source `85dbe150` / SHA256
`5b1128667e6436b9865c31dedcb50e22318a43250e884046b155e51dc955ad82`
and the motion-rollout matching runtime. Python runner source is
`e751a8838d941fd21af426b2b3565c2889c1eef9`. No native rebuild or new loss.
Keep RGB64/12M/B8/T32/H16/Auto, initialization seed 41001, rates 4e-4, motion
cross-entropy, loss scales and initial tensor identity. The 448 new fitting
branches and sampler replace the prior coverage-only data mixture together;
volume, source weighting and within-batch correlations change. This is not an
isolated action-confounding estimate or fresh confirmation seed.

Four lanes share one fitting parent/repeat index and window start, two sample
original fitting streams, one samples supplement waiting, and one cycles a
seeded terminal-pool permutation including all 84 paired terminal streams.
Half the updates place the parent at the end of observation where branch lengths
allow; the other half sample valid common cuts. Short branches clamp the common
cut earlier. All targets remain real; the first arrival is a context reset,
while incoming source actions and actual terminal/reward targets are preserved.
Repeat0 shares parent RNG across plans; repeat1 uses evolving per-plan RNG.
A repeat index does not imply common RNG or independent experience.

The fixed plan replays 2,097,152 stored arrivals over 464 streams / 60,254 stored
arrivals, with 51,698 distinct stream/index slots selected. That last count
includes repeated shared prefixes across branch files and is not unique emulator
experience. Collection added only 13,068 new actions / 52,122 frames; no emulator
work is part of this learner. Paired lanes expose 488,428 future and 560,148
shared-prefix arrivals, including 436,680 valid intervention suffix targets.
All 112 parent/repeat groups are selected 56–94 times. Exposure totals:

|Item|Declared count|
|---|---:|
|Valid H16 suffix targets|989,057|
|Observed terminal targets|10,892|
|Valid suffix terminal targets|10,562|
|Positive reward rows|32,687|
|Natural source resets|7,872|
|Context and natural resets|73,282|

Keep periodic and dense fitting evaluation on the original sixteen streams;
development uses its original eight streams. Both original selection tables
match the control. All 448 paired fitting origins get a separate one-draw
post-training diagnostic. Unavailable tails are unscored; this is fitting evidence,
not an added held-out gate. Its time is outside the established main loop timer.
Original observed/one/eight-step, 16/32 energy, serving/ending/false-alarm gates
remain unchanged, plus >=90% rendered ball presence in each retained freeflight
plan at eight decisions. Presence alone does not qualify location/calibration.

## Qualification and active allocation

Ten new sampler tests and fourteen existing corpus/coverage tests pass. CPU plan
review takes 3.755s and independently verifies actual window-derived counts,
common cuts, all groups, all paired terminal availability and original evaluation
selections. No model or emulator is constructed during those checks.

The one-update native preflight completes in 27.956s: 517 input pins, 204 matched
initial tensors, eighteen initial/restore arrays, eleven finite losses, exact
suffix-mask/total identities, 65 changed logical parameters and 204 frozen tensors
pass. The retained H0 initial checkpoint restores exactly. The host guard/audit
passes nine sealed pins with no unfinished children or recovery. This verifies
execution, not learned competence or a raw-gradient measurement.

Root `/mnt/data/kindle-breakout-world-paired-actions-20261107`. The separate
`paired-learning-protocol.json` pins 524 inputs; `paired-learning-queue.json`
allocates train 3600s and restore 600s under
`kindle-world-paired-actions-learning-20261107`. Preserve all collection/preparation
failures and successful receipts; none of the closed collectors/learners repeats.
The fresh learner starts from initial tensors, not the one-update fixture.

After successful execution, independently audit the exact plan, 8,192 loss rows,
frozen tensors, original initial/dense controls and restore before declaring
new-final-checkpoint-only frozen branches (1800s). No automatic successor,
unchanged extension or sweep. Precise long futures, calibrated rewards/endings,
brick memory, late/tunnel/second-wall coverage, untouched families and fresh
confirmation remain required for the full world-model goal.
