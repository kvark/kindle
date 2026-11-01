# Settling cost sharply lowers the edge-reset teacher's score

Retain the current Breakout teacher. Adding the fixed0.5 impact-segment
displacement penalty lowers mean reference return **185.375→80.375** on the
eight retained first episodes. Only two lanes improve, and both policies have
zero two-wall successes. The candidate fails the predeclared10%-mean/five-lane
screen for a separate standard24 assessment. No expanded assessment,
coefficient retry, native teacher change or learner follows.

| Lane | Retained edge-reset teacher | Edge reset plus settling |
| --- | ---: | ---: |
| 0 | 151 | 50 |
| 1 | 97 | 112 |
| 2 | 256 | 66 |
| 3 | 226 | 64 |
| 4 | 204 | 60 |
| 5 | 331 | 105 |
| 6 | 82 | 135 |
| 7 | 136 | 51 |

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-edge-settling.md)
combines the current edge-reset rule with the previously tested settling cost.
The older pre-edge-reset comparison improved only144.875→152.750; that finding
does not transfer to this stronger control. All other response coefficients,
detectors, target geometry, planning horizon and action costs stay fixed.
No refit, timing-window search or three-pixel aim experiment is repeated.

The reused cohort limits generalization. This is a hand-written pixel reference,
not native Kindle learning. The result rejects this fixed combination; it does
not show that all uncertainty-aware controls fail or identify why the return
drops. Recent response ambiguity and contact errors remain observations.

Collection and exact replay finish **10:47:56 UTC November1**,source`d105cee`;
the full controller takes40.337s. Collection12,880 actions/51,509 frames takes
24.952s and replay10.409s. Four fixed cost fixtures, exact policy-diff
checking, source/ROM/package checks and the5.174s preflight pass without emulator
construction. Collection uses native pooled RGB/full18/sticky.25/repeat4 with
unchanged caps. All selected episodes end naturally. Full pixel/action/debug/
reward/boundary replay and its video pass. All24 declared inputs and442 active
Qbert inputs remain unchanged, preserving PID781585/invocation
`539c22282c4e42c7a22bc150961ef963`. A separate scalar recount verifies the16
control/candidate selected episodes. No native model, RAM-based decisions,
GPU work, learner updates, failures or repeated original stages occur.
Retain CPU co-host timing without correcting the Qbert run.

Evidence: `/x/Temp/kindle-breakout-edge-settling-20261101`.
`comparison.json`, `completion-review.json` and `independent-recount.json`
retain selection, timing and hashes; `edge-settling-eight/review.json` retains
all completed episodes, partial tails and replay evidence.

Complete: do not rerun collection, replay or passed preflight. The five-game
goal remains unfinished.
