# Qbert RGB64: the stronger learner stays ahead on both evaluation cohorts

The [fixed crossed assessment](../experiments/2026-11-01-cdp-qbert-rgb64-cross-cohort.md)
keeps learner10007 ahead of12011 on both retained environment cohorts. Their
mean-return gaps are3,728.125 and3,540.625 points. Learner10007 passes the
unchanged15,000/90%-first-pyramid gate on both cohorts;12011 misses15,000 on
both, despite24/24 first pyramids in every trained cell. The evaluation-cohort
change alone does not explain the original learner gap.

| Frozen trained agent | Cohort4000010007 | Cohort4000012011 |
| --- | ---: | ---: |
| Learner10007 | 15,710.417;24/24;passes | 16,218.750;24/24;passes |
| Learner12011 | 11,982.292;24/24;fails score gate | 12,678.125;24/24;fails score gate |

| Actual-initial agent | Cohort4000010007 | Cohort4000012011 |
| --- | ---: | ---: |
| 10007 | 1.042;0/24 first pyramids | 4.167;0/24 |
| 12011 | 156.250;0/24 | 168.750;0/24 |

| Initial-subtracted mean | Cohort4000010007 | Cohort4000012011 |
| --- | ---: | ---: |
| 10007 | 15,709.375 | 16,214.583 |
| 12011 | 11,826.042 | 12,509.375 |

The gaps remain3,883.333/3,705.208 after subtracting each agent's own initial
result. Learner10007 has the higher lane mean in6/8 and7/8 lanes on the two
cohorts. Full lane means remain in `comparison.json`. These dependent cells
compare frozen agents and environment cohorts; they do not isolate learned
weights from checkpoint configuration or posterior randomness, establish a
population-wide ranking, or add an independent learner confirmation. The
original12011 failure remains unchanged. Do not select whichever cell passes.

Together with the [current training/frozen phase analysis](2026-11-01-cdp-qbert-rgb64-phases.md),
this directs attention toward differences in later-play decisions between the
trained agents. Both collected substantial later-phase experience, and the
weaker agent had higher late native movement agreement on its own online
states. Those observations still do not distinguish representation precision,
optimization, teacher limitations or coverage of particular difficult states.
They do not select another loss, weight sweep or third learner. The earlier
NOOP-downweight comparison remains negative and should not be repeated merely
because aggregate movement agreement is lower than wait agreement.

## Fixed evaluation and evidence

Four new greedy arms evaluate each final and actual-initial checkpoint on the
other cohort. Four completed diagonal arms are reused without repeating their
GPU assessments. Keep sampled categorical posteriors, each checkpoint's own
config, native447fe50/SHA35ad0db and exact runneraba4300d. Guidance and labels
are off. The original200k training times25m09.797s/25m27.366s remain unchanged;
no learner, build, model/loss change or training update occurs.

Every new arm selects the firstthree natural episodes per eight lanes=24,
preserves357 finite tensors, and haszero updates/truncations. All611 pinned
inputs, source checkpoints, full recorded-outcome replays/videos, host guards,
matrix checks and original invocation/terminal collection pass. Excess episodes
and partial tails remain retained. Ordinary frozen logs lack live pixel hashes;
outcome replay cannot prove unavailable live-image equality.

The new arms total140,488 actions/561,791 frames/**117.898913s**,zero updates.
Original main succeeds **21:51:01.430 UTC November1** after3m40.509s including
audits/replays; original collector completes **21:51:22.608**, source63dd44a.
MainPID850983/invocationef4177111f444dea8f2d905f7f800544 and collectorPID850986/
invocation053db596d5a44ccab149a1d1f3063bb8 remain the unique original services.
Both services are terminal success; no stage fails or repeats. The earlier CPU
preparation co-host times remain recorded without correcting Breakout's wall.

Root `/x/Temp/kindle-qbert-rgb64-cross-cohort-20261101` is complete; do not rerun
the crossed arms, retained diagonals, successful preparation or collector.
No GPU allocation remains active at closure. No extension, cap change, retry,
checkpoint selection or automatic successor is selected. The five-game goal
remains unfinished; the fixed RGB64 recipe still passes on one of two learners.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `b3daecda9ced211e70ee1ca60cf2f3bf0eacb8f4de3ba4e8a86ddb1f740f2b75` |
| Comparison | `514c9b4053408f838ab9e6cf3679ccab9d2edcc18a50e21f7855f44eb894c6a9` |
| Original completion | `5d9842de922c277ccee7e965edf4fad863adda48b9238f6231b1bcd6b62fa4c0` |
