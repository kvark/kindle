# Qbert cohort choice changes a near-threshold gate, while the learner gap remains

Learner14009 stays ahead of16001 on both retained evaluation cohorts. However,
16001 scores **15163.542** on the development cohort and **14913.542** on its
original cohort, passing the15k mean requirement only on the former. Cohort
choice affects this near-threshold outcome, but does not explain the entire
learner gap. Allfour trained cells clear24/24 first pyramids.

| Frozen trained agent | Cohort4000014009 | Cohort4000016001 |
| --- | ---: | ---: |
| Learner14009 |16525.000; passes|15906.250; passes|
| Learner16001 |15163.542; passes|14913.542; fails mean gate|

The original fresh16001 confirmation **remains failed**. These cells reuse the
same two trained agents and are dependent assessments, not new learner
confirmation. Do not choose whichever cohort passes or report three successful
cells as independent reliability. The fixed recipe still passes its original
primary greedy assessments onone of two learners. No third learner, extension,
mode sweep, loss change or automatic successor is selected.

| Actual-initial agent | Cohort4000014009 | Cohort4000016001 |
| --- | ---: | ---: |
|14009|147.917;0/24 first pyramids|150.000;0/24|
|16001|66.667;0/24|86.458;0/24|

| Initial-subtracted mean | Cohort4000014009 | Cohort4000016001 |
| --- | ---: | ---: |
|14009|16377.083|15756.250|
|16001|15096.875|14827.083|

The within-cohort raw mean gaps are **1361.458 /992.708**, and initial-subtracted
gaps **1280.208 /929.167**. Learner14009 has the higher mean in5/8 and6/8 lanes.
Per-lane results remain in the matrix artifact. These compare frozen agents
and environment cohorts; they do not isolate weights from checkpoint config,
posterior randomness or differing visited states, or establish a population
ranking. The task gate applies to raw final returns, not initial-subtracted means.

The preceding [phase diagnosis](2026-11-02-cdp-qbert-replay-value1-phases.md)
places the original score gap after the first pyramid and finds substantial
later-play training exposure with similar late native movement agreement.
The crossed result adds a cohort contribution to the near-threshold failure,
while preserving a later-play difference between agents. It does not identify
representation precision, critic calibration, teacher limitations or difficult
state coverage as the cause, and does not select another loss intervention.

## Fixed evaluation and complete evidence

The [protocol](../experiments/2026-11-02-cdp-qbert-replay-value1-cross-cohort.md)
runs four new greedy final/actual-initial crossed arms and reuses the four
completed diagonal arms without repeating their GPU work. Each new arm selects
the firstthree natural episodes per eight lanes=24; all havezero updates and
zero truncations. Keep sampled categorical posteriors, each checkpoint's own
config, native447fe50/SHA35ad0db, runnere304af89 and the matching color-masks
runtime. Guidance and labels are off. Training remains200k/18715 updates and
25m36.935s/24m45.754s for14009/16001. No build, new learner or training occurs.

All **957 pins**, source checkpoints,357 finite/invariant tensors per new arm,
full recorded-outcome replays/videos, host guards, matrix checks and original
invocation/terminal collection pass. Excess episodes and partial tails remain
retained. Ordinary frozen logs lack live pixel hashes; outcome replay cannot
prove unavailable live-image equality. The new arms total **138952 actions /
555639 frames /115.405828s**, withzero updates.

Original mainPID890878/invocation`68f3f48fbc714bfbbb4a0701a1ea3799` launches
03:43:03.876 UTC November2/sourceb883162 and succeeds **03:46:51.221** after
3m47.345s including audits/replays. Original collectorPID890881/invocation
`ae591eeba4444a5cafe8c9f13d8b7aa3` succeeds **03:47:07.244**. Both original
services are terminal success. No preparation, GPU, replay, audit or collector
stage fails or repeats, and no cap changes after selection.

The separate terminal reader passes **03:47:46.833** in3.310s (3.330s wrapper),
verifying all957 pins, original journals/identities, completion files, all eight
matrix cells, unchanged diagonal decisions, lane arithmetic and new-arm counts.
It constructs no native agent/emulator and repeats no original stage. The
earlier10.229s/8.404s CPU preparation co-host costs remain recorded against the
completed Breakout comparison without correcting its wall time.

Root `/x/Temp/kindle-qbert-replay-value1-cross-cohort-20261102` is closed. Do not
repeat the crossed arms, retained diagonals, successful preparation, collector
or terminal reader. No GPU allocation remains at closure; check live state
before separately selected work. The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`7e472c0b7288fab9a4efda90b239f1184d93abeedf920893e0c9066527de8c8c`|
| Comparison |`0d1089d34c3cd5bb1d7ed4d24c50e73c9d50c1f535a207e076010c95e71afb03`|
| Original completion |`cd59f0ffcf7907724d490623f777037b3730acc80243a512155c90a7e85bac43`|
