# Qbert12M/RGB64 clears the greedy gate in 25m10s on one learner

The fresh RGB64 learner reaches **15,710.417 greedy return and24/24 first
pyramids**, passing the unchanged mean15,000/90% gate. This is **9.56% above**
the retained RGB48 control's14,339.583. Sampled return rises3.55% to14,250 but
still misses the mean gate. Retain this useful development result; one learner
on a reused seed/cohort does not establish reliability. No extension, automatic
confirmation or successor is selected by this completed comparison.

| Frozen mode | RGB48 final / own initial | RGB64 final / own initial | RGB64 first pyramids | RGB64 gate |
| --- | ---: | ---: | ---: | --- |
| Greedy, primary | 14,339.583 /1.042 | 15,710.417 /1.042 | 24/24 | Pass |
| Sampled | 13,761.458 /171.875 | 14,250.000 /195.833 | 24/24 | Fail mean |

All four RGB64 arms select24 natural episodes, with guidance off, zero updates
and no truncations. Actual-initial weights clear no pyramids. The four arms
total156,312 actions /625,086 frames /146.320s. Equal greedy initial scores do
not imply equal trajectories: both sampled and greedy initial traces differ
across geometry, as permitted by the protocol.

Training completes **200,000 actions /799,823 frames /18,715 updates in
25m09.797s**, below30 minutes. It takes11.90% less total training time than the
original RGB48 control's28m33.661s. Initial-subtracted return/minute rises24.35%
greedy and17.38% sampled, but the qualified faster teacher also changes CPU cost.
Recorded learning time actually rises16.73%,1051.583→1227.550s, while acting
falls566.324→185.532s. These are wrapper wall timers, not isolated kernel times;
do not attribute the total speedup to resolution or native optimization.

The only learner-setting change is RGB48→RGB64 at12M/H5/R24/encoder4e-5/base
behavior4e-5/greedy collection/expert100. Encoded width576→1024 and parameter
count14,123,473→16,334,353 change posterior/prediction-head shapes and
initialization together. Native447fe50, runneraba4300d and fast teacherfa261c53
are unchanged. The older unaided4M RGB64 negative result remains separate.
No isolated representation or capacity mechanism follows.

The final50k-action imitation CE changes.051060→.049491, a descriptive training
fit measure. Actual guidance overrides44,600 actions (22.3%); online guided
scores are not unaided competence. Full pixel/action/corrective-label replay
verifies all200k training actions. All521 pins,357 bounded-matching initial
tensors,47 pre-update transitions,18,715 finite reports,100k/9,340 four-file
recovery,357 invariant tensors in every frozen arm, frozen outcome replays,
videos, guards, curves, coverage, exposure and comparison pass. Ordinary frozen
logs lack live pixel hashes; their replay does not verify unrecorded live images.

All five original services succeed: main **20:11:19**, postprocessing
**20:11:52**, terminal collection **20:12:24 UTC November1**, source`b8de34f`.
No failure, retry, cap change, repeated GPU/replay/collector stage or build occurs.
Startup and the two separately reported Breakout CPU diagnostics co-hosted;
retain their measured costs without correcting training wall time. The generated
comparison plot is inspected and explicitly labels online versus frozen results.
No learner or GPU allocation remains active at closure. Check live state before
future work; do not rerun this learner or its collectors. The five-game goal
remains unfinished.

## Bounded qualification and full-run provenance

The bounded check completes **8,192 actions / 32,760 frames / 733 updates in
57.956s**. The 1,024-action/4,096-frame frozen stage takes **.849s**, with zero
updates and no natural completed episodes. The conservative 200k projection
is **25m56.425s**, within the fixed 30-minute limit. This establishes execution
and early affordability, not a gameplay improvement.

All 511 pinned inputs, 357 source-derived initial tensors, finite supervised
reports, changed CNN/reward/actor/value parameters, 357 invariant frozen tensors,
full training pixel/action/corrective-label replay and both host guards pass.
First update376/residual .75 match the declaration. Actual guidance is 3,179
actions (38.806%); the six completed training episodes and unfinished guided
returns are not frozen competence.

RGB48→RGB64 is the only learner-setting change against the retained supervised
12M/H5/R24/encoder4e-5/base behavior4e-5/greedy-collection/expert100 recipe.
Encoded width rises576→1024 and parameters14,123,473→16,334,353. Image geometry,
posterior/prediction-head dimensions and initialization change together; there
is no cross-geometry initial-value or action-equality claim. The earlier
[unaided4M RGB64 result](2026-10-25-cdp-cheap-qbert.md#rgb64-costs-more-and-lowers-frozen-quality)
was negative and remains unchanged. Neither result isolates spatial information.

Native447fe50, runneraba4300d and the qualified faster teacherfa261c53 remain
unchanged at `/x/Temp/kindle-qbert-color-masks-20261101/runtime`; no build.
The original RGB48 control used the slower equivalent teacher, so wall-time
changes are not an isolated geometry or backend effect. Keep one GPU resize
from native pooled RGB and Auto precision.

CPU preflight passes two CLI paths, 19 configuration cases, schedules,
all 357 retained Qbert12M48 and 357 Breakout12M64 shapes, 442 control inputs and
16 runtime files in 1.161s/1.196s wrapper. No native agent or emulator is
constructed. Launch19:32:42 UTC on November 1, source4c7293c, originalPID836463/
invocation65b044b246934957af5dec3dffb418f5. Main succeeds19:34:02; original
terminal collection completes19:34:32. No failure, cap change, GPU-stage or
collector repeat occurs. Brief read-only source/status work co-hosts with the
check; retain measured time without correction. The bounded check is complete.

A [fresh full200k comparison](../experiments/2026-11-01-cdp-qbert-rgb64-200k.md)
launches at 19:38:45 UTC on November 1 with 100k/9,340 four-file recovery and
final/actual-initial greedy-primary plus sampled 24-episode arms. The original
15k/90% gate stays, against retained14,339.583/13,761.458. Its CPU qualification
passes five CLI paths, 28 config cases, teacher/mode/header/recovery/retention
checks, 49 paths and 17 scripts in 3.493s/3.594s wrapper. All 511 bounded and
442 control inputs pass. No bounded experience is reused in the full learner.

The full learner uses source b8de34f, original PID836919 and invocation
425aae9cee2541fe9623b753225af57f in `kindle-qbert-rgb64-200k-20261101`.
All original observers subsequently succeed. Startup review at19:41:33 passes
all521 pinned inputs,357 initial
tensors matching the bounded run, 47 matching pre-update transitions and
2,074 finite supervised reports through 22,504 actions. First six update actions
are 376/392/400/408/424/432. All 47 pre-update transitions differ from RGB48;
cross-geometry equality is not required. The review takes 1.465s on CPU while
training runs; retain the co-host timing caveat without correction. This is a
startup check, separate from the completed frozen results above.

The original 100k recovery observer succeeds at 19:51:22 UTC, with the declared
100,000 actions / 9,340 updates, all 357 finite tensors and 9,405 finite reports
through 100,696 logged actions. All four retained checkpoint files match their
receipt hashes, totaling 206,067,159 bytes. Retention takes .907s without changing
the live source. The checkpoint preserves weights/metadata, not replay, live
belief, environments or RNG. No recovery stage is repeated.

Full declaration SHA256:
`3a1f56ed897d49ef58eedae0f315309452e2b8513d3eb7f601dcf23db025351e`.
Startup evidence is `full/qbert-12m-64-b16-h5-r24-encoder4e5-greedy-expert100-200k-seed10007/startup-review.json`.

Evidence root `/x/Temp/kindle-qbert-rgb64-20261101`. The bounded trial is
`native-check/qbert-12m-64-b16-h5-r24-encoder4e5-greedy-expert100-check-seed10007`.
The five-game goal remains unfinished. Check live state before GPU work.

- Bounded declaration: `12a7a8ab4e034b3d5549768d3e1660cc08589078a39f073be3ef5400913fa7ab`.
- Bounded review: `a5da340483d764c82d0f6c13e4557aca9fba5338618b906dabaf58493921449e`.
- Bounded guided replay: `02a8805e9221ab5ba656bbaf1ff9501488454e98ad1db566b045fcf5e4d778f9`.
- Full learning review: `5e4b7c15701e6aac74062473895717d1adc79addbe8ae3626c0caf5e405b0c47`.
- Full guided replay: `e53df449bf5ca0324f566d16d8eab63acf7f63388e1dccbbd57177762859fca6`.
- Full comparison: `ae3f04296dca1f994444210c9b08b26db733bb4de399614718cccc6fdf855aad`.
- Original postprocessing: `d1d054409c3ddf8c16c54d8b5913f32c3511686a133f214e68285b0fbbaad09b`.
- Original completion: `fb7d2f6645e84ff5eaab2f7751372ddec79ce8f20622286cfcded0aa92e4dded`.
