# Bounded Qbert actor/critic rate check

Select one fresh execution and early-cost check within the unfinished five-game
goal after the H3 comparison and all original services close. The retained
H5/R24 recipe passes the15k/90% gate on8039 but misses on10007: greedy14,339.583
and sampled13,761.458,each24/24 first pyramids. More replay work,shorter imagined
rollouts and temperature0.5 have not improved its absolute primary score.

Test behavior learning rate **4e-5→8e-5**,changing actor and critic adaptation
together. Keep world/base/encoder rates4e-5 and dynamics4e-4. This tests whether
faster behavior adaptation helps at fixed data and update budgets; it does not
identify an actor-versus-critic bottleneck. Earlier Pong B16 behavior8e-5 improved
11.875→12.375 but missed15 and trailed B8;that is neither Qbert evidence nor a
universal rate recommendation. No new loss or native implementation is needed.

Keep centeredCDP/12M48/Auto/N8/B16/T16/H5/R24,greedy collection,sampled posteriors,
actor-unimix0,disagreement1,expert100/unit weights/world gradients,all-state
corrective labels,20% episode continuation guidance/cap1920,full18/sticky.25/
repeat4 and one GPU resize from native pooled frames. Use native447fe50/SHA35ad0db
and exact retained runneraba4300d. Use the qualified equivalent Python teacher
overlay at `/x/Temp/kindle-qbert-color-masks-20261101/runtime`,teacherSHAfa261c53.
Its130 CPU tests,472 matched snapshot decisions and full200k label/pixel/action
replay already pass;reuse that qualification. Teacher bytes and CPU cost change;
any whole-training timing comparison must disclose this,without claiming an
isolated native speedup. Original runtime and completed evidence remain intact.

Reuse development learner10007/evaluation4000010007. Fix8,192 aggregate actions,
733 updates,first update376 and residual credit.75. Restore final weights for
1,024 greedy frozen actions withzero updates. R24 schedules remain100k/9,340
and200k/18,715. No old weights,experience or checkpoint selection.

Require357 finite/control-matching initial tensors,matched pre-update semantic
events,357 invariant frozen tensors,exact config/cadence/native/runner/teacher,
finite supervised reports,changed CNN/reward/actor/value weights,full training
pixel/action/corrective-label replay and both host guards. CPU preflight parses
both actual CLI paths,checks valid/invalid audit configs,the three schedules,
357 schema tensors,retained control inputs and all16 overlay package files
without native or emulator construction. Preserve failures;never repeat a
completed GPU stage to repair CPU evidence.

Training/frozen/audit caps180s/60s/180s;service480s. Require
`check_training_seconds * 200000 / 8192 * 1.10 <= 2100` (35 minutes).
A pass establishes execution and early cost only. A full learner needs separate
selection,fresh200k actions,retained four-file100k recovery and final/actual-initial
greedy-primary and sampled24-natural-episode assessments. Keep the15k/90% gate
and retained10007 controls unchanged. Reused development seed/cohort limits
reliability claims. No automatic full learner,rate sweep,extension or retry.

Root `/x/Temp/kindle-qbert-behavior8e5-20261101/native-check`,trial
`qbert-12m-48-b16-h5-r24-encoder4e5-behavior8e5-greedy-expert100-check-seed10007`,
service `kindle-qbert-behavior8e5-check-20261101`.

## CPU qualification

Preflight passes15:12:46 UTC November1 in1.101s (1.133s wrapper):two actual
CLI paths,18 config cases,357 schema tensors,442 retained control inputs,all
three schedules and16 overlay package files. Native/emulator constructions
arezero. Source review catches the copied preparer's historical H3 protocol
constant before selection;the explicit `select_behavior_check.py` entry point
binds this protocol while preserving all CPU-qualified sources. No failed
check or repeated preflight/GPU stage occurs.
