# Breakout: fixed observation filter, separately learned forecast dynamics

The phase is complete and **fails retention**. Keep the coverage H0 control.
Recorded short forecasts improve and the primary 16/32 ball statistic improves
16.58%, but the eight-step, energy, per-command presence and ending32 gates fail.
Full world-model competence remains unproven. No extension, repeat, sweep or
automatic successor is selected. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-07-breakout-world-fixed-teacher.md).
Root `/mnt/data/kindle-breakout-world-fixed-targets-20261107`.
Source/native/runner `8c6fbae2cada9941c36987fdac5f40d8a98fabe3`, native SHA256
`b40c9d2b42ce18b3aa4e0693d4ef47df1ee555b4364a8929d693a3109a51eb7c`.
The two preceding failed qualification roots remain preserved below.

## Question and intervention

Retained H0 reconstructs observed balls accurately but forecasts drift. This tests
whether separate forecast dynamics can learn a stable, image-informed state
space. It is a native Kindle CNN/categorical RSSM/RGB decoder, with no behavior
learner, external pretrained encoder or object/RAM/phase labels in training.
Pixels remain the inspectable reference; this is not a CDP superiority test.

Warm-start from H0's 8192-update final, preserving all 204 weights/moments and age.
An independently allocated, immutable filter includes the original recurrent
core, encoder and posterior. Predictions filter only their observed prefix, then
use the learned student transition under submitted actions. Decoder and 36
student visual weights stay fixed; 32 dynamics/reward/continuation weights learn.
Targets use the frozen filter and independent per-lane seeds from model seed and
optimizer age. Frozen means fixed functions, not a constant random realization.

Use existing H16 categorical overshooting KL (scale1) plus two-sample recurrent h
energy (scale100), stopped targets/seed, separate prior draws, real submitted
future actions and full Bx16 averaging. Both losses receive independently named
copies of the same reset mask; retain terminal arrivals and mask later episodes.
No future image/state enters a predicted suffix. Image-informed targets are
privileged training targets, not forecasts. Straight-through derivatives remain
biased. One-step prior KL and observed RGB/head supervision remain.

4096 new updates end at12288. RGB64/12M/B8/T32/full BPTT/Auto, mixing.01,
seed41001, rates4e-4, RGB100, KL1/.1/free-nat1, reward/continuation1 stay fixed.
Use the original first4096 4/2/1/1 plan rows, 16 fitting/eight development streams,
32,850 fitting arrivals. All 488,356 valid suffix targets, 4,755 suffix terminals,
5,419 observed terminals, 17,425 positive-reward rows and 32,833 unique arrivals
match. This combines warm initialization, a filter/predictor split, fixed targets
and two losses; it does not isolate a target-drift mechanism. Reused families and
seed are development, not an untouched test or fresh confirmation. No new emulator
actions/frames, data collection, actor or critic.

## Recorded physical precision

Ball centers within one RGB64 pixel, first fixed draw, varying visible support:

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development fixed-filter phase |108/112|98/109|43/109|17/107|10/114|4/95|
| Fitting H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting fixed-filter phase |159/163|156/177|82/180|53/176|40/171|20/179|

Observed >=90% and one-step >=89/109 pass; eight-step >=28/107 fails. All384
matched dense-initial/final origins, 20 identity-array comparisons, 108 complete
cached initial cohort arrays, 24 fixed observed RGB/encoder/posterior arrays and
34 finite export files pass. Observed filtering is exactly preserved; imperfect
forecasting remains even on fitting data. This does not prove information absent
or isolate encoder, sampling, dynamics or optimization as a single cause.

## Fixed-action forecasts and events

Only the new final checkpoint receives the separately declared84 frozen groups:
history16 primary/32 secondary, eight draws, all1/4/8/16/32 endpoints. Original
oracle and control arrays are reused, with no old-model or collection rerun.
The primary category-balanced, survival-weighted observable ball statistic falls
from **.1695022191 to .1414054329**, a16.576% improvement. It misses the unchanged
25% threshold **.1271266643**. This statistic and internal latent losses do not
certify physical competence or calibrated probabilities.

| Check | H0 | New phase | Unchanged gate |
|---|---:|---:|---|
| Freeflight8 presence: NOOP/LEFT/RIGHT/REVERSE |64/63/63/61 of64|63/62/48/64 of64|>=90% each: fail RIGHT|
| Ending16 Brier |.003345|.000253|<=.01: pass|
| RIGHT16 ending probability; oracle0 |.110494|.024970|<=.10: pass|
| Ending32 Brier |.144566|.334714|<=.10: fail|
| Serve NOOP8 false-ball fraction |0|0|<=.10: pass|
| Non-ending cases >5% |0/304|0/304|category means <=5%: pass|

New non-ending maximum is3.265%; ending32 mean is13.548% versus oracle53.125%.
RIGHT32 per-origin ending probabilities are retained individually in
`branch-comparison.json`; the non-ending false-alarm list is empty. All commands
share oracle freeflight ball paths through8, yet RIGHT misses16/64 visible balls.
It is not valid to attribute this discrepancy solely to moving representation
targets: the full filter and its observed outputs were fixed in this phase.

All model/oracle related-pair precision at1/4/8/16/32 changes
70.66/45.08/22.45/20.22/5.90% →71.61/50.72/19.04/24.92/8.42%.
These support-conditioned pairs are not independent episodes or a replacement for
the primary distribution statistic. Eight-step recorded and fixed-plan metrics
move differently; support/actions differ.

`supplementary-metrics.json` retains every action/horizon cell and every individual
non-ending false alarm >5%. Regressions versus H0: ball energy80/210, paddle
energy98/210, reward Brier29/210, ending Brier127/210 and alive-only brick
Brier26/209. Related cells are not independent significance tests. Full raw
per-action reports include all probabilities, rewards, brick cells and both
histories. Aggregate gains must not hide these regressions.

## Execution, cost and footprint

112 affected Rust CPU checks (83 world checks rerun after mask repair, 29 config
checks reused), 88 packaged Python checks and strict core/binding Clippy pass.
First successful native composition preflight:484 input pins,30.953s,204 exact
inherited tensors,18 cached initial/18 trained restore arrays,two finite14-report
updates,all32 active weights changed,36 visual weights exact and204 frozen
inference tensors. Target repeatability after updates/restore, prefix causality,
lane permutation, GPU anchor equality, exact observed visual arrays and old-native
v6 rejection with its GPU loader disabled pass. Existing primitive evidence is
reused, not rerun. The extra filter is not a second learner.

Learning491 input/123 CPU-audit pins pass:4096 finite14-report rows,all32 active
weights changed,36 visual weights exact,72 visual optimizer moments changed as
allowed,204 initial/frozen tensors,full source plan/exposures and18 final restore
arrays. All saved teacher SHA/GPU readbacks remain exact, including initial,
intermediate10240 and final12288. Checkpoints embed v6 scope/start/objective and
the original anchor; ordinary restore validates teacher SHA before GPU loading
and accepts no phase override. Four successful guards/36 sealed files pass with
no host recovery,NVML or unfinished child. The separately declared920-pin frozen
assessment takes135.799s; CPU branch comparison36.831s and learning audit6.407s.

| Timing scope | Inherited H0 | New phase | Sum |
|---|---:|---:|---:|
| Native learning calls |529.515s|388.861s|918.376s|
| Train/evaluation loop |796.900s|459.954s|1256.854s|

The phase loop excludes construction,dense-initial evaluation and input pinning;
observed worker/polling span is482.530s,restore span14.317s. Frozen assessment,
qualification,failed preflights,CPU analyses,builds and packaging are outside those
learning timers and separately retained. This is a warm phase,not uninterrupted
learning,an independent seed or a matched fresh-budget comparison. Small gains
at added cost do not establish improved learning per GPU-hour.

Student68 logical/8,852,675 scalar weights/204 tensors is not the total system
footprint. The independent filter adds39 logical/6,203,136 scalar parameter
copies (24,812,544 bytes). Its reported session allocation is35,799,168 bytes,
excluding driver/staging and not peak memory or an exact total-system increment;
ordinary H0 already had observation activations. The embedded106,261,452-byte
anchor includes all original weights/moments,not GPU optimizer state for the
filter. Native teacher forward passes/host transfers are included in phase time.
The834,940,974-byte disposable build cache is released; wheels,runtimes,CPU test
executable,source archives and checkpoints remain retained.

## Preserved failures and limits

Original root `/mnt/data/kindle-breakout-world-fixed-teacher-20261107`:
f19e3fec/nativea74334ab qualifier fails construction before any update because
full checkpoint logical layout does not match a subset inference plan. Original
CPU gradient fixture mistakes a scalar zero sentinel for a gradient and uses a
wrong prior prefix; corrected autodiff checks pass. Original Clippy nested-if,
preparation import and v4/v5 earlier-header rejection fixtures remain preserved.

Loader root `/mnt/data/kindle-breakout-world-fixed-teacher-loader-20261107`:
first repair fails Rust borrow checking; explicit scoped closure arguments repair
it. e651e95e reaches the first update after204 inherited tensors/18 initial arrays
and prefix/lane checks,then fails its report assertion. Individual loss values
were not saved. CPU reproduces16 duplicate named mask inputs; native set_input
writes only the first match. Distinct masks/unique-input assertion repair the
composition. No advanced checkpoint or full learner from either failed qualifier
is reused,and neither failed qualifier is relabeled as passed.

The new declaration initially rejects intentionally caught CPU assertion text;
matching the successful83-pass summary repairs it without running a native job.
An ad-hoc system-Python NumPy import failure is corrected with the project venv.
All original logs/source receipts and separate repairs remain retained. No failed
learner,assessment or collector repeats.

`comparison.png`,`learning-curves.png` and three full causal APNGs (freeflight/
NOOP,contact/LEFT,life_loss/NOOP) come only from cached arrays. They fix lane0 and
draw0,not a best match; display resizing changes no model input or metric. Frames
are decoded and checked losslessly. The isolated offline plotting environment is
recorded. No new queries or emulator actions are used for visualization.

Accurate all-horizon physics,event timing/calibration,persistent removed-brick
memory,late-game/tunnel/second-wall coverage,untouched families and fresh
confirmation remain required. This oracle has no above-top-brick endpoints.
Do not promote this phase on latent loss,reconstruction,selected examples or
aggregate energy. Keep coverage; the world-model goal remains active and unproven.
