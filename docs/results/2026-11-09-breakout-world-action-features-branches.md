# Frozen action-feature world: failed broader assessment

The final action-feature world fails the original distribution/event gates.
Primary balanced survival-weighted ball energy is **0.22488128106733637** versus
retained1M one-hot0.17236570461161804,H0 reference0.16950221908977256 andrequired
<=0.12712666431732942. Lower isbetter. The completed learner still passes its
three intermediate dense gates; that does not establish a useful full world.

[Assessment protocol](../experiments/2026-11-09-breakout-world-action-features-branches.md),
[completed learning](2026-11-09-breakout-world-action-features-learning.md).

## What improved and what failed

Supported ball-position pairs under chosen actions improve while the joint
score worsens. These pair counts omit predicted survival and cannot replace
the primary metric. They also differ from recorded-action dense samples.

|Model|8 actions|16 actions|32 actions|
|---|---:|---:|---:|
|H0|22.453%|20.223%|5.896%|
|Retained1M one-hot|47.695%|62.478%|2.642%|
|Action features|51.303%|71.870%|17.507%|

Ending Brier at16/32 is **0.022597853809/
0.340063950908**;original limits are.01/.1.
RIGHT16 mean ending probability is0.144352903914
against an oracle zero in this cohort. There are **296/304** nonending
origin-plan alarms above5%,versus265/304 for the retained one-hot model;
maximum risk is0.577013420502. Category means and
individual alarms are distinct. These are selected conditional states,not
natural-frequency event calibration or independent trials.

Serve NOOP eight-step detection is0;all four freeflight plans detect64/64 of64
at eight steps. Those presence tests pass but do not establish precise motion.

|Original intermediate requirement|Result|
|---|---|
|observed_ball|Pass|
|one_step_ball|Pass|
|eight_step_ball|Pass|
|serve_noop|Pass|
|ending_brier|Fail|
|ending16_brier|Fail|
|ending16_right|Fail|
|nonending_false_alarms|Fail|
|mean_ball_energy_reduction|Fail|
|freeflight_each_plan_presence_at8|Pass|

All per-action/horizon regressions and every individual alarm remain for both
histories. Primary-history regressions versus H0:

|Metric|Regressed/scored cells|
|---|---:|
|ball_energy|86/210|
|paddle_energy|200/210|
|reward_brier|205/210|
|ending_brier|196/210|
|brick_brier|19/209|

## Execution, scope and next diagnostic

Only the new final16,384 checkpoint receives84 logical groups:seven reused
categories,42 plans,history16 primary/32 secondary,eight draws and1/4/8/16/32
endpoints against the unchanged cached oracle. There are33 distinct retained
contexts,no above-top-brick endpoint,andno untouched-family orfresh-seed test.
The runner's historical raw split label does not make these families fresh.

Native evaluation completes in175.330s,with zero training
updates,new emulator actions orframes. All240 source/before/after arrays stay
bitwise exact;metadata persists v25 andthe table. The single native job and
observer/scorer finish successfully,with871 execution pins,235 CPU audit pins,
239 closure pins,one host-only guard/nine sealed files,210 oracle-field joins
and168 feature checks. No native orCPU stage fails orrepeats. Original failures
in the earlier semantics,Clippy andlearning declaration remain separately.

Root `/x/Temp/kindle-world-action-features-branches-20261109` isclosed.
Scientific source71a83bce;exact native1331b702/SHA15ea18b5 andmatching
runtime-v2/runner. Invocation`af96f4e020a54a30a2bb63aeb112ba6c`,observer
`385eccd608544bf084764e10632447f6` areinactive/successful.

The hypothesis did not pass the broader retention criteria. Keep the action
feature option default-off;retain the checkpoint andits improved conditional
point metrics as development evidence. Do not extend orrepeat this learner.

The gap between spatial point metrics andthe joint score now justifies a
separately declared [CPU-only saved-survival decomposition](../experiments/2026-11-09-breakout-world-action-features-survival.md).
It substitutes evaluation weights whileholding images fixed;it does not repair
a learned head orisolate a causal training defect. No learner isselected.
Fullphysics,events,brickmemory,late-game/above-top,untouched families andfresh
confirmation remain required. The full goal isactive andunproven.

The subsequent [CPU diagnostic](2026-11-09-breakout-world-action-features-survival.md)
is complete. Ordinary ball energy .224881 becomes .185873 under oracle-assisted
survival, with `life_loss` the largest residual category. The subsequent
[edge-rebound audit](2026-11-09-breakout-world-edge-rebound.md) finds surviving
NOOP/LEFT/reversal paths there, correcting the earlier respawn attribution. It does not alter
any original score,gate orretention result.
