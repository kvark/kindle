# Decoded-RGB event heads: fixed learning comparison closed

The fixed fresh learner closes **3/3 dense gates**, with mixed point results and **12.13% more native learning time** than the retained early physical-feature control. It is not adopted from these results. The full world-model goal remains unproven.

| Model | Observed | 1 action | 4 actions | 8 actions | 16 actions | 32 actions |
|---|---:|---:|---:|---:|---:|---:|
|Actual initial|0/112|0/109|1/109|0/107|0/114|1/95|
|Retained one-hot|103/112|102/109|69/109|53/107|22/114|8/95|
|Early physical features|106/112|101/109|66/109|53/107|25/114|11/95|
|Late command|99/112|99/109|73/109|56/107|28/114|5/95|
|Posterior history|108/112|99/109|45/109|19/107|5/114|0/95|
|Retained wider motion|107/112|105/109|77/109|58/107|31/114|11/95|
|RGB event heads|103/112|99/109|71/109|52/107|30/114|8/95|

Intermediate development gates: 3/3. These are observed >=.90, one-step >=89/109 and eight-step >=28/107. They do not establish mastery.

Native calls 2064.391s; loop 2233.998s. Retained early-feature control: 1841.013s / 1998.783s; one-hot: 2203.088s / 2369.265s; wider motion: 3888.344s / 4065.276s.

Fresh 16,384 updates, RGB64/1M/B8/T32/H16/full BPTT/Auto/41001; qualified decoded-RGB event readout with local width16 and fixed LEFT/RIGHT/FIRE table, RGB100, KL1/.1/freeNat1, mixing.01, event losses1, rates4e-4. Same original two-cycle plan; no new game actions or frames. Development origins and seed are reused. Source/runtime differs from the historical learner; 32 cached default-off trained forecasts match, without proof of all old/new trained trajectories being identical. Two head input shapes and initialization differ; first-update checks and age1 saving are inside the call timer. RGB3/encoder/RSSM geometry stays; event heads widen. These are selected development comparisons, not held-out confirmation.

All 16,384 finite 13-report rows, first qualified report, total/mask identities, plan/counts, 240 initial/frozen tensors, 80 changed parameters, 18 restore arrays, 384 dense initial origins and 58 finite forecast exports pass. Detailed per-horizon, paddle/brick/event regressions remain in learning-review.json and raw exports.

The completed learning declaration allocates no automatic assessment or successor. A separately justified frozen assessment is selected below. Full distributional physics/action responses, natural event calibration, persistent removed-brick memory, late/tunnel/second-wall/above-top coverage, untouched families and fresh confirmation remain required.

Artifacts: `/x/Temp/kindle-world-rgb-event-heads-learning-20261110`.

## What improved and regressed

Relative to early physical features, development observed/one/eight/thirty-two
ball matches fall106→103,101→99,53→52,11→8 on unchanged supports. Four/sixteen
improve66→71 and25→30. Fitting observed/one/four/eight/sixteen/thirty-two is
147/163,155/177,111/180,86/176,51/171,21/179 versus147/163,159/177,114/180,
90/176,43/171,24/179. No consistent ball-tracking or efficiency advantage is established.

Development paddle one/eight/thirty-two matches improve71/128→78/128,
45/126→50/126,30/118→33/118; observed118/128→117/128 regresses.
False ball detections at1/8/32 change0/9/9→1/9/6. Brick balanced error at32
changes.060017→.060814; this is not persistent-memory proof.

Events are mixed. At8, positive-reward Brier improves.028400→.020979 (4 positives
among126 targets). At16, pointwise terminal Brier improves.001012→.00005164,
but32 worsens.001345→.003007; these slices have zero positive terminal labels
on124/118 targets. They do not establish ending recognition, natural calibration
or cumulative survival accuracy. All class-conditional scores and other regressions
remain in the pinned raw JSON and review.

Actual costs are **2,064.391s native calls/2,233.998s loop**, versus early
1,841.013s/1,998.783s:1.121334x/1.117679x. The new run costs less than the wider
model's3,888.344s/4,065.276s, but this is not a useful-learning/hour gain. Timers
include synchronous first-update checks/age1 saving, and are not GPU kernel times.
Source/native/initialization/run conditions also differ; no isolated mechanism
or broad seed reliability is claimed.

## Closed execution and raw verification

Scientific source`aa8859bde6a1b8d1d314554589b1e9aaac001516`; native
`45633605d587df73ec4c824f75b2552934d7ac32`, SHA256
`f71e9354a7ba420a263ab790e297385df608a76d34cc59174a7be07f4c2a7b8d`,
with the matching packaged runtime/runner. No rebuild or overlay.

All240 execution/172 CPU-audit/46 CPU-plan/180 closure pins pass. The independent
terminal review verifies485 distinct files across eight execution/analysis/
closure maps, including the native qualifier. It checks240 actual-initial,
240 first-update and240 final/frozen tensors by dtype/shape/bytes, plus all18
restored forecast/source arrays by bytes. Raw dense gates agree with summaries.
All80 logical parameters change across training. The exact two-cycle plan,
16,384 finite13-report rows,counts/supports,384 dense initial origins and58 finite
exports pass. No replay/live-belief/RNG checkpoint continuation is used.

Two host guards and18 sealed files pass. All four services are inactive,
MainPID0/Resultsuccess; their recorded invocation/PID completion journals verify:

- Learner/restore:c1449b7c333a4f598f9992e266c34431/PID2258744.
- Original observer:b418b44ffbf84d29b1042d2d2c6466b1/PID2258750.
- Original CPU audit:6916431b180c4a99b2a85448e324692d/PID2258755.
- Final images:3360692f09af4e8cba1114b49413cf04/PID2261360.

The final image review verifies34 inputs and15 closed files,60 physical joins,
all six curve groups/four draws andfour fixed row0/draw0 grids in2.724s. All five
final figures, including the dense comparison/loss plot, are viewed. Short images
learn, but long paths still drift, split into diffuse streaks or lose a coherent
ball. The fixed examples do not isolate a causal component or represent the
aggregate. No native learner, restore, observer, final audit or final-image job
fails or repeats. Earlier preparation failures remain separately below.

## Provisional evidence retained

The2,048-update review was provisional and did not change stopping/selection.
Its34 inputs/17 closed files,six groups/four draws/four fixed grids remain under
`milestone-2048-v1`. Development natural observed/1/4/8/16/32 then was
22/26,17/27,3/26,1/27,0/27,0/25 versus23/26,21/27,3/26,0/27,0/27,0/25.
Learn calls256.800s versus228.781s,1.122469x. These smaller curve subsets are not
the final dense gates. Original preparationv1 fails AST validation on an unmatched
parenthesis; its source/plan/failure receipt remain. Separately namedv2 scripts/
plan pass and execute once without model/GPU/emulator queries. No failed check
is relabeled as passed. The active report is preserved in Git and
`published-active-v1.md`; original native fixture/build failures stay in the
[qualification report](2026-11-10-breakout-world-rgb-event-heads.md).

## Next bounded question

After reviewing the closed result and all five figures, select the
[frozen chosen-action assessment](../experiments/2026-11-10-breakout-world-rgb-event-heads-branches.md)
for separate preparation/declaration. Its purpose is to measure the intended
joint image/event calibration benefit that point matches cannot establish.
The original primary energy/ending/false-alarm gates stay unchanged. No further
training, adoption, extension, collector or automatic successor is selected.
Full physics/events,brick memory,late/above-top coverage,untouched families and
fresh confirmation remain. Actor/critic,CDP and five-game work stay parked.
