# Intermediate Pong learns to win with a larger update budget

The4M/RGB48/R32 recipe wins most frozen matches on two fresh learners: seed1009
scores **11.0 with23/24 wins in48m53s**; confirmation4013 scores **7.333 with22/24
wins in50m14s**. Both actual-initial controls lose24/24, and all audits pass.
Both miss the unchanged mean-score target of15. This confirms useful compact
learning on two selected seeds, not broad reliability or a hardware speedup.

## Short R16 screen

The fixed200k-action screen improves the frozen score but fails the Pong gate.
Fresh seed1009, centered CDP/action-effects1, Size4M/RGB48/Auto,
N8/B8/T16/H15/R16, full BPTT/microbatch8, replay100k, full18/sticky.25/repeat4.
Native max-pooled frames receive one GPU resize; no action aid. Actual trainable
count3,578,225. All346 initial tensors match the bounded smoke.

| Policy | Selected natural matches | Mean | Wins | Gate |
|---|---:|---:|---:|---|
| Fixed200k final |24|−17.125|0|fail|
| Actual initial |24|−20.375|0|fail|

Training completes200000 aggregate actions,799850 emulator frames and24970
updates in621.700s (10m22s),321.70 actions/s, first update at248 and zero debt.
There are123 completed training matches; last50 mean−17.56. The selected frozen
gain is3.25 points, with no wins. It does not meet the mean≥15/90%-wins gate.

Final frozen collection uses68880 actions/275483 emulator frames in25.69s and
collects25 natural matches. Selection remains the first three per lane; the one
excess match and all tails remain in the full replay/video. Actual initial uses
24200 actions/96771 frames in9.05s and collects24 matches. Both have zero updates,
no truncations and exact equality of all346 frozen tensors.

All24970 reports, counters/config/source checks, finite checkpoints, full CPU
action replays/videos and host guards pass. Service
`kindle-cheap-pong-200k-20261025` exits successfully. Artifacts and independently
reconciled action/frame/time curve:
`/x/Temp/kindle-cheap-20261025/pong-4m-48-200k`.
Review SHA256`1382642300c86ff51d61924997c6db618f685523fa062f422c77b5bb2461e648`.
Native`bfb8c31`/SHA`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`,
runner`a99416e`, RX7900XT/RADV.

This is one reused development seed with changed capacity, resolution, replay
ratio and runtime relative to historical controls. A small frozen gain is useful
evidence, not reliable winning or a matched-resolution ablation. No unchanged
extension follows. The later fresh recipe test is reported below.

## R32 and500k actions

The [declared test](../experiments/2026-10-26-cdp-cheap-pong-update-budget.md) keeps
the model, image geometry, native binary, seed1009 and all346 actual initial
tensors. Replay ratio16→32 and experience budget200k→500k change together.
Centered CDP/action-effects1, Auto,3,578,225 parameters,N8/B8/T16/H15, full
BPTT/microbatch8, replay100k, full18/sticky.25/repeat4 and one GPU resize remain.
There is no action aid; actor_unimix stays0.

Training completes **500,000 aggregate actions /1,999,756 emulator frames /
124,939 updates in2,933.450s (48m53s)**,170.45 actions/s. First update248, zero
debt,157 natural training matches, final online last50 mean8.66. The predeclared
55-minute cap is met. The100k/200k/300k/400k recovery checkpoints each pass finite,
configuration and counter checks before replacement; evaluation uses fixed500k
final weights. No selected checkpoint, cap increase or extension occurs.

| Fixed policy | Selected natural matches | Mean | Wins / losses | Gate |
|---|---:|---:|---:|---|
|500k final|24|11.000|23 /1|fail: mean below15|
|Actual initial|24|−20.375|0 /24|fail|

Frozen seed4000001009, first three natural sampled matches per lane. Final
collection uses119,416 actions/477,607 frames in44.441s, collecting26 natural
matches; the two excess matches and unfinished tails remain in the full replay.
Actual initial uses24,200 actions/96,771 frames in9.078s and collects24 matches.
Both arms have zero updates or truncations, and all346 tensors remain unchanged.

All124,939 numeric reports, complete source/configuration/counters, initial/final
checkpoints, full CPU replays/videos and three host guards pass. Service
`kindle-cheap-pong-48-r32-500k-20261026` exits successfully. Independent
action/frame/time curves and the R16/R32 comparison by actions, updates and wall
time are retained. The comparison does not extrapolate the short control or
treat the differently budgeted frozen endpoints as a replay-ratio-only effect.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-500k-r32`, including
`learning-review.json`, `curve/`, `budget-comparison/` and actual initial/final
weights. Review SHA256
`8aa8857a90d050bda830b475a2e8b392bb3c06bfce04e35687bd8f953f68c506`.
Source at launch`4918f78`, native`bfb8c31`, runner`bcef410`, same native SHA above.

This is a fresh life on reused development data. The recipe learns to win most
matches but does not meet the complete score gate. Historical successful500k
learners took about37 minutes on different GPU/precision/model settings; smaller
parameter count has not established a speedup here. Preserve the useful result
without extending it unchanged. The focused Freeway seed confirmation follows
after review; Pong's remaining score gap stays open.

## Fixed-recipe seed4013 confirmation

The [focused confirmation](../experiments/2026-10-26-cdp-compact-pong-confirmation.md)
keeps the native recipe and budget fixed. Only learner/environment seed4013 and
evaluation seed4000004013 change. CLI`1ed07af` adds an optional imagination-length
argument relative to the primary runner; both retain H15. Full native config
matches the primary except seed. The new initial weights differ as expected;
retain this life's own actual-initial control.

Training completes **500,000 aggregate actions /1,999,806 emulator frames /
124,939 updates in3,014.113s (50m14s)**,165.89 actions/s, first update248 and zero
debt. The55-minute cap is unchanged. There are138 natural training matches;
final online last50 mean3.20. All four100k recovery checkpoints pass finite,
configuration and counter checks before fixed500k replaces them.

| Fixed policy | Selected natural matches | Mean | Wins / losses | Gate |
|---|---:|---:|---:|---|
|500k final, seed4013|24|7.333|22 /2|fail: mean below15|
|Own actual initial|24|−20.333|0 /24|fail|

The final policy wins91.67% of selected matches but misses the score gate.
Frozen final uses121,080 actions/484,284 emulator frames in51.321s; initial
uses26,416/105,641 in11.324s. Each collects exactly24 natural matches, the first
three per lane; partial tails remain in the full replay/video. There are zero
updates or truncations, and all346 tensors remain unchanged in each arm.

All124,939 numeric reports, full source/config/counter checks, finite checkpoints,
full CPU action replays/videos and three host guards pass. Service
`kindle-cheap-pong-48-r32-500k-4013-20261026` exits successfully. Independently
reconciled curves show slower learning on4013 than1009; the comparison preserves
separate frozen policies and their own initial controls. No earlier checkpoint
selection, cap increase, resume or extension occurs.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-500k-r32-seed4013`, including
`learning-review.json`, `curve/`, `seed-comparison/` and `postprocess-review.json`.
Review SHA256`0ed92b0c8c06fee2eab15accf6d8b085f38d8960b3d42120ed520a968297f2c0`.
Curve SHA256`851db14046df0d39af9d62f531703ab0be1d3c7e0d0c4d323ea190a7a13f398c`;
comparison SHA256`bde887e6bde03a794b7eab75497e084766701c29482b1895354549eb1bacb52e`.
Source at launch`923b47b`, native`bfb8c31`, runner`1ed07af`, native SHA above.

Seed4013 was selected after use on Boxing/Freeway and is not held out. Two
learners and48 evaluation matches do not establish broad training-seed
reliability. Both compact policies are useful, both mean15 gates remain unmet,
and neither is extended unchanged. The next bounded test addresses Qbert's
remaining task failure using the separately declared six-action vocabulary.

## Exploration-bonus scale in the completed learners

A CPU-only analysis checks all 124,939 sequential reports from each completed
500k run and verifies the metric definitions against native source `bfb8c31`.
The reported imagined intrinsic reward is already scaled. Subtracting it from
the combined imagined reward gives the predicted extrinsic **net** mean.

| Seed | Last 25k updates: scaled bonus | Predicted extrinsic net mean | Ratio of these means |
|---|---:|---:|---:|
|1009|0.0105571|0.00280554|3.763|
|4013|0.0103200|0.00298180|3.461|

These are window averages of minibatch means. Pong's positive and negative
rewards can cancel: the ratios do not compare absolute reward magnitudes and
do not establish that the bonus caused the failed mean-score gates. Both
learners already win most matches. Earlier windows have negative predicted
extrinsic net means, so no positive bonus/net ratio is reported for them.

Holding the recorded predictions fixed, coefficient .01 instead of 1 would
reduce the late bonus/net ratios to .03763 and .03461. Actual training changes
states, ensemble predictions and critic targets; this calculation predicts no
score gain. A smaller game-specific bonus is a candidate for a later bounded
test with the model and experience budget fixed. No GPU allocation, new policy
decisions, learner updates or change to the active Breakout run occurs here.

Artifacts: `intrinsic-scale-review.json` in each run directory above, produced
by `/x/Temp/kindle-cheap-20261025/review_pong_intrinsic_scale.py`.
Review SHA256, seed1009:
`4fbc3c0821a4e173d800533ffb30a64c7cb2235d0c369c66c5399d7fa4180e6d`;
seed4013: `f1c1639879017667066582217bc362dfd893f32cf9a04e64ce3d9ce8c47c1c3a`.
The analysis takes 1.35s/1.66s CPU wall time and preserves source/log hashes,
early/middle/late distributions and the offline rescaling calculation.

## Smaller bonus improves the fixed-budget result

Coefficient .01 reaches **14.9583 with24/24 frozen wins**, versus the coefficient1
control's11.0 and23/24. Its own actual initial scores−20.375 with0/24 wins.
This is promising development evidence, but **14.9583 remains below the unchanged
mean15 gate**. Do not round it into a pass. One focused seed4013 confirmation is
selected; no extension or favorable-checkpoint selection follows.

Fresh1009/eval4000001009,4M/RGB48/Auto/N8/B8/T16/H15/R32,3,578,225 parameters,
full18/sticky.25/repeat4, actor_unimix0 and no external aid remain fixed. Only
native intrinsic_reward_scale changes1→.01; the ensemble stays enabled. All346
initial tensors and the complete initial evaluation's semantic action/outcome
trace match the primary control. The CLI adds an optional imagination-length
argument relative to that control; both configurations retainH15.

| Coefficient, seed1009 | Actions / emulator frames / updates | Training wall | Frozen final / initial | Wins |
|---|---:|---:|---:|---:|
|1|500k /1,999,756 /124,939|48m53s|11.000 /−20.375|23/24|
|.01|500k /1,999,728 /124,939|49m55s|14.958 /−20.375|24/24|

Recorded training time is2,995.273591s,166.93 aggregate actions/s, within the
unchanged55-minute cap. First update248, zero debt,202 natural training matches,
final online last50 mean9.50. Online curves are close across much of training;
the improved frozen result does not establish faster execution or seed reliability.

Final/initial frozen evaluations use71,840/24,200 actions,287,302/96,771 emulator
frames and30.745/10.271s. Each collects exactly24 natural matches, the first three
per lane; partial tails remain, with zero updates/truncations. All346 tensors
remain unchanged per arm. All124,939 strict reports, four recovery checkpoints,
full configuration/source/counters, finite checkpoints, CPU replays/videos and
three host guards pass. The service exits at07:27:49 UTC; its journal is retained.

In the last25k updates, observed scaled bonus is0.000102626 and inferred extrinsic
net prediction0.00436039, a ratio0.02354 versus the control's3.763. These are
different learned trajectories, not the earlier fixed-prediction rescaling.
Signed cancellation still limits the ratio; it does not prove why the score rose.

The comparison plot initially failed because the isolated CPU plotting environment
lacked safetensors. Installing version0.8.0 there completes the unchanged script
in3.22s; the failed attempt and repair receipt remain. No GPU work or evaluation
is repeated. The comparison curves are reconciled and visually reviewed.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-r32-intrinsic001-500k`, including
`learning-review.json`, `curve/`, `coefficient-comparison/`, `terminal-journal.txt`
and `comparison-repair-review.json`. Review SHA256
`c9160f58dc2c25b751a3b332ca85838b3c8ad3257b69bfb7df0215ab88b1c285`;
curve`3525882ff33978150c8dfc260f5944d703757864c93965d3983a07d721e178ca`;
comparison`94c2ba7bc4fee016512b8c7371af3db1d9b151ca4b48fcdda6213f1a8dc20606`.
Source at launch`12a6553`, native`bfb8c31`, runner`1ed07af`, replay`e853818`.
The [fixed-recipe confirmation](../experiments/2026-10-26-cdp-pong-small-intrinsic-confirmation.md)
uses previously evaluated4013, not a held-out seed. The gate remains unchanged.

## Smaller-bonus seed4013 confirmation

The fixed-recipe confirmation scores **14.0 with 24/24 frozen wins**, versus
7.333 and 22/24 for its coefficient1 control. Its own initial scores−20.333
with0/24 wins. Both smaller-bonus learners improve over their respective controls,
but **both still fail the unchanged mean15 gate**. Retain this useful recipe and
move the next allocation to the four-action Breakout test; no third Pong seed or
unchanged extension is selected.

Only learner/evaluation seeds change from the coefficient.01 primary to
4013/4000004013. Native configuration differs from the same-seed coefficient1
control only in intrinsic_reward_scale; all346 initial tensors and the complete
initial semantic evaluation trace match that control. Keep4M/RGB48/Auto,
3,578,225 parameters, N8/B8/T16/H15/R32, full18, actor_unimix0 and no aid.

| Coefficient, seed4013 | Actions / emulator frames / updates | Training wall | Frozen final / initial | Wins |
|---|---:|---:|---:|---:|
|1|500k /1,999,806 /124,939|50m14s|7.333 /−20.333|22/24|
|.01|500k /1,999,727 /124,939|49m58s|14.000 /−20.333|24/24|

Training takes2,998.127072s, with150 natural matches, first update248 and zero
debt. The online last50 mean14.82 is separate from the frozen14.0. The reviewed
curves show an earlier rise than the same-seed coefficient1 control, while
execution cost remains similar. The two coefficient.01 policies score14.9583
and14.0 separately; these are two reused development seeds, not held-out tests
or48 independent learners.

Final/initial frozen evaluations use96,072/26,416 actions,384,239/105,641 emulator
frames and40.330/11.195s. Each collects24 natural matches, the first three per
lane, with partial tails retained and zero updates/truncations. All346 frozen
tensors,124,939 strict reports, four recovery checkpoints, full configuration,
source/counters, finite checkpoints, complete CPU replays/videos and three host
guards pass. The original service exits successfully at08:27:53 UTC, after
51m52s including evaluations/audit; its journal is retained. Both comparison
scripts complete without repair and their plots are visually reviewed.

In the last25k updates, observed scaled bonus is0.0000951322 and inferred
extrinsic net prediction0.00605091, ratio0.01572 versus3.461 for coefficient1.
Different learned trajectories and signed cancellation limit this comparison;
it does not establish the mechanism behind the score gain.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-r32-intrinsic001-500k-seed4013`,
including `learning-review.json`, `curve/`, `coefficient-comparison/`,
`seed-comparison/`, `comparisons-review.json` and `terminal-journal.txt`.
Review SHA256`b82d9054f0edf366fbd6256570ad567612a0a88bbdcd333265de8b31ceecb432`;
curve`8318ea9ab0373f61cc5c584d8d794855877a531f84ce409e843e838d7b4e760e`;
coefficient comparison`c7e6a85f189111c20df5d48772cdcfeb4f136646c9b92f3ebe104d2291e80dd7`;
seed comparison`a1895ad57fc6a9e595e1cecba63f6cfdbbda67764af3da41ed4b19df5d33bc39`.
Source at launch`1ff41da`, native`bfb8c31`, runner`1ed07af`, replay`e853818`.

## Larger replay batch: source feasibility only

A source/CPU review identifies B16/T16 as a possible later cost experiment,
keeping the coefficient.01 recipe, N8 and ratio32. Centered CDP requires full
batch BPTT, so world_microbatch_size must also change8→16. This is not a selected
allocation, GPU benchmark or learning result; the Qbert500k/R16 test remains
active. A full Pong run is not justified by fewer optimizer steps alone.

Replay warmup requires batch_size×batch_length eligible sequences. With eight
fresh lanes and context1, the expected first update moves248→376; verify the
actual startup before using those counts. At500k actions, scheduled updates
move124,939→62,454. Sampled positions are15,992,192 versus15,988,224, nearly equal
apart from the different warmup. The count calculation matches both completed
B8/T16 controls at ratios32 and16; B16 execution remains unmeasured.

The completed coefficient.01 primary spends2,801.583s learning and193.690s in
other stages, averaging22.424ms per update. Holding that other cost fixed, a
40-minute500k run would require the larger update to cost at most1.575× the
current update; a30-minute run requires at most1.147×. These are requirements,
not predictions: doubling batch size increases work per update. Optimizer
cadence, centering population and learning-rate warmup per action also change,
so sampled-position accounting does not establish equivalent learning.

Any later B16 allocation needs one bounded initialization/learning/frozen-restore
check before a full trial. No native code change, agent construction, GPU work,
new policy decisions or queue is performed by this review. Source hashes and
completed control receipts are retained in
`/x/Temp/kindle-cheap-20261025/pong-batch-width-feasibility-review.json`, SHA256
`917fa8d8e999358830264b06ed1a376d786f76c1e66de183f2d6eb8a36ff944a`.

The [bounded B16 check](../experiments/2026-10-26-cdp-pong-batch16-check.md) is now
prepared for16,384 training actions and1,024 frozen restore actions. Its full
configuration/counter/tensor audits and unselected guard are prepared; no GPU
work has started and Qbert remains the active allocation. This is not a full
Pong learning trial or an automatic queue.

## B16 bounded check passes with lower early wall cost

The existing-native B16 check completes16,384 actions/65,522 emulator frames and
2,002 updates in **72.224s**, versus94.054s for the matching B8 action window:
**23.21% less training wall time**. Construction is separate,2.299s versus2.140s.
B8 performs4,035 updates and516,480 sampled positions; B16 performs2,002 and
512,512. Warmup, cadence, centering population and trajectories differ. This
short timing result is not full-budget throughput or frozen learning evidence.

Fresh1009, coefficient.01/4M/RGB48/Auto/N8/B16/T16/H15/R32,3,578,225 parameters,
full BPTT/microbatch16, full18 and no aid. Full config differs only in batch_size
and world_microbatch_size8→16. First updates376:1,384:2,392:3 match the source
calculation. All346 initial tensors match the B8 control, all2,002 numeric
reports and exact counters pass, and CNN0's1,200 weights plus the first reward
projection's163,840 weights all change. That excludes unchanged selected weights,
not incorrect gradient scale or poor features.

Ordinary restore executes1,024 frozen actions/4,096 frames in0.492s, zero updates
and zero natural episodes; all346 tensors remain unchanged. This checks restore,
not natural-match competence. Both guards pass; service exits09:38:03 UTC after
1m23.065s including construction/restore/audit. No retry, cutoff or native change.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-b16-shape-check`, including
`review.json`, `startup-review.json`, selected declaration and terminal journal.
Review SHA256`41adf95403303df1d12d86d1fcf260c62081e1ef1c07ea4d637c1aa82a5fca8e`;
source at launch`255bf94`, native`bfb8c31`, runner`1ed07af`.

After reviewing this result, select one [fresh500k B16 learning test](../experiments/2026-10-26-cdp-pong-batch16-budget.md)
with a45-minute cap and the original mean15/90%-win gate. It must establish
actual frozen quality and full-budget cost; no result follows from this check
alone. The validation life is retained separately and is not the performance
trial or an independent training seed.

## B16 at the fixed500k budget

B16 completes **500,000 actions /1,999,733 emulator frames /62,454 updates in
2,325.367s (38m45s)**, using **22.37% less training wall time** than B8's49m55s
at the same experience budget. Action throughput is215.02/s,1.288× B8. Frozen
quality is lower: **11.875 versus14.958**, both24/24 wins. Both miss the unchanged
mean15/90%-win gate. This is a useful measured cost tradeoff, not a quality pass
or evidence that batch16 is an interchangeable replacement.

| Same reused seed1009, coefficient.01 | B8 | B16 |
| --- | ---: | ---: |
| Aggregate actions |500,000 |500,000 |
| Emulator frames |1,999,728 |1,999,733 |
| Updates / first update action |124,939 /248 |62,454 /376 |
| Sampled replay positions |15,992,192 |15,988,224 |
| Training wall |2,995.274s |2,325.367s |
| Construction, separate |2.140s |2.203s |
| Fixed frozen final / actual initial |14.958 /−20.375 |11.875 /−20.375 |
| Selected wins, final / initial |24/24 /0/24 |24/24 /0/24 |

Only native batch_size and world_microbatch_size change8→16;4M/RGB48/Auto,
N8/T16/H15/R32, full18, actor_unimix0 and no aid remain fixed. All346 actual
initial tensors and semantic frozen-initial traces match. Cadence, centering
population and optimizer warmup per action differ; no learning-rate rescaling.
The online curve develops earlier in wall time, while its fixed frozen endpoint
scores lower. Neither curve nor speed substitutes for that endpoint. The short
shape check's cost is additional, and this reused seed is not independent
confirmation.

B16 completes154 natural training episodes, mean−4.545 over all completions and
online last50 mean8.18, with zero debt. All62,454 strict numeric reports, four
100k recoveries, complete config/source/counters and finite checkpoints pass.
Final/initial frozen collection uses95,664/24,200 actions,382,597/96,771 frames
and40.406/10.252s. Select24 natural matches per arm, retaining two excess final
matches and partial tails. Zero updates/truncations, all346 unchanged frozen
tensors, complete CPU replays/videos and three host guards pass.

Curves and the B8/B16 comparison complete and are visually reviewed. Service
`kindle-cheap-pong-b16-intrinsic001-500k-20261026` exits successfully at10:28:51 UTC,
40m38.181s total service wall. Source at launch`8a966a2`, native`bfb8c31`,
runner`1ed07af` and replay`e853818`; no cutoff, retry or native change.
Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-r32-intrinsic001-500k`,
including `learning-review.json`, `curve/`, `batch-comparison/`, four recovery
reviews and terminal journal. Review SHA256
`bc6e9fbc5889066141be478cf115aa5648383f4a7e87656bbe3bdf124b0b229f`;
curve`32e73c1401181bfe3060b960934601ec25b70079e4e9c092522079a73321a63f`;
comparison`944e622a80ba30f3cb212300fd3cf0b9e46566c5675896cc759572c8e9157738`.

No extension or seed confirmation is selected for this lower-scoring recipe.
The next selected step is the separately prepared
[bounded Qbert capacity check](../experiments/2026-10-26-cdp-qbert-capacity-check.md).
Its different game/capacity still requires its own learning test; Pong's speed
result does not establish Qbert competence or justify a full allocation.
