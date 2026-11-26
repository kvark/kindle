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

## Doubling B16 learning rates lowers the frozen score

The doubled-rate policy scores **9.125 versus11.875** for the original B16 rates,
with24/24 natural wins for both. Own actual initial scores−20.375/0wins. Training
takes **38m32s**, essentially unchanged from38m45s. The rate change fails to
recover the quality lost with B16 and still misses the original mean15 gate.
Retain the original rates; no extension or seed confirmation is selected.

Fresh1009/eval4000001009,4M/RGB48/Auto/N8/B16/T16/H15/R32,3,578,225 parameters,
full BPTT/microbatch16, coefficient.01, actor_unimix0, full18 and no aid.
Only native base/encoder/dynamics rates double4e-5/6e-6/4e-4→8e-5/1.2e-5/8e-4.
Behavior inherits the base;1,000-update warmup, moments, clipping, slow critic
and normalization schedules remain fixed. Both runs keep500k actions/62,454
updates,15,988,224 sampled positions and first376. All346 actual initial tensors
and the complete semantic initial evaluation trace match the B16 control.
This is one reused development seed and a fresh life, not independent confirmation
or an equivalence between B8 and B16 optimization.

| Rate scale | Actions / emulator frames / updates | Training wall | Frozen final / initial | Natural wins |
|---|---:|---:|---:|---:|
|1|500k /1,999,733 /62,454|38m45s|11.875 /−20.375|24/24|
|2|500k /1,999,797 /62,454|38m32s|9.125 /−20.375|24/24|

Recorded training time2,311.673782s,216.29 actions/s; construction is separate,
2.111s. All133 training episodes are natural, with zero debt and final online
last50 mean2.34 versus the control's8.18. Reviewed curves show similar early
learning, then slower improvement with doubled rates after about210k actions.
Wall-time ratio.9941 is not evidence of a material speed gain.

Final/initial frozen evaluations use110,376/24,200 actions,441,457/96,771 emulator
frames and46.835/10.312s. Each collects and selects24 natural matches, with no
excess episodes; partial tails remain in complete traces. Zero updates/truncations,
all346 unchanged tensors per arm, all62,454 strict reports, four recovery audits,
full config/source/counters, complete CPU replays/videos and three guards pass.
Service exits11:59:15 UTC,40m37.303s total wall. Completed curves and comparison
pass and are visually reviewed; no checkpoint selection or cutoff occurred.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-r32-intrinsic001-lr2-500k`.
Learning review SHA256`915aae2ff28e0dcabcd55c1fa7acc3e5e3aa84dcf4ee9c05994a60d69fbff84b`;
curve`c9e192407ad3897e935438f0432f945567451123720ee0728b94c986f39b6d5f`;
comparison`e7b304e0da51be66d6a028989a015dbafca58d2e5844f0e9f424844b47c5e14b`.
Source at launch`d4d1093`, native`bfb8c31`, CLI`22bd61b`, replay`e853818`.
The fresh-rate CLI change passes278 applied vector tests; native binary and
original default rates remain unchanged. After reviewing this result, select
the prepared Qbert5% persistent-action-aid screen under its25-minute cap.

## Shorter imagination saves time but lowers the frozen score

The [fixed-budget H5 test](../experiments/2026-10-26-cdp-pong-short-imagination.md)
completes500k actions/1,999,739 emulator frames/124,939 updates in2,286.891s
(**38m07s**),23.65% less training time than H15's49m55s and1.310× action
throughput. Frozen score falls **14.958→8.667**, with24/24 wins for both.
Own actual initial remains−20.375/0wins. The original mean15 gate still fails;
retain H15, with no extension or H5 seed confirmation selected.

| Imagined rollout | Actions / updates | Training wall | Frozen final / initial | Wins |
|---|---:|---:|---:|---:|
|H15 control|500k /124,939|49m55s|14.958 /−20.375|24/24|
|H5 candidate|500k /124,939|38m07s|8.667 /−20.375|24/24|

Only native imagination_length15→5 changes. Keep centered CDP4M/RGB48/Auto,
3,578,225 parameters,N8/B8/T16/R32, microbatch8, replay100k, coefficient.01,
actor_unimix0, original rates, full18/sticky.25/repeat4/noops0 and no aid.
Fresh1009/eval4000001009 is reused development data. All346 actual-initial
tensors and the complete semantic initial evaluation trace match the H15
control. The CLI now permits optional fresh learning rates; both use defaults
and the native binary is unchanged. H5 changes imagined actor/critic learning
and bootstrapping, while replay/update cadence and15,992,192 sampled positions
stay fixed. Native max-pooled frames receive one GPU resize.

First update248, zero debt,186 natural training matches, online last50 mean11.44;
construction2.101s is separate. The reviewed online plot shows slower early
learning per action, then a better late online score and earlier progress in
wall time. That does not carry over to the fixed frozen endpoint. The45-minute
training cap is met; no cutoff, resume or selected checkpoint.

Final/initial frozen use95,752/24,200 actions,382,980/96,771 emulator frames and
40.561/10.095s. Exactly24 natural matches are collected/selected in each arm,
with partial tails retained, zero updates/truncations and346 unchanged tensors.
All124,939 numeric reports, four recovery audits, full config/source/counters,
finite checkpoints, complete CPU replays/videos and three guards pass. Curves
and comparison pass and are visually reviewed. Service exits13:18:17 UTC,
40m02.184s total wall. The conditional4013 preparation remains unselected
because the primary gate fails; no agent or GPU work is started for it.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-h5-r32-intrinsic001-500k`;
review SHA256`b4f5a05008975edbd766bdf490e35ee4c2ed957a4d6d6f541738d0757f1d8d23`;
curve`42800c5367f0c9d0fee5cde611ba2bb939fa1db6ffc72e74538cd5c436fa641b`;
comparison`54c91f2ec846d10c907cf2d3a27dafffa1cf180d562c884cfe5aaf3a49f87cfb`.
Launch source`36500e7`, native`bfb8c31`, CLI`22bd61b`, replay`e853818`.
After completed review, select the separately prepared Breakout coefficient.01
test at its fixed minimal4/200k budget; this result does not alter its recipe.

## Host restart interrupts B16/R40

The declared500k-action/78,067-update R40 experiment is **incomplete**. Complete
persisted records reach265,440 actions,1,061,594 emulator frames and41,416 finite
learner reports, last timestamp14:22:08 UTC. The last saved checkpoint is200k/
31,192 updates, matching its previously audited346 finite tensors. No run_end,
terminal training-guard result or original frozen evaluation exists. The393-byte
torn training record and3,600 trailing NUL guard-log bytes are preserved and
hashed separately. Complete prefix action/update/frame/episode accounting passes;
this does not certify unrecorded work or a clean exit.108 completed training
matches are recorded. Last complete262,144-action progress is1,487.186s
(24m47s),40,902 updates and expected.25 credit; actual terminal wall time is
unknown. The196,608-action cost comparison remains valid, without500k quality.

A new boot starts15:34 UTC on kernel7.0.0-38, replacing7.0.0-34. AMD module
identity changes to`9BDBC866F55BF296F1B7FA2`; RX7900XT and Mesa26.0.8-1ubuntu0.3
remain. The interruption cause is unmeasured. Current-boot host scan passes,
and a512-action/2,048-frame frozen restore completes in0.251s after construction,
with zero updates and all346 tensors unchanged. Its guard passes and service
exits15:48:47 UTC; no automatic driver reset, reload or reboot occurs.

A separate assessment of the saved200k policy scores **−9.708 versus own
initial−20.375**, both0/24 natural wins. This is not the missing500k endpoint or
a matched-budget quality comparison with the completed500k controls. The
original mean15/90%-wins gate fails at this saved checkpoint. No resumed
training, checkpoint selection, seed confirmation or extension follows.

Final/initial frozen evaluations use87,792/24,200 actions,351,124/96,771 emulator
frames and33.532/9.252s. Each completes exactly24 natural matches, first three
per lane, with no excess, truncations or updates; partial tails remain. All346
source/frozen tensors stay exact per arm, both guards and complete CPU replays/
videos pass, and the initial semantic trace matches the old-boot B16/R32 control.

The separate frozen service exits with a CPU-audit failure at15:51:45 UTC:
its new wrapper incorrectly imports the task-replay checker for Pong. A separate
CPU-only auditor uses the same match-replay checker as the existing Pong
controller, reuses the completed final replay and generates the initial replay.
All checks then pass. The original wrapper, failed journal and correction
receipts remain; no GPU stage repeats. The interrupted learning curve is
reconstructed only through the last complete progress record and reviewed.

Artifacts under`/x/Temp/kindle-cheap-20261025`:
`pong-4m-48-b16-r40-intrinsic001-500k/interruption-review.json`, SHA256
`9236dc07e625372fd5ffe4abd323de4e57cb24c665019349cd919c5dcf33dc79`;
`interrupted-curve/curve.json`, SHA256
`f0210ad741a4131a04ae07c20e82afd64a35f63f316cfbc182f2474001c81bc6`.
Frozen assessment:`pong-4m-48-b16-r40-interrupted-200k-frozen/review.json`, SHA256
`16302a91140207e2591f6d1e35b435ffe262d70f1ea9590864d644a3bee29f7e`;
CPU repair`8741ebec1a44b38e19fcdbc1094c751069792b26a40dc6b246da43a1726adf33`.
New-boot evidence:`newboot-recovery-20261026-a8648671`, restore review SHA256
`5a47b92a473bb8cc1aa60328534d5818e72e108f7b2322e8aa121b34d541d81c`.
Training source`647b457`, frozen wrapper source`32e6714`, native`bfb8c31`,
runner`22bd61b`, replay`e853818`. Original declarations/logs are unchanged.

## Six actions save time but lower frozen quality

The six-action run completes500,000 actions/1,999,742 emulator frames and124,939
updates in2660.589s (**44m21s**),11.17% less training time than the full18
control's49m55s. Frozen final score is **10.375 versus own initial−20.042**,
24/24 versus0/24 wins. Full18 scores14.958/24 wins at the same budget. Both miss
the unchanged mean15 gate. Retain full18; no extension or conditional4013
confirmation is selected. The prepared second-seed refusal/decision is retained.

Only action_count18→6 and published→published-minimal change: NOOP, FIRE, RIGHT,
LEFT, RIGHTFIRE, LEFTFIRE. Keep4M/RGB48/Auto/N8/B8/T16/H15/R32, microbatch8,
replay100k, coefficient.01/actor0, original rates and no aid. Native frames
receive one GPU resize. Trainable parameters change3,578,225→3,568,997 with
different initial dimensions/weights; own initial controls are required, and
no matched-weight claim is made. Fresh1009/eval4000001009 are reused development
seeds. Same RX7900XT/Mesa/native binary; kernel7.0.0-34→7.0.0-38 and AMD module
changes remain a comparison confound.

First update248,15,992,192 sampled replay positions, zero debt,143 natural
training matches and final online last50 mean9.88. Construction1.949s is separate.
The reviewed curve is slower per action in the middle of training and catches
up late; its late wall-time advantage does not carry over to frozen quality.
The55-minute training cap is met without cutoff or selected checkpoints.

Final/initial frozen checks use81,944/23,936 actions,327,735/95,723 emulator
frames and31.254/9.175s. Each collects/selects exactly24 natural matches, first
three per lane, with zero updates/truncations and no excess matches; partial
tails remain. All346 tensors stay unchanged per frozen arm. All124,939 numeric
reports, full configuration/source/counters, four recovery checkpoints, complete
CPU replays/videos and three guards pass. Curves/comparison pass and are visually
reviewed. Service exits17:28:27 UTC after45m50.755s total wall.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-r32-minimal-intrinsic001-500k`.
Learning review SHA256`e7337fd5992b55a336069cffba47407eb6958593179fc553f750fab702b89c15`;
curve`e767f56faa346f748534613f7a9304964d7e7233e7ce2e9db4c167cc132a9ea9`;
comparison`31fe2d3a9281b716e640e3dba87190e2ef85440f42b1ce576d6a50bee7bee06f`;
completion`588861c219ac101a476c83a37faa86b5810ac662490a7dcd182010874363400f`.
Launch source`5043b7a`, native`bfb8c31`, CLI`22bd61b`, replay`5043b7a`.
After this completed review, select only the separately declared bounded
Breakout actor/critic-gradient execution/restore check; no full learning life
follows automatically.


## Entropy review and unselected greedy-action assessment

A2.98-second CPU review of the last20% of completed learner updates finds mean
imagined policy entropy1.295/0.977 for the two B8 coefficient.01 seeds and1.203
for B16 seed1009. Entropy coefficient times weighted entropy divided by weighted
absolute normalized advantage is0.00849/0.00714/0.00839. These are descriptive
minibatch values, not raw gradient norms or live frozen-action probabilities.
Redundant full18 action labels also limit interpretation. This does not establish
that entropy regularization causes the missed mean15 gate; no entropy-rate
training change is selected from this evidence.

Prepare an alternative that requires no training: assess the fixed retained
B8/4M/RGB48/coefficient.01 seed1009 final policy and its own actual initial
weights using native `--greedy`. It was prepared unselected while QbertRGB64
ran. After reviewing Qbert's lower718.750/0of24 result and passing complete
audits, **select this frozen assessment**. It is now complete; the result follows.
This changes evaluation policy; preserve the original sampled14.958/−20.375
result and its failed mean15 gate. A greedy result cannot relabel that failure.
The existing native argmax path needs no code change, rebuild or qualification.

The selected assessment retains full18/sticky.25/repeat4/native
frames/one GPU48 resize, eval4000001009, N8, first3 natural matches per lane,
24 per arm,600k-action/300-second caps, zero updates/truncations, excess/tails,
all346 unchanged tensors and complete CPU replay/video/guard audits. Keep
mean15/90%-wins. The original500k/124,939-update training cost remains49m55s;
this adds only frozen assessment. One reused development seed; no checkpoint
selection, extension, automatic confirmation or claim of isolated causality
versus the older-kernel sampled control.

CPU preparation verifies both source checkpoints, exact sampled score/criteria
recomputation, frozen-only commands, source/runtime/ROM pins and refusal while
unselected. Pong's natural-score keys replace task-score keys before validation;
the earlier preparation is retained. Active Qbert scripts/pins remain unchanged.
Entropy review:`/x/Temp/kindle-cheap-20261025/pong-entropy-review.json`, SHA256
`d0bebc8c428ba1f690b2b4a9ea5d415de3bd29b1668b5623a88cea360d3cd154`.
Candidate:`/x/Temp/kindle-cheap-20261025/pong-4m-48-intrinsic001-500k-greedy-frozen`;
preparation`18f6fd69bedcfe82cca8c1a4c7bb2152e68d04db853402255f50314610502b7b`;
unselected declaration`52edc14fd1930f83e68116be965aebe9ad8cd5595d07bbc3a8e3079c9888c44b`;
controller`adf1206b9339e26b9467c65974c52a9f187a535a8f8acee58e052577a2b7b501`.


## Greedy evaluation does not improve the retained Pong policy

Native greedy evaluation scores **14.333 versus sampled14.958**, both24/24 wins
and both below the unchanged mean15 gate. Own initial greedy scores−20.833 with
0/24 wins, versus sampled−20.375. Retain sampled evaluation; no greedy seed
confirmation or training extension is selected. This is one reused learner and
a changed evaluation protocol, not a new independent training result.

The fixed retained B8/4M48/Auto/N8/T16/H15/R32/coefficient.01 final and actual
initial weights keep their full18/sticky.25/native one-resize configuration.
The assessment starts18:43:09 UTC/source`9989fa6`, owner
`kindle-cheap-pong-greedy-frozen-20261026`, with zero new training actions or
updates. Original training cost remains500k/124,939 updates/49m55s. Startup
verifies unchanged native config, retained counters and `evaluate_greedy`.

Final/initial use77,640/20,288 actions and310,513/81,145 emulator frames in
29.564/7.757s. Exactly24 natural matches per arm, no excess, all partial tails
retained, zero updates/truncations. All346 frozen tensors and original source
files remain unchanged. Full configuration/counter/finite-tensor, CPU replay/
video and both host guards pass. Service exits successfully18:44:25 UTC,
1m16.097s total wall. No retry, GPU repetition or host recovery. The sampled
control ran on the earlier kernel/module; this is not an isolated sampler effect.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-intrinsic001-500k-greedy-frozen`.
Selected declaration SHA256`0249a71b189474ea3dbb5446a8909127b3aba2dab2be29428c2b05005da7d6c0`;
startup`cdc25bf469e693cd62790679d85e32eeede753b059fb75b658fb4a329ad6c117`;
review`36b31032e9a76f6e1975e412ed6840bfe303970682c578af40c0b7838bafdf33`;
completion`d689faa46e32f87424ee2950f3134d0dc8f7acb6e5fe894608591683dcfc9c92`.
Native`bfb8c31`, runner`fbdaee3`, replay`5043b7a`; no next allocation selected.
## B16 actor/critic gradients lower frozen quality

The fresh500k comparison finishes in **37m52s**, but enabling actor/critic
gradients lowers frozen score **11.875→−0.542** and wins **24/24→10/24** against
the B16 control. Own initial scores−20.375 with0/24 wins. Retain disabled
actor/critic gradients for this Pong recipe; no extension or seed confirmation.
Both candidates miss the original mean15/90%-wins gate. The2.30% wall reduction
does not justify the quality loss, and the older-kernel control limits attribution.

Fresh1009/eval4000001009 keeps4M/RGB48/Auto/N8/B16/T16/H15/R32, coefficient.01,
actor0/full18/original rates/no aid. Only native `actor_critic_gradient` changes.
Training completes500,000 aggregate actions/1,999,767 emulator frames/62,454
updates in2271.883s, first376, zero debt and191 natural matches. All62,454
numeric reports pass with nonzero initial policy/value losses. Initial346 shared
tensors and semantic frozen-initial traces match the control; the enabled
checkpoint contains357 tensors, including11 frozen actor copies.

Frozen final/initial collect92,720/24,200 actions and370,844/96,771 frames in
35.543/9.297s. Each has exactly24 natural matches, no excess or truncations;
all partial tails remain. Zero updates and all357 frozen tensors unchanged.
Every config/source/counter check, four100k recoveries, complete CPU replays/
videos, three guards, curves and comparison pass. The service exits successfully
19:36:47 UTC,39m28.691s total. No training restart, cutoff or GPU repeat.
The separate progress-observer exit143 and reattachment remain documented in
the [experiment](../experiments/2026-10-26-cdp-pong-actor-critic-gradients.md).

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-r32-acgrads-intrinsic001-500k`.
Review SHA256`209c008b9c3770af2f59c7c69a22e26e4a9df676b689a341da7ba156ed2ad03e`;
curve`419d47f1ef9f6a6a3055bb15c915a7a342c3c299c8750855166ff4fb13c5636a`;
comparison`811f4fa23c91200153e072d8f9f8a8e7728fdca0770838533e066ee132063bd2`;
completion`9942fe280aa39ecd3b8e2efad5fc214406e5e7d5035e6128cc5d7fd4ba7acaeb`.
Launch source`7ebd7f4`, native`bfb8c31`, runner`fbdaee3`, replay`5043b7a`.
The separately prepared bounded Qbert pixel-guidance check is selected after
review; no further Pong allocation follows.

## A pixel-rule development match loses

A CPU-only native-RGB reference loses its single development match on seed9001,
score−15, after4302 actions/17,208 frames in2.175s. Exact fresh pixel/action/
reward/boundary replay takes2.176s. No agent, GPU, RAM or learning is involved;
no training aid or full-cohort assessment is selected from this weak result.

The rule projects an incoming white ball toward the green paddle, reflects
vertical motion between fixed board bounds, offsets the target toward an edge
away from the visible opponent, and estimates the paddle's stopping position
using1.5 times its previous-step displacement with a four-pixel dead zone.
These are hand-coded assumptions derived from existing native images and one
fixed-action calibration. One development match does not establish reliability
or isolate any rule's effect. Native learning remains stronger than this rule.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-pixel-guidance-dev/v1`;
policy`pong_pixel_policy_v1.py`, driver`run_pong_pixel_dev_v1.py` in the runtime
root. Review SHA256`9fbc646c8f1843d34f55d542dea62864f59f40126bde7c50401b2f3a499f6cfc`;
trace`0c813912cd86bca12a00d44156f2524ab9aec47d53cede78fca4a4fb987e34a5`.

## RGB32 bounded check prepared, not selected

Prepare16,384 actions/4035 updates plus1024 frozen restore, with180s/60s caps,
on the existing4M/B8/T16/H15/R32/coefficient.01/actor0/full18 recipe. Only native
observation geometry48→32 changes; keep Auto, original rates, no aid, fresh1009/
eval4000001009, first248 and zero debt. Preserve the actual initial checkpoint,
all346 frozen tensors, finite reports, selected weight changes and both guards.
Geometry changes parameter shapes and initialization; do not assert matched
initial tensors versus RGB48. The control also used the older kernel/module.

The existing action256 snapshot retains a ball-colored spot at32: local peak
RGB162/104/60 versus background144/72/17, substantially weaker contrast than48.
This establishes visibility in that image, not adequate detail for learning.
Keep native max-pooled frames and one GPU32 resize. The completed RGB48 control's
matching16,384-action prefix takes94.054s; no candidate timing is measured yet.

Queue/cap checks, source compilation and unselected-launch refusal pass. No new
native build or GPU work occurs. The active Qbert run must complete and be
reviewed before selecting this check; no full learning allocation is selected.
Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-32-b8-smoke`;
controller`run_pong_rgb32_smoke.py`. Preparation SHA256
`07fa7fe6727ff31049cf4ffb254ff0feab731b11707c3bbf5de483a930099ec0`.


After the completed Qbert pixel-guided comparison and all audits are reviewed,
select this bounded check only. Qbert score 633.333 and coverage 11.667/21 are
below its unaided control, with 0/24 pyramids; it receives no extension.
The GPU is idle before selection. Keep the declared 16,384/4,035 and 1,024-action
budgets, 180s/60s caps and all checks. Owner
`kindle-cheap-pong-rgb32-smoke-20261026`. No full RGB32 learning run is selected.


## RGB32 bounded check passes with lower early cost

The bounded check completes **16,384 actions/65,523 frames/4,035 updates in
82.320s**, versus **94.054s** for the RGB48 control prefix: **12.48% less early
training time**, at identical action/update counts. Actual trainable parameters
are 3,244,305, comprising 2,814,336 world and 429,969 behavior parameters,
versus 3,578,225 at RGB48. Construction takes 1.975s separately.

All 4,035 finite reports, native config/source/counters, first248/zero debt,
selected CNN/reward-weight changes and both guards pass. Ordinary restore uses
1,024 actions/4,096 frames in 0.433s, zero updates and no natural matches;
all 346 tensors stay unchanged. Service starts20:13:27/source`4171753` and exits
successfully20:14:56 UTC, 1m29.582s total. This establishes bounded execution
and early cost, not frozen competence, sufficient visual detail or a full-run
speedup. Geometry/initial weights/trajectories and the control's kernel differ.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-32-b8-smoke`.
Review SHA256 `d05b0446290cd51b296becc6ffa7b58a24eaecb0701eff3b93360b6b87f638df`.
After reviewing the saving and all checks, select the fresh comparison below.

## Fresh RGB32 learning comparison

Select one fresh 500k-action/124,939-update run, retaining the strongest compact
B8 recipe:4M/Auto/N8/B8/T16/H15/R32/coefficient.01/actor0/full18/original rates,
no aid and disabled actor/critic gradients, fresh1009/eval4000001009. Only native
image geometry changes48→32. Keep native max-pooled RGB and one GPU resize.
Own actual-initial weights must match the bounded RGB32 initialization, but no
bounded experience is reused and no initial-weight match to RGB48 is claimed.

Keep15,992,192 sampled positions/239,882,880 imagined actor rows, first248 and
zero debt. Recoveries at100k/200k/300k/400k and fixed final500k; training cap
50 minutes (control55). Each frozen arm retains600k actions/300s maximum and
selects the first3 natural matches per lane,24 per policy, keeping excess/tails.
Final and actual initial use zero updates/truncations, original mean15/90%-win
gate,346 unchanged tensors and full config/source/counter/replay/video/guard/
curve comparison. Geometry changes shapes and initialization; an older-kernel
control and one reused seed limit attribution and reliability claims.
No cap increase, extension, earlier-checkpoint selection or automatic next run.

Queue flags/caps/recovery schedule, script compilation, unselected-launch refusal
and the completed-control comparison reader pass. Native unchanged; runner
`137ed77`, frozen replay`5043b7a`, overlay`runtime-qbert-guidance`.
Preparation SHA256 `a52432ef9fd7838f54b03820feb04359f82eaad704c9093e8bdafdeffba979e4`.
Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-32-r32-intrinsic001-500k`;
controller`run_pong_rgb32_500k.py`, owner
`kindle-cheap-pong-rgb32-intrinsic001-500k-20261026`.


The fresh run starts20:18:26 UTC/source`ebcd406`, invocation
`fceafef2d0154c7b978d3c167e73c167`. Startup verifies exact native config,
3,244,305 parameters, fresh counters, no external aid, all346 actual-initial
tensors matching the bounded RGB32 check, and first248:1/256:2/256:3 finite
updates. Four recovery audits, progress observation and final CPU postprocessing
are attached. Full-budget cost and frozen quality remain pending.
Selected declaration SHA256 `d5ef6bcf9c03a7f52ecc3cf76213c8e1509a6f5ae9067c1b448faa5789051f57`;
startup `dc7413938c9ce06c67b103ad404197fedb8cf2599bf9f3533822e98b29e45285`.


The100k recovery passes24,939 updates and all346 finite tensors. Its complete
105,552-action log prefix has26,325 finite reports;102,400 progress is518.986s,
25,539 updates and zero debt. The same process continues. During training, a
separate purported CPU Qbert fixture accidentally executes13.690s of zero-update
inference on the integrated AMD GPU. It exits; the original failure/correction
are retained in the cheap-Qbert report. This adds a co-host timing caveat;
there is no RX7900XT restart or training-cap change.


The200k and300k recoveries pass49,939/74,939 updates and all346 finite tensors.
Their complete log prefixes contain51,477/75,133 finite reports at206,160/
300,784 actions. Latest captured299,008-action progress is1517.113s with74,691
updates and zero debt. Same PID32428/invocation; fixed500k/frozen evaluation
remains pending. Recovery SHA256s:200k`a6396fcefcd7f6722f0a11dc5d50df046b05e31637cee8268cf62b2f4a5b260a`;
300k`2b0e92c5477cb8d2e00d27c25c04db50f5f622eabedcd0bb6a4ade0fb34d383a`.


## RGB32 saves time but substantially lowers frozen quality

The fresh500k comparison completes in **42m19s**, **15.24% less training time**
than RGB48's49m55s, but frozen score falls **14.958→2.292** and wins **24/24→16/24**.
Own initial is−20.375/0 wins. Retain RGB48; no RGB32 extension or confirmation.
The unchanged mean15/90%-win gate fails. The completed plot shows slower online
learning at both matched actions and elapsed time; its late rise does not
justify an unchanged extension. This result does not invalidate the separately
successful RGB32 Boxing/Freeway recipes.

Training completes500,000 actions/1,999,751 emulator frames/124,939 finite updates
in2538.665s, first248, zero debt and154 natural matches. Actual3,244,305 parameters
and all346 initial tensors match the bounded32 check; no experience is reused.
No matched-initial-weight claim is made versus RGB48. Native config differs only
in observation geometry, with native max-pooled frames and one GPU resize.
One reused seed, changed shapes/initialization, the control's older kernel/module
and the recorded13.690s unintended integrated-GPU inference limit attribution.

Frozen final/initial use108,528/23,360 actions,434,066/93,415 frames and41.661/
9.020s. Select first3 natural matches per lane,24 each; one excess final match
and all partial tails are retained. No truncations or learning updates; all346
tensors remain unchanged. Full source/config/counter checks, four recovery
audits, complete CPU replays/videos, three guards, curves and comparison pass.
Service exits successfully21:02:34 UTC,44m07.700s total wall. No restart, cutoff
or GPU stage repeat. Launch source`ebcd406`, native`bfb8c31`, runner`137ed77`,
frozen replay`5043b7a`.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-32-r32-intrinsic001-500k`.
Review SHA256`b6cd101b672c312ec59db081e5f744de509de4dfe77859df8c505bbe657386eb`;
curve`211bca2a82aa98b715d8597c06429e709bfda1d51be8f3807023bfc3a3976ef4`;
comparison`2c4857c6dbe07b98d43314df4d5576dd66ce3fa2a7a028d2c09251e1dfb33c0b`;
completion`7997eb67114b8020b0c77efec86e62575a46de40ba02e7cee658a5f481884f3c`.
After review, select the prepared fresh Qbert capped-guidance comparison.


## Separate actor/critic rate: bounded check

After the completed capped-Qbert comparison remains below its unaided control,
select a bounded Pong check of **B16 with only the actor/critic learning rate
raised from4e-5 to8e-5**. B16's completed500k run costs38m45s versus B8's49m55s,
but scores11.875 versus14.958. Doubling all rates gave9.125. That failed global
change does not isolate the actor/critic rate; a separate behavior rate tests
whether the cheaper update cadence can retain more policy learning while keeping
the world/encoder/dynamics rates fixed. This is a hypothesis, not evidence of
learning-rate equivalence between batch sizes.

The native runtime already reads `behavior_learning_rate` when configuring the
behavior optimizer. Add fresh-only `--behavior-learning-rate` to the Python CLI;
omitting it preserves the native fallback to the base rate. Reject invalid F32
values/ratios and any restore override before creating environments or output.
**460 relevant CPU tests pass**, including unchanged base/CDP rates and default
behavior, explicit forwarding, invalid values and restore refusals. Native
binary, architecture, optimizer coefficients and learning operations are unchanged.
The existing `runtime-qbert-guidance-cap` overlay remains compatible.

The selected bounded allocation keeps4M/RGB48/Auto/N8/B16/T16/H15/R32,
coefficient.01, actor mixture0, full18 actions, disabled actor/critic gradients,
base4e-5/encoder6e-6/dynamics4e-4, and1009/eval4000001009. Train4096 aggregate
actions for466 updates (first376, zero debt), with a90-second cap, then restore
for1024 frozen actions with a60-second cap. Require config/source/counter checks,
all346 initial tensors matching the B16 control, selected CNN/reward/actor/critic
weight changes, finite reports/checkpoints, all346 unchanged frozen tensors and
both host guards. This does not evaluate natural-match competence. No experience
from this check will be reused in a future fresh run. A full learning allocation
requires reviewing this bounded result; no automatic next queue follows.

Evidence: `/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-behavior-lr2-smoke`;
owner`kindle-cheap-pong-behavior-lr2-smoke-20261026`;
controller`run_pong_behavior_lr2_smoke.py`. CLI validation is retained at
`/x/Temp/kindle-cheap-20261025/cdp-behavior-rate-cli/validation.json`.


The bounded run starts21:37:12 UTC/source`a2c467c` and exits successfully at
21:37:35 UTC. It completes4096 actions/16,384 emulator frames/466 updates in
15.9788s; frozen1024 actions/4096 frames take0.4375s, with zero updates and zero
natural matches. All346 initial tensors match the B16 control, all346 frozen
tensors are unchanged, all466 reports and checkpoints are finite, selected
CNN/reward/actor/critic weights change, and both guards pass. This is execution
and restore evidence only. Review SHA256
`651052b477f825a024637248c9a961b8aebd19c285288e7e42897346ab858d01`.

## Selected fresh B16 behavior-rate comparison

After reviewing the bounded success and terminal service, select **one fresh
500k-action/62,454-update B16 run**, with a45-minute training cap. Only behavior
rate becomes8e-5 versus the original B16 control; keep world4e-5, encoder6e-6,
dynamics4e-4,4M/RGB48/Auto/N8/B16/T16/H15/R32/coefficient.01/actor0/full18,
disabled actor/critic gradients,1009/eval4000001009, and original optimizer
coefficients/warmup. First update376, zero residual debt, recoveries at100k/
12,454,200k/24,954,300k/37,454 and400k/49,954 updates. No smoke experience is
reused. Native`bfb8c31`, CLI`a2c467c`, frozen replay`5043b7a`; no native rebuild.

Assess only fixed final and actual initial policies, each first3 natural matches
per lane (24 total),600k-action/300-second evaluation caps, zero updates and
truncations, retained excess/tails,346 unchanged tensors, complete action/reward/
boundary replay and videos. Keep mean15/90%-wins. Require all finite reports,
config/source/counter audits, four recovery receipts, matched346 initial tensors
and semantic initial traces, guards and learning curves. Compare original B16
11.875/38m45s, globally doubled rates9.125/38m32s and B8 14.958/49m55s. Controls
use older kernels; neither timing nor optimizer equivalence is isolated. One
reused development seed is not independent reliability evidence.

The controller, queue, complete-control readers, comparison and postprocessing
compile; unselected launch is refused, and an unchanged-rate control falsely
labelled as the new candidate is rejected before output. No cap increase,
resume, extension, selected earlier checkpoint, conditional seed campaign or
automatic next GPU allocation follows. Full frozen quality remains unmeasured.

Artifacts:
`/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-r32-intrinsic001-behavior-lr2-500k`;
owner`kindle-cheap-pong-behavior-lr2-500k-20261026`. Controller
`run_pong_behavior_lr2_500k.py`, comparison`compare_pong_behavior_rate.py`,
postprocessing`finish_pong_behavior_lr2.py`; exact hashes in the declaration.


The fresh run starts **21:41:35 UTC/source`523eaca`**, invocation
`8b9cabed37a048cb869b93bed3082a0d`. Startup verifies RX7900XT, exact declared
native config,3,578,225 parameters, fresh counters and first376:1/384:2/392:3
updates. All346 actual-initial tensors match both the completed B16 control and
the bounded check. Progress, four-recovery monitoring and fixed-final
postprocessing are attached. Selected declaration SHA256
`0e22b1b038bdc53c4452373d0181bc417498d0bf2175f5373ca70a7735406350`;
startup`7f96a8eb44ad24d50b859de3f3bec2c4a3d2d0b15313f73aedba66cc2fa7138e`.

The first CPU startup-receipt attempt used the plotting environment, whose
transitive checkpoint-audit import lacks`ale_py`. The failure is retained in
`startup-review-first-attempt-failure.json`. Repeating only those read-only
checks under the existing Atari Python environment passes. Training retains
the same PID/invocation and cap; no GPU stage repeats. Frozen learning quality
remains pending.


The100k recovery audit passes at21:49:35 UTC:100,000 actions/12,454 updates,
346 finite saved tensors and13,275 finite reports in the complete106,576-action
prefix. Training debt at the latest progress is zero; the original service
remains live. Receipt SHA256`709f5129dcb938cada64df1cd4a9a42a5187de06581089d7bad5d1f019d5b5f2`. This is recovery/accounting evidence;
fixed-final frozen quality remains pending.


## Behavior-rate doubling gives a small gain but still misses the gate

The selected fresh run completes500,000 aggregate actions/1,999,751 emulator
frames and62,454 updates in2257.534s (**37m38s**), with first update376 and zero
residual debt. The original service exits successfully at **22:20:58 UTC**;
total service wall time39m22.621s includes construction, evaluations and audits.
Only behavior rate becomes8e-5; world/encoder/dynamics rates remain unchanged.

| Fixed500k recipe | Training wall | Frozen mean | Natural wins |
|---|---:|---:|---:|
| B8, original rates |49m55s|14.958|24/24|
| B16, original rates |38m45s|11.875|24/24|
| B16, all rates doubled |38m32s|9.125|24/24|
| B16, behavior rate doubled |37m38s|12.375|24/24|

All four own-initial controls score−20.375/0 wins. The new final policy's
conditional stream-bootstrap interval is[10.5,14.0];24 matches are not24
independent learners. The0.5-point difference from original B16 is a small
unconfirmed gain. It remains below B8 and the unchanged mean15/90%-wins gate.
Training takes24.63% less time than B8, but older control kernels limit isolated
timing attribution. Retain B8 as the strongest compact recipe; no extension,
selected earlier checkpoint or seed confirmation is selected from this result.

All62,454 reports are finite. Config/source/counter audits, four100k recoveries,
346 matched initial tensors across all four arms, identical semantic initial
traces,346 unchanged frozen tensors per assessment, full action/reward/boundary
replay, videos, host guards, curves and comparison pass. Final evaluation uses
105,448 actions/421,732 frames/40.308s and completes28 natural matches, selecting
the original first3 per lane. Initial uses24,200/96,771/9.311s and24 matches.
Both perform zero learning/truncation; excess episodes and partial tails remain.
The earlier CPU startup import failure remains retained; no GPU stage repeats.
Online curves improve sooner at the higher behavior rate, but fixed frozen
quality remains the decision measure. All seeds/cohorts are reused development
evidence; neither mastery nor optimizer equivalence is established.

Artifacts remain in the declared run directory. Learning-review SHA256
`050f0b1868c9d15ec022a22021458556ed61ba69b7b3ae8af6b145400d642909`;
comparison`1957825072c2db06040e74a01bc87397cbc82da796285016dfb850cb69b6774c`;
postprocessing`e1197f168f5845b2acffa1a71143f15ae27cd038f3cb5206aef598d95160e441`.


## Larger capacity at lower replay ratio passes the bounded check

The native12M/RGB48/Auto/N8/B16/T16/H15/R16 candidate completes16,384 aggregate
actions /65,523 emulator frames /1,001 updates in **78.872s**, versus94.054s
for the4M/B8/R32 control prefix:16.14% less training time. Construction2.171s
is separate. Coefficient.01, actor0, original rates, full18, no aid or imitation.
All finite reports,14,123,473 parameters, exact config/source/counters, changed
CNN/reward weights and both guards pass. First376, residual credit.5.
All346 actual initial tensors match the retained same-seed12M48 initial.
Frozen1024 actions take.708s, zero updates and346 unchanged tensors.
Service exits01:04:22 UTC on October27,1m28.029s total; no GPU repeat.

This measures execution and early cost, not natural-match competence. Capacity,
batch width and replay ratio change together; native/runner and control kernel
differ. Historical12M64 Pong success motivated the candidate without predicting
AMD learning. The completed compact4M recipes remain the measured task controls.

After reviewing the bounded result, select one fresh500k-action/31,227-update
comparison with45-minute training cap and fixed final/own initial24-natural
matches. Mean15/90%-wins remains unchanged. No smoke experience is reused,
no extension or automatic successor is queued, and no full-run score is known.
See the [protocol](../experiments/2026-10-27-cdp-pong-capacity-r16.md).
Artifacts:`/x/Temp/kindle-pong-capacity-20261027`.
Bounded review SHA256:
`b56204eb1e0d6233546e1d2832aba8b8ca37a526f3a60bffcfacbde2f2845b5f`.


## Larger capacity with less replay work reaches the gate

The fixed final12M/RGB48/B16/R16 learner scores **15.625 with24/24 wins**,
versus its actual initial **−20.292 with0/24 wins**. It meets the unchanged
mean15/90%-wins gate on this reused development seed. Training completes
**500,000 aggregate actions /1,999,714 emulator frames /31,227 updates in
2,487.555s (41m28s)**,16.95% less wall time than the strongest4M/B8/R32 control.
One passing learner does not establish training-seed reliability.

| Fixed500k recipe | Training wall | Frozen mean | Wins | Gate |
|---|---:|---:|---:|---|
|4M48/B8/R32|49m55s|14.958|24/24|fail: mean below15|
|4M48/B16/R32|38m45s|11.875|24/24|fail: mean below15|
|12M48/B16/R16|41m28s|15.625|24/24|pass on this learner|

The [declared recipe](../experiments/2026-10-27-cdp-pong-capacity-r16.md) keeps
centered CDP/Auto/N8/T16/H15, coefficient.01, actor mixture0, original rates,
full18/sticky.25/repeat4/noops0 and one GPU resize from native frames. Capacity,
batch/microbatch and ratio change together versus4M/B8; actual parameters are
14,123,473. Native/runner and the older control kernel also differ. This is a
useful measured recipe comparison, without an isolated capacity or speed claim.
No external aid, expert imitation or restored experience is used.

First update376, residual credit.5, and all four100k recoveries pass. Training
finishes within the45-minute cap;200 natural training matches finish with last50
mean12.28. The frozen final uses74,512 actions /297,998 frames in37.763s; own
initial uses24,376 /97,476 in12.525s. Each selects the first three natural sampled
matches per lane, exactly24 completed matches, zero truncations and zero updates.
All346 tensors remain invariant in each frozen arm. Conditional stream-bootstrap
95% interval for the final mean is[14.833,16.292]; the win Wilson interval is
[.862,1]. These intervals describe this policy, not training-seed reliability.

All31,227 finite reports, exact config/source/counters,346 actual initial tensors
matching the bounded same-seed check,47 prelearning transitions, recovery audits,
three guards and CPU comparison pass. Complete recorded-action replay checks
rewards, frame counts, boundaries and outcomes, and reconstructs videos. Ordinary
unguided logs contain no live pixel hashes; retain the protocol's explicit scope
correction rather than claiming direct live/replay image equality. All42 declared
input pins remain unchanged. Service exits successfully01:51:06 UTC on October27;
CPU plots/comparison finish01:51:35. No GPU or audit stage repeats.

The separately staged Breakout adapter uses16.038s of measured CPU test/replay
portions during this life, in addition to55.935s of earlier reference development.
It constructs no native agent and changes none of the active pins. Retain this
co-host timing caveat; no causal timing adjustment is claimed.

Artifacts, curves, comparison and videos:
`/x/Temp/kindle-pong-capacity-20261027/pong-12m-48-b16-r16-intrinsic001-500k`.
Learning review SHA256:
`0a0eb5df3a5b51bfcb44ff8d8c86dfc2cbecb3ede3ead9dcbd69acfa6a3de63b`;
completion`6decc800e5f68332065cb68a395d55b14ebc51de2a4ac9dec12b9796f61dc465`;
comparison`5626775b6f31b2bf739d7a1dbb1825eb21705c12b5ec11bc3d568bcb4d81c9b8`.
The completed learner is terminal. The result justifies the separately selected
[fresh4013 confirmation](../experiments/2026-10-27-cdp-pong-capacity-seed4013.md),
with identical recipe and budgets. It is not an extension or broad seed campaign.


## Capacity confirmation improves4013 but narrowly misses the gate

The fresh4013 learner scores **14.833 with24/24 wins**, versus its own actual
initial **−20.542 with0/24 wins**. It narrowly misses the unchanged mean15 gate.
The identical12M48 recipe passes on1009, so **one of two learners passes**.
Both improve their paired compact4M/B8 controls while using about17% less
training time; this is useful development evidence, not broad reliability.

| Seed |4M48/B8/R32 mean /wall |12M48/B16/R16 mean /wall |12M gate |
|---|---|---|---|
|1009|14.958 /49m55s|15.625 /41m28s|pass|
|4013|14.000 /49m58s|14.833 /41m31s|fail: mean below15|

All four final policies win24/24 natural matches. The new run completes
**500,000 aggregate actions /1,999,720 emulator frames /31,227 updates in
2491.468s**, within its45-minute cap. First update376, residual credit.5,
175 natural training matches and last50 online mean15.14. The exact recipe is
centered CDP12M/RGB48/Auto/N8/B16/T16/H15/R16, coefficient.01, actor0, original
rates/full18, no aid or imitation. Actual parameters14,123,473. Seed4013 and its
evaluation cohort were used in earlier4M development. Capacity, batch, ratio,
native and older control kernel confound attribution against compact controls.

Final evaluation uses85,368 actions /341,420 frames /43.478s; actual initial
uses23,408 /93,605 /12.004s. Each completes exactly24 natural matches, first3
per lane, zero learning/truncation and346 invariant tensors. Conditional
stream-bootstrap mean interval[13.000,16.708] and win Wilson[.862,1] describe
this frozen policy, not training-seed reliability. Do not pool episodes as
independent learners or relax the gate because the point estimate is close.

All31,227 reports, config/source/counters,53 declared pins, four100k recovery
checkpoints, finite tensors, guards, recorded-action outcome replays, videos,
curves and paired comparison pass. Ordinary unguided logs have no live pixel
hashes, so direct live/replay image equality remains untested. Startup checks
fresh world weights and the expected update trace; no cross-seed weight equality
is claimed. Service exits02:41:08 UTC on October27, CPU postprocessing02:41:23.
The initial progress-observer flag error is retained, corrected without
interrupting training or repeating GPU stages. Co-hosted CPU reference work is
recorded in the protocol. Neither learner is extended; no third seed is selected.

Artifacts: `/x/Temp/kindle-pong-capacity-20261027/pong-12m-48-b16-r16-intrinsic001-500k-seed4013`.
Learning review SHA256`a5d261a0e75ec8146ccdc361cabca6581fb403c1a0e5a27d931487239c86d437`;
completion`cfb0fbe1738dc8728302a50799105a12003069eafaa04200c0757af5e517442e`;
comparison`5583ccb3e8a3d724c0c437e0b18385235d44efea003d764ef3557eb1c146f67e`.


## A calibrated pixel reference wins one development match

One external CPU controller replaces the old1.5-times-displacement stopping
heuristic with a fixed calibrated response and five-command predictive search.
The detector, reflected incoming-ball target and four-pixel opponent-dependent
target offset stay as in the retained reference. Fit response coefficients only
from the first3000 actions of its existing9001 trace, excluding clipped boundary
positions. The2874 fit samples have2.117-pixel mean absolute one-step error;
1226 later same-episode samples have2.122. This chronological split is correlated,
not independent evaluation. It is calibration of a hand-coded reference, not
Kindle pretraining or a native model change.

With the fixed candidate on the same reused9001 development seed, the natural
match improves **−15→+12**:11,216 actions /44,863 emulator frames,6.652s collection
and6.530s exact fresh input/output-pixel/action/reward/boundary replay. Both use
full18/sticky.25/repeat4/noops0/native RGB. The declared16384-action/60-second
per-pass caps remain fixed. Planning and response calibration change together;
this does not isolate either effect. No native module loads or learner runs.

The candidate wins one match but remains below mean15 on that match. It does
not justify a Pong guidance adapter, full reference cohort, native allocation
or reliability claim. Keep the completed native12M recipe as the stronger
measured learning result and the five-game goal open. No further CPU variant
is selected here.

All56 active Qbert pins and source hashes remain unchanged. Successful fit,
collection and replay take13.225s on one affined low-priority core during Qbert;
retain this timing caveat. Artifacts:`/x/Temp/kindle-pong-guidance-20261027`.
Reference review SHA256`0da6d71a9496e435413b5ba70429a92ff11157fde83d9772a15372239de57c3a`;
response fit`6860f3f127ebcc24118368eed925c8c0aefe19a3ac2694052998d26facf6466b`. The old losing reference remains
retained; native training/inference stays on Meganeura and Blade.

## Prepared greedy comparison of both retained12M learners

The two completed12M learners win24/24 sampled matches each, but their means
15.625/14.833 leave only one above15. Prepare a fixed four-arm assessment:
both retained final policies and each learner's actual initial weights, using
the existing native greedy path. The earlier4M greedy result was slightly worse
(14.333 versus14.958), so this is a model-specific check, not evidence for a
global sampler change. No new learner or further training is proposed.

The exact original Pong runner is exported from`002e403b`; its SHA256
`814eef835aebdcc6d6adbe7a670f922f48525dafcf51ee0b2bcd9e53482dcdba` matches
both controls. Use their original packaged runtime/native79c9a7f, unchanged
wrapper, ROM, host, seeds4000001009/4000004013, full18/sticky.25/repeat4/native
frames/one GPU48 resize. Only greedy action selection and output locations
change. Each arm keeps first3 natural matches per8 lanes=24,600k-action/300s
caps, zero updates/truncations and the original mean15/90%-win gate. Require
346 invariant tensors, source-file invariance, full outcome replay/videos and
host guards. Ordinary frozen logs lack live pixel hashes.

Four commands parse, five scripts compile, all four retained checkpoints have
346 finite tensors each, and every source sampled score/criterion recomputes
exactly from its verified replay. The controller refuses while unselected.
Preparation/checking use.361s/1.429s on one affined low-priority core, with no
native module, agent or environment construction and all47 active Qbert pins
unchanged. This supplies preparation evidence only; no greedy score exists.

An additional.016s CPU check matches all10 original runtime pins from each
source declaration, covering the native binary and all nine Python files.
Receipt SHA256`b3426a3b3fb40c69764ecee0e83f72cbd64afebf4950e1e07da949177080423a`.

The assessment remains **unselected** while the R16/20% Qbert learner runs.
It cannot start before that service and its postprocessing finish successfully
and a separate selection supplies the terminal receipt. No automatic launch,
retry, extension, checkpoint selection or default change follows preparation.
Both seeds/cohorts are reused development evidence. Original sampled outcomes
and training costs41m28s/41m31s remain; changed greedy results cannot relabel
the sampled failure or become new independent learners.

Artifacts:`/x/Temp/kindle-pong-greedy-20261027/pong-12m-48-two-seed-greedy-frozen`.
Prepared declaration SHA256`ad41c3ae99e06594269030199d5fb228c98e4420e8af718eb4bcd4ffee4923a8`;
CPU preflight`615dc7fdb3d22e8193629f8333c44fe7c1ecb70755ea12dd54a5b2474a222ebe`;
controller`acf3f12b2df12be2476c540537d5ce7485a0b1159490f56948ca14764265ac20`.

## Paired greedy assessment selected after Qbert completion

Qbert R16/20% and all postprocessing finish successfully at05:23:55/05:24:19
UTC. Its complete receipt is SHA256
`eeccfd460ba6afe09618cfd21e250f24bcf44b294ab5c4c68cc84f8b8b6c0515`.
Select the prepared four-arm greedy assessment without changing its58 pins,
original runtime, seeds, fixed arm order, checkpoints, action/time caps or
mean15/90%-win gate. This tests whether greedy inference improves the retained
12M pair at no additional training cost. The earlier4M greedy regression remains
recorded; this is not a global default change. Both original sampled outcomes
and training costs remain. No retry, extension, cap increase, checkpoint
selection or automatic successor. The previous unselected preparation is
historical; this separate decision authorizes only the fixed assessment.

## Greedy evaluation passes the gate on both retained12M learners

Both retained12M policies score **15.208 with24/24 wins** under greedy action
selection and the unchanged mean15/90%-win gate. The fixed four-arm assessment
adds **zero training actions or updates**. It slightly lowers1009 and improves
4013, leaving both just.208 above15. Retain this as a useful12M development
option. It is not a global greedy default or new independent training evidence.

| Retained learner | Sampled final / own initial | Greedy final / own initial | Final wins, sampled / greedy | Retained training |
| --- | ---: | ---: | ---: | ---: |
|1009 |15.625 /−20.292 |15.208 /−20.583 |24/24 /24/24 |41m28s |
|4013 |14.833 /−20.542 |15.208 /−20.625 |24/24 /24/24 |41m31s |

Each actual-initial control has0/24 wins under either action policy. The
sampled4013 failure remains recorded. Both development seeds and evaluation
cohorts are reused; this changes the frozen action-selection protocol and
cannot create extra independent learners. Conditional stream-bootstrap95%
intervals are14.292–16.250 for1009 and13.958–16.292 for4013; both cross15.
The24/24 win interval is.862–1.000, also below.9 at its lower endpoint. These
are the existing descriptive uncertainties, not a new gate or a broad
reliability claim. The prior4M greedy regression remains recorded.

Keep12M48/Auto/N8/B16/T16/H15/R16/coefficient.01/actor0/full18/original rates,
no training aid or imitation; each learner originally used500k actions/31,227
updates. Exact original runner/runtime/native79c9a7f, wrapper, host, ROM,
sticky.25/repeat4/noops0/native frames/one GPU48 resize and evaluation seeds
4000001009/4000004013 match the sampled controls. Greedy action selection and
output locations are the only evaluation changes. All58 pins and every source
file stay unchanged.

| Fixed evaluation arm | Aggregate actions | Emulator frames | Evaluation seconds |
| --- | ---: | ---: | ---: |
|final1009 |78,080 |312,258 |42.953 |
|initial1009 |21,448 |85,774 |12.010 |
|final4013 |82,128 |328,463 |45.156 |
|initial4013 |21,112 |84,432 |11.797 |
|Total |202,768 |810,927 |111.916 |

All four arms complete first3 natural matches per8 lanes,24 selected per arm,
with zero updates/truncations and346 invariant tensors each. Source identity,
config/counters, complete recorded-action reward/frame/boundary replays,
videos and host guards pass. Ordinary frozen logs lack live pixel hashes;
these are outcome replays, not direct live-image equality. No cap, retry,
checkpoint selection or GPU repeat. Main service takes3m32.855s including
construction/audits, exits05:29:46 UTC; terminal observer completes05:29:51
on October27. Original training costs remain and the evaluation adds cost.
No extension, third learner or successor allocation is selected.

Artifacts:`/x/Temp/kindle-pong-greedy-20261027/pong-12m-48-two-seed-greedy-frozen`.
Audit SHA256`d13581a9dd6227a9639e7abf99160650b6799d533565f6f3ab0ae1fe1a41f6d7`;
completion`57128cfdbbb46719cd1b87c27063b7f36834473318800880e2043d46ef1e5b35`;
startup`f627e2e491af68cb5f93ee659ee1ac6fb743b4d0ec3c9d10a20bcb2cd7dfca36`;
selected declaration`9a7c489a213fa7c58720f72921aafe9ea0c8c28677953fbc5ecc6850374130f9`.
