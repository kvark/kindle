# Small-image Qbert gains score and coverage but completes no pyramid

Latest: the [compatible RGB control](2026-10-29-cdp-qbert-rgb-control.md)
favors retained CDP4M/expert100 on the fixed learning-per-minute comparison:
greedy9,654.167 in26m07s versus RGB6,090.625 in28m41s. Both miss15k;one reused
seed/cohort with strong supervision does not establish broad reliability.
All audits pass;no successor is selected. Earlier development results follow.

The fixed200k screen learns in19m45s: frozen809.375 versus actual initial180.2083.
It reaches more of the first pyramid, but neither policy completes one. The
existing15k/90%-first-pyramid gate fails; this is useful development progress,
not successful Qbert play or seed reliability.
The coefficient100 follow-up below also scores809.375, completes no pyramids
and reduces mean coverage to13.708/21 in20m12s. A stronger bonus does not improve
the fixed-budget result on this reused seed; both runs pass all audits.
The six-action test below finishes in18m21s but scores605.208 and reduces coverage
to10.167/21, still0/24 pyramids. Faster execution does not improve this task result.

[Declaration](../experiments/2026-10-25-cdp-cheap-qbert.md). Fresh seed1009,
Size4M/RGB48/Auto,3,578,225 trainable parameters, centered CDP/action-effects1,
N8/B8/T16/H15/R32, full BPTT/microbatch8 and replay100k. Full18/sticky.25/repeat4,
native max-pooled frames and one GPU resize. No aid or reward change. RAM is used
only by the existing post-hoc CPU pyramid observer, never by the learner.

| Fixed policy | Natural episodes | Mean score | First pyramids | Mean maximum initial cubes |
|---|---:|---:|---:|---:|
| 200k final |24|809.3750|0|16.625/21|
| Actual initial |24|180.2083|0|5.833/21|

Final coverage ranges12–20 cubes, median16.5; initial coverage ranges0–11,
median6. The trained policy reaches20 cubes once. This localizes the remaining
task gap to incomplete first-pyramid play; it does not establish whether more
experience, exploration, visual precision or dynamics changes would fix it.

An October26 CPU follow-up tracks which cubes ever meet the existing color
criterion, as well as the maximum simultaneously colored count. Final mean
ever-colored coverage is16.667, and no episode ever colors all21; one cube is
missed in21/24 episodes. Thus the evaluated policy has a per-episode coverage
gap. This does not identify the underlying learning failure or a successful
intervention. The8.27-second saved-action replay reproduces all original
episode/tail outcomes, rewards, frames and resets, with no agent or GPU work.
`tile-coverage-review.json` retains per-cube RAM addresses, masks and source/script
hashes; it introduces no new spatial-label interpretation or training input.

## Scale of the existing exploration bonus

A CPU review of all49,939 learner reports checks the diagnostic semantics against
the actual native-source revision. The reported imagined intrinsic reward is
already scaled; the reported total imagined reward includes that bonus. Subtract
the bonus to estimate the mean predicted extrinsic component on the same rows.

| Update window | Mean intrinsic bonus, coefficient1 | Inferred extrinsic mean | Bonus/extrinsic at1 | At100, holding recorded predictions fixed |
|---|---:|---:|---:|---:|
|1–1000|0.017948|0.209685|8.56%|8.56 times|
|1001–25000|0.001255|0.572724|0.219%|21.91%|
|25001–49939|0.001305|0.894999|0.146%|14.58%|

The coefficient100 candidate would make the late bonus more substantial, but
would also dominate predicted extrinsic reward during the first1000 updates.
These are arithmetic rescalings of recorded minibatch means, not predicted
training outcomes or per-action bonus distributions. A new learner would change
the states, ensemble, predictions and return normalization. The bonus enters
imagined actor/critic and replay critic targets; it is not an action override.
The small late mean does not prove that weak exploration causes the coverage gap.

This motivates a [bounded coefficient test](../experiments/2026-10-26-cdp-qbert-intrinsic-scale.md),
with the model, geometry, native runtime, seed and200k budget fixed. Its completed
result follows below. The analysis constructs no agent and takes0.513s of CPU wall time.
Evidence: `intrinsic-scale-review.json`, sibling`review_qbert_intrinsic_scale.py`;
review SHA256`679f92860b5d67bda49716ff7b162fc1f312db22fc08a02753969b4ea4a48b14`.

## Completed run and audit

Training completes200,000 aggregate actions,799,342 emulator frames and49,939
updates in1,184.523s (19m45s),168.84 actions/s. First update248, zero debt,
442 natural training episodes, final online last50 mean677.5. Frozen evaluation
uses seed4000001009, first three natural sampled episodes per lane. Final uses
15,376 actions/61,462 frames/6.221s; initial uses9,232 actions/36,888 frames/3.740s.
Each arm selects all24 completed episodes, with no excess or truncations;
unfinished tails remain in the complete logs and replays. Zero frozen updates,
all346 tensors unchanged per arm.

All49,939 numeric reports, finite checkpoints, complete configuration/source/
counter checks, full CPU replays/videos and three host guards pass. Initial346
tensors match the qualified4M/RGB48 smoke. Service
`kindle-cheap-qbert-r32-200k-20261025` exits successfully without a retry, cap
increase or extension. Native`bfb8c31`, runner`a99416e`, native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`, RX7900XT/RADV.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-200k`.
`curve/` contains independently reconciled action/frame/time curves;
`coverage-review.json` retains the selected per-episode cube counts and hashes.
Complete action and learner windows remain as diagnostic evidence, separate
from frozen competence. Review SHA256
`f69a5feac5e341b1749c2503d59c6290bb1126af1ea3ca7d80054be8b20e1060`.

Seed1009 is reused development data. Capacity, resolution, runtime and budget
differ from historical controls; no isolated speedup or superiority claim.
Retain this result without extending it. The subsequent Breakout tests are
reported separately; the stronger-bonus comparison is a fresh controlled test.

## Coefficient100 does not improve the fixed-budget result

The [declared test](../experiments/2026-10-26-cdp-qbert-intrinsic-scale.md) changes
only intrinsic_reward_scale1→100 in the native configuration, retaining
4M/RGB48/Auto/B8/T16/H15/R32, seed1009, actor_unimix0 and no external action aid.
All346 actual-initial tensors match the control. Their frozen action/reward/
boundary/frame traces and replay outcomes also match, excluding wall timestamps.
The CLI adds optional mixture/horizon controls; their values stay0 and15.

Training completes **200,000 actions /799,449 emulator frames /49,939 updates
in1,211.601s (20m12s)**,165.07 actions/s, within the unchanged30-minute cap.
First update248, zero debt,385 natural training episodes, final online mean701.5.
The100k recovery checkpoint passes its finite/config/counter check before fixed
200k replaces it. No favorable-checkpoint selection or extension occurs.

| Intrinsic coefficient | Frozen final / initial | First pyramids | Mean maximum initial cubes | Training wall |
|---|---:|---:|---:|---:|
|1|809.375 /180.2083|0/24|16.625/21|19m45s|
|100|809.375 /180.2083|0/24|13.708/21|20m12s|

Coefficient100 coverage ranges11–15 cubes, median14, versus control12–20,
median16.5. Equal mean score does not imply equal policies or trajectories.
Neither recipe reaches the existing15k/90%-first-pyramid gate. The larger bonus
does not improve frozen score or completion and has worse coverage on this seed.

Final evaluation uses17,232 actions/68,890 frames/8.044s; initial uses9,232/
36,888/4.289s. Both select all24 natural episodes, with no excess episodes,
truncations or updates. Partial tails remain. All346 frozen tensors stay exact.
All49,939 strict numeric reports, full config/source/counters, finite checkpoints,
full CPU replays/videos and three host guards pass. The service exits successfully.

The actual late bonus/extrinsic means become0.151820/0.958327, or15.84%, versus
0.001305/0.894999, or0.146%, in the control. In the first1000 updates the ratio
is8.57 times. Late policy entropy is1.4395 versus1.5558. These are minibatch means
along different learned trajectories, not matched-state comparisons or a causal
diagnosis of the coverage loss. The stronger coefficient changes learning, but
its larger measured bonus does not establish useful exploration.

Source at launch`0b151c6`, native`bfb8c31`, runner`1ed07af`; native SHA256 remains
the same as the control. Artifacts:
`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-intrinsic100-200k`, including
`curve/`, `coverage-review.json` and `coefficient-comparison-v2/`.
Review SHA256`095b4e03bf4ea634828d3cbab71943edc7f33f78781c2fad8c411d8677095317`.
Comparison SHA256`b5c7667373210a47c6b26ef19d93d15d84b8e39a74858e32f19da59a1ca5a1e7`.
The first CPU comparison included wall timestamps in initial episode identity;
v2 excludes only those fields, retaining v1. Scores, coverage, curves and bonus
statistics are unchanged; no GPU run is repeated.

Do not extend this failed setting or launch a broad coefficient/seed sweep.
The bounded H5 Breakout warm start is selected after this completed review.

## Six action choices run faster but reduce frozen coverage

The [six-action test](../experiments/2026-10-26-cdp-qbert-minimal-actions.md)
completes200k actions/49,939 updates in **1,101.180s (18m21s)**,181.62 actions/s,
with799,288 emulator frames, first update248 and zero debt. The native config
matches the coefficient1 control except action_count18→6;3,568,997 parameters.
The `published-minimal` protocol retains repeat4, sticky.25, zero no-op starts
and100000-frame episode limit. Network dimensions, initialization and policy
probabilities change, so this is a labelled protocol variant with its own
actual-initial control, not a matched-initial-weight comparison.

| Fixed policy | Selected natural episodes | Mean score | First pyramids | Mean maximum initial cubes |
|---|---:|---:|---:|---:|
| Six-action200k final |24|605.208|0|10.167/21|
| Own actual initial |24|148.958|0|4.667/21|
| Full18 coefficient1 control |24|809.375|0|16.625/21|

The faster execution does not improve frozen play. Final coverage ranges6–17,
median9; no episode completes the first pyramid. The original15k/90% gate fails.
Training finishes489 natural episodes with final online last50 mean656.5. The
100k recovery checkpoint passes before fixed200k replaces it; no earlier-policy
selection, cap increase or extension occurs.

Frozen final collects25 natural episodes using20,496 actions/81,948 frames in
9.431s. Initial collects26 using9,712/38,810 in4.457s. Selection remains the first
three per lane; one/two excess episodes and partial tails remain in full replay
and video. All25 final episodes average632.0, distinct from the selected24 mean
605.208. Zero truncations/updates, all346 tensors unchanged in each arm.

All49,939 numeric reports, source/config/counters, finite checkpoints, complete
CPU replays/videos and three host guards pass; service exits successfully.
Curves, coverage and the full18/minimal6 comparison are independently reconciled
and viewed.138 CPU replay tests pass staged and applied before launch. Native
`bfb8c31` and runner`1ed07af` remain fixed; source at launch`e853818`.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-minimal-200k`, including
`curve/`, `coverage-review.json` and `action-vocabulary-comparison/`.
Review SHA256`d63361a46d6cd30f5c540014c6b45014b57bec4128d6554c89be72801e2de3aa`;
curve SHA256`979db7a72c88ec75de57f91848709968f22f8a9be1b8d5485d0481f7d17729f0`;
comparison SHA256`7563683311c2ce2cd737840943d22352af1181b6a7cf318523c9dfbf68fefdf8`.
Seed1009 is reused development data. No extension or broad action/seed sweep.

## Cost estimate for more experience at lower replay ratio

The full18 coefficient1 learner spends1,103.742s of1,184.523s in learning
(93.18%). Its online last50 score rises from294 at49,152 actions to677.5 at200k;
that does not predict further frozen gains. The saved coverage evidence above
still shows cubes never colored during an episode, without identifying a cause.

A CPU-only planning calculation considers fresh4M/RGB48/H15 with500k actions
and ratio16, retaining the full18 coefficient1 recipe otherwise. Expected
updates are62,470, versus49,939 in the completed200k/R32 test:2.5 times the
experience and1.251 times the updates. The existing completed R16 control
confirms the first-update248/count convention; a new run must verify it itself.

Scaling recorded learning cost by updates and acting/remaining overhead by
actions estimates1,582.620s (**26.38 minutes**), including two hypothetical saves
but excluding construction. That leaves217s against a proposed30-minute cap
which would include construction. This is a cost estimate, **not a measured
speedup, learning prediction or selected allocation**. Trajectories and unit
costs can change. The calculation performs no GPU work or training extension.
If selected later, experience and ratio change together; retain own initial/
fixed-final controls and the unchanged15k/90%-pyramid gate. The active Pong
confirmation and prepared Breakout candidate are unaffected.

Artifact:`experience-budget-projection.json` in the full18 coefficient1 run;
SHA256`c79bf10b5d53c07885687ba547689a2984f37d44371b1c287ca972a2209c2db2`.
Script:`/x/Temp/kindle-cheap-20261025/review_qbert_experience_budget.py`.
The receipt pins completed review/curve/coverage/R16-control identities and
retains the formula, stage costs, observed curve points and assumptions.

The [candidate controller and queue](../experiments/2026-10-26-cdp-qbert-experience-budget.md)
are subsequently prepared, with complete queue-difference checks and refusal
before GPU artifacts while unselected. Its R16 recovery checker passes against
an actual completed R16 log/checkpoint and rejects invalid counters. It remains
an option pending review of the active Pong confirmation; no GPU launch or
automatic queue follows from this preparation.

## Completed compact encoders do update

A bounded CPU checkpoint comparison checks the full18 Qbert200k control and
the Breakout4M/RGB64/H15/mixture.01/200k control, both using Auto precision and
49,939 updates. Each own-initial/final pair changes all55,808 elements across
the12 CNN parameter tensors (four kernels, biases and normalization weights).
Every corresponding final first/second optimizer moment is nonzero. The first
reward projection also changes all163,840 elements, with nonzero moments.

This rules out wholly unchanged CNN tensors in these two completed runs. It
does not establish useful visual features, correct derivative scale or a cause
of failed task gates. Relative RMS kernel changes are2.57–7.50% for Breakout
and2.69–7.94% for Qbert; those magnitudes do not establish an optimal learning
rate. Raw derivatives and counterfactual learning remain unmeasured.

The0.16-second analysis constructs no agent, performs no GPU work and leaves
all source checkpoint files unchanged. It retains complete per-tensor counts,
RMS values, shapes, review identities and source hashes in
`/x/Temp/kindle-cheap-20261025/compact-encoder-update-review.json`, SHA256
`8d98e37a8e7341e51fda839174f4972d7c6127b6c5e8eb32ccb38905cfa2dae1`.
Script:`review_compact_encoder_updates.py` in the same root. An initial CPU
attempt expected a top-level initial-checkpoint field absent from the older
Qbert review; using its existing evaluated-source checkpoint identity completes
the analysis. The correction receipt is retained; no experiment is repeated.

## More experience at R16 does not improve the result

The500k/R16 policy scores **719.792 versus initial180.208**, with0/24 first
pyramids. Mean first-pyramid coverage is14.125/21, below the200k/R32 control's
16.625/21; its score is also below809.375. Training completes in **26m53s** within
the original30-minute cap. More experience at lower replay ratio fails to
improve this development recipe. No extension or confirmation is selected.

Fresh1009/eval4000001009, full18/coefficient1/4M/RGB48/Auto/N8/B8/T16/H15,
3,578,225 parameters, full BPTT/microbatch8, replay100k, actor_unimix0 and no aid.
Native train_ratio32→16 and experience200k→500k change together:2.5× experience
and1.251× updates, not an isolated ratio effect. All346 initial tensors match
the full18 control. Native frames receive one GPU resize; sticky.25/repeat4,
zero no-op starts and100000-frame episode limit remain.

| Budget / ratio | Actions / emulator frames / updates | Training wall | Frozen final / own initial | First pyramids | Mean coverage |
|---|---:|---:|---:|---:|---:|
|200k /R32|200k /799,342 /49,939|19m45s|809.375 /180.208|0/24|16.625/21|
|500k /R16|500k /1,998,525 /62,470|26m53s|719.792 /180.208|0/24|14.125/21|

Recorded training time1,612.906818s,310.00 aggregate actions/s, first update248,
zero debt and1,001 natural training episodes. The final online last50 mean721.0
is separate from frozen719.792. Reviewed wall-time curves are similar over much
of their shared window. Collecting experience faster does not establish faster
useful learning, and neither budget passes the original15000/90%-pyramid gate.
The joint budget/ratio change and one reused seed limit broader conclusions.

Final/initial frozen evaluations use19,752/9,232 actions,78,972/36,888 emulator
frames and9.218/4.248s. Select the first three natural episodes per lane,24 each;
final collects26 with two excess episodes retained, initial24. Partial tails
remain in complete traces. Zero updates/truncations; all346 tensors unchanged
per arm. Final coverage median15, range6–19; no selected episode reaches21 cubes.
All62,470 strict reports,250k recovery, full config/source/counters, finite
checkpoints, full CPU replays/videos and three host guards pass. The service
exits successfully09:33:03 UTC, total wall27m36.119s including evaluations/audit.
The completed coverage report and comparison plots pass and are reviewed.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r16-500k`.
Learning review SHA256`c641ae25ce3e38aaf9bb69c5b0e2bd88e80c8d159fefcb6458bf8d647e4c5f19`;
curve`db89a4a471912fe69c7417334df687bebdeef95889768714f8aadf459c1ca427`;
coverage`f3225efe2f54c427313be05b37e9a724b8a34d23e2f2229c808de910af79085d`;
comparison`9ed385d96f93a153105f9050c23e8eaa2c08cd9ddf23554894d9681a4fb6445e`.
Source at launch`640f614`, native`bfb8c31`, runner`1ed07af`, replay`e853818`.
The next selected work is the bounded PongB16 learning/restore check; it does
not select a full Pong run or restart any completed Qbert training.

## Larger capacity at RGB48: bounded check

The existing-native12M/RGB48/B16 check passes16,384 actions/65,481 emulator frames
and2,002 updates in149.387s, versus93.813s for4M/B8's matching action prefix.
This larger recipe costs1.592× as much early wall time; it is not an isolated
capacity effect or a full-budget prediction. Construction is separate,2.472s
versus1.969s. B16 samples512,512 positions in2,002 updates; B8 samples516,480
in4,035. Capacity, initialization, cadence, centering and warmup per action differ.

Fresh1009/eval4000001009, Auto/N8/T16/H15/R32, full microbatch16, coefficient1,
full18 and no aid. Runtime confirms14,123,473 parameters and first updates
376:1,384:2,392:3. All2,002 reports, full config/source/counters, finite346 tensors
and both guards pass. CNN0's2,400 weights and reward layer0's655,360 weights all
change; that is not proof of correct gradient scale or useful representation.
Own actual initial is retained; no equality claim across model sizes.

Ordinary frozen restore runs1,024 actions/4,096 frames in0.802s with zero updates,
zero natural episodes and all346 tensors unchanged. No natural-episode competence
gate is tested. Service exits10:35:06 UTC,2m39.870s total wall, without retry,
cutoff or native change. Source at launch`47690dc`, native`bfb8c31`, runner`1ed07af`.
Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-12m-48-b16-check`;
review SHA256`924d6a6931f7f90580649b3f28eaf1043d5ebd4e7600108ac9073123c82bac47`;
initial metadata`463c6e54d23ab7d8212b5fcdf3cc8a0d8b84300b74e7a7a83781b9cc155c5aca`.

After reviewing the check, select a separate fresh
[200k-action capacity/batch learning screen](../experiments/2026-10-26-cdp-qbert-capacity-budget.md)
under a35-minute cap. Keep the original frozen15k/90%-pyramid gate and own actual
initial. The short validation cost is additional, and its weights are not resumed.

## Larger capacity at the fixed200k budget

The12M/RGB48/B16 policy scores **959.375 versus own initial165.625**, with0/24
first pyramids. The4M/B8 control scores809.375, but costs19m45s versus **32m19s**
for the larger recipe. Mean first-pyramid coverage is16.292/21 versus16.625/21.
The modest score increase costs63.70% more training time and does not improve
coverage or pass the original15000/90%-pyramid gate. No extension or confirmation
is selected for this recipe.

Fresh1009/eval4000001009,12M/RGB48/Auto/N8/B16/T16/H15/R32,14,123,473 parameters,
full BPTT/microbatch16, coefficient1, actor_unimix0, full18 and no aid. Native
frames receive one GPU resize; sticky.25/repeat4/noops0 remain. Capacity and batch
change together against4M/B8, including initialization, centering, cadence and
warmup per action. Both200k budgets sample nearly equal replay positions:
6,388,224 versus6,392,192, with24,954 versus49,939 updates. Each uses its own
actual initial; no matched-initial claim across sizes. Seeds are reused development
data, and the separate bounded check adds cost without supplying an independent seed.

Training completes **200,000 actions /799,491 emulator frames /24,954 updates**
in1,939.063095s,103.14 actions/s, with first update376 and zero debt. All350
training episodes are natural; final online last50 mean822.0 is separate from
frozen959.375. Reviewed curves show better learning per action after early
training, but slower online improvement over much of the shared wall-time window.

Final/initial frozen evaluations use28,472/9,232 actions,113,848/36,891 frames and
17.098/5.528s. Select the first three natural episodes per lane,24 per arm;
final collects28 with four excess episodes retained, initial24. Partial tails
remain in complete traces. Zero updates/truncations and346 unchanged tensors
per arm. Final coverage median16.5, range10–18; initial mean5.292/21.
All24,954 strict reports,100k recovery, full config/source/counters, finite
checkpoints, complete CPU replays/videos and three host guards pass. Service
exits11:15:13 UTC with33m15.306s total wall, within the original35-minute training
cap. Completed curves, coverage and comparison pass and are visually reviewed.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-12m-48-b16-r32-200k`.
Learning review SHA256`9f5a0fc158fe8601fa475f4b949b658ec8cc5ae90989af8c215aa8849e03086e`;
curve`ee9bc3853703e61e57f623a01bf53d1ecaa95402216a06607a87f9a44422ecc9`;
coverage`564bb72d4b5d09c12ae5d08c44f5d148a16452e17929c4f0d34b3dff7a3b186a`;
comparison`15141dfdbe8a98498ad27678396a1c74f7e967a32fb776681d77d411be950796`.
Source at launch`be6c942`, native`bfb8c31`, runner`1ed07af`, replay`e853818`.
After review, select the prepared PongB16 doubled-rate test at its fixed500k
budget; no further Qbert allocation follows from this result.

The larger policy also leaves cubes untouched in every selected frozen episode.
An additional saved-action CPU replay reproduces every original transition,
episode outcome and partial tail. Its mean ever-lit count equals maximum
simultaneous coverage,16.292/21, and0/24 episodes ever light all21 cubes. Two
observer addresses (32 and40) remain untouched in all24 episodes, and42 in23/24;
no new spatial labels or causal explanation are inferred from those addresses.
Color reversions occur in all24 episodes, but losing a previously complete
pyramid does not explain the failure. The4M control similarly never lights all21,
with mean ever-lit16.667 versus simultaneous16.625. This distinguishes incomplete
coverage from full coverage followed by loss; it does not diagnose policy,
representation or exploration as the cause. RAM remains strictly post-hoc.
The15.156s CPU analysis creates no agent, policy decisions or learner updates.
Tile-review SHA256`ecbcbea37581c6439b0076271c9842ba52a38eaefed326d35d63f9e186291a3a`;
artifact:`qbert-12m-48-b16-r32-200k/tile-coverage-review.json` under the same root.

## Persistent random-action controls

Two declared CPU controls compare uniform random full18 actions held for1 versus
16 agent steps. These are reference behaviors, not Kindle learning or a training
aid. Both use eight independent environments, the reused evaluation seeds
4000001009 plus the standard lane offsets, sticky.25/repeat4/noops0/native frames,
24 selected natural episodes and a60k-action/120s cap per arm. Actions depend
only on the existing persistent-control RNG and fixed hold length; pixels,
rewards and RAM never determine actions. RAM measures post-hoc cube coverage.

| Hold actions | Actions / emulator frames | CPU wall | Natural score | Mean maximum cubes | Mean ever-lit cubes | First pyramids |
|---|---:|---:|---:|---:|---:|---:|
|1|9,136 /36,516|4.225s|131.250|3.917/21|4.000/21|0/24|
|16|12,544 /50,148|5.300s|207.292|7.042/21|7.167/21|0/24|

Both collect exactly24 natural episodes, retain partial tails and have no
truncations. Held16 episodes last453.167 actions on average versus331.708 for
held1, so their experience per natural episode differs. They use the same RNG
seed rule, but different draw cadence and trajectories. Neither ever lights all21
or approaches the learned policy's coverage. Held16 does reach the two observer
addresses never lit by the12M policy, in3/24 and2/24 episodes respectively;
held1 reaches them in0/24 and1/24. These are observer addresses, not new spatial
labels or a causal diagnosis.

A second pass reproduces every action/reward/boundary/frame count, episode result
and partial tail for each control, in4.217/5.301s. No GPU work, agent construction
or learning occurs. This supports a bounded hypothesis about coherent exploration;
it cannot predict whether5% held-action blocks help a learned policy. The
[training-aid candidate](../experiments/2026-10-26-cdp-qbert-persistent-aid.md)
was selected after the completed Pong rate comparison. Its subsequent frozen
score678.125 and coverage15.125/21 are below the unaided control; see the completed
training result below. The random-control benefit did not transfer to this aid.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-persistent-random-reference`;
review SHA256`b916cc639c11d38bd5a7f0eafa9ef4ad0569d215a21e2fbeb91891fc1c4f92e1`;
declaration`4f01f70b87669577fc19833e175988aa59c607bb1b4facb17782b48562c0fc2f`.
Script:`qbert_persistent_reference.py`; traces and complete episode records stay
outside Git. No reference success or five-game reliability is claimed.

## Persistent training aid does not improve frozen coverage

The5% held-action training aid scores **678.125 versus809.375** for the unaided
control, both0/24 first pyramids. Mean frozen cube coverage drops16.625→15.125/21.
Own initial remains180.208/5.833 cubes. Training takes **20m17s versus19m45s**;
the aid does not improve useful learning at this fixed200k budget. Retain the
unaided recipe; no extension or seed confirmation is selected.

Fresh1009/eval4000001009, centered CDP4M/RGB48/Auto/N8/B8/T16/H15/R32,
3,578,225 parameters, full BPTT/microbatch8, coefficient1, actor_unimix0, full18,
sticky.25/repeat4/noops0 and original rates. Native config, all346 initial tensors
and the complete semantic initial evaluation trace match the control. Only the
training wrapper adds5% persistent-uniform16-action blocks. The complete ledger
verifies10,039 overrides,5.0195% of actions; frozen evaluation has no aid.
Both runs use49,939 updates and6,392,192 sampled positions. Reused development
seeds do not supply independent confirmation.

Training completes200,000 actions/799,371 emulator frames in1,217.328262s,
with first248, zero debt and408 natural episodes. Construction2.079s is separate.
Final online last50 mean672.0 is close to the control's677.5, while fixed frozen
score and coverage are worse. Reviewed action/time curves show alternating leads,
not a sustained improvement. The CPU random-control coverage result did not
transfer into a benefit from a sparse training aid.

Final/initial frozen evaluation uses13,496/9,232 actions,53,944/36,888 emulator
frames and6.216/4.260s. Each collects and selects24 natural episodes with no
excess; partial tails remain. Zero updates/truncations and346 unchanged tensors
per arm. Final coverage median15, range13–17. All49,939 strict reports,100k
recovery, full config/source/counters, complete override ledger, CPU replays/videos
and three guards pass. Service exits12:24:16 UTC,20m56.234s total wall; completed
curves, coverage and comparison pass and are visually reviewed. Original15k/
90%-pyramid gate remains unmet.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-aided-200k`.
Learning review SHA256`9b0971ba1acd65196f147b112e072dbbd03fdc83096eb38d61d3610cb242e005`;
curve`de5da4b1305a0670ab6fe090d366a4389754c26b2fb67c0167077906e0516fdb`;
coverage`ed24bd5b317c6e74bb68a3b7cc65bdf10fd4fe3f2dcfb2363a345ff5de0e8665`;
comparison`9a81593fb54b680ed8e29bef01ed96fa32f767398e327898c559d41b33b1364d`.
Source at launch`1c31c4c`, native`bfb8c31`, CLI`22bd61b`, replay`e853818`.
After review, select the bounded PongH5 test, preserving B8's original update
cadence and rates while shortening imagined actor/critic rollouts.

## Reward timing and a longer imagination candidate

A 1.54-second CPU analysis of the completed Qbert and four-action Breakout logs
measures observed reward timing and projects rollout cost. Qbert has 91,787
origins after aggregate100k actions in completed natural episodes, each with
30 future actions available. On these identical origins, positive reward occurs
within the next5/15/30 actions at13.84%/30.81%/45.43%. Current-action reward and
unfinished tails are excluded. Overlapping origins are correlated; these are
recorded on-policy trajectories, not independent trials or counterfactual rewards.

Native H15 already bootstraps the critic beyond15, with discount horizon333 and
lambda.95. The timing result motivates a bounded H30 hypothesis without claiming
that H15 cannot learn delayed rewards. Longer learned rollouts may accumulate
error. Adding one copy of the recorded imagination and behavior-training cost
to measured H15 training time projects1,648.550s (27m29s) for200k/H30. Graph,
trajectory and unit costs can change; this is not a speed or learning result.

The [H30 candidate](../experiments/2026-10-26-cdp-qbert-long-imagination.md) keeps
4M/RGB48/B8/T16/R32, original rates, bonus1/full18/no aid,200k actions and49,939
updates, with a35-minute cap. It was prepared while Pong B16/R40 trained, then
selected after reviewing that run's interruption and saved-policy assessment.
The completed result follows below. The original15k/90%-first-pyramid gate
remains. The timing analysis itself performs no GPU work or controller sweep.

Analysis:`/x/Temp/kindle-cheap-20261025/longer-imagination-review.json`, SHA256
`c83ffa66d39124883d67bf5721ebcc1bf19976d1f8c109eb5fbde03d4f2df5e6`;
script`review_longer_imagination.py`, SHA256
`5a6c5ef66629de16dde7077d8ced770f601a1fb4f7ecedf597be04a23b3a3958`.
The script checks complete source-log/review hashes, action/update counters and
natural-episode counts. Breakout timing is descriptive context, not a second
selected candidate. Actor, learner and GPU work remain zero.

## Longer imagination costs more and lowers frozen coverage

H30 scores **582.292 versus809.375** for H15, with mean maximum first-pyramid
coverage **9.583 versus16.625 of21**. Both have0/24 first pyramids. H30 takes
**26m43s versus19m45s**,35.34% more training wall time at the same200k actions
and49,939 updates. Retain H15; no extension or seed confirmation is selected.
The complete recipe remains unsuccessful under the original15k/90% gate.

| Fixed final / actual initial | H15 control | H30 candidate |
| --- | ---: | ---: |
| Frozen mean score |809.375 /180.208|582.292 /180.208|
| Mean maximum cubes lit, of21 |16.625 /5.833|9.583 /5.833|
| First-pyramid successes |0/24 /0/24|0/24 /0/24|
| Training seconds |1184.523|1603.180|
| Aggregate actions / emulator frames |200000 /799342|200000 /799441|

Fresh1009/eval4000001009, centered CDP4M/RGB48/Auto/N8/B8/T16/R32,
3,578,225 parameters, full BPTT/microbatch8, replay100k, coefficient1, actor
mixture0, original rates/full18/sticky.25/repeat4/noops0 and no action aid.
Native max-pooled frames receive one GPU resize. Only native imagination15→30
changes; discount horizon333 and lambda.95 stay fixed. Both sample6,392,192
replay positions, while imagined actor rows double95,882,880→191,765,760.
This is not matched compute. Same GPU/Mesa/native binary, but kernel7.0.0-34
versus7.0.0-38 and a changed AMD module confound isolated attribution. All346
initial tensors and complete semantic initial evaluation traces match.

H30 completes389 natural training episodes, first update248 and zero debt.
Construction2.190s is separate from1603.180s training. Final online last50
mean654.5 is close to H15's677.5; the action/time curves show no sustained
advantage and frozen coverage is much lower. Recorded reward timing did not
translate into a benefit from longer imagined rollouts in this test; no model-
error or gradient mechanism is established.

Fixed final/own-initial evaluation uses18,104/9,232 aggregate actions,
72,388/36,888 emulator frames and7.319/3.747s. Each collects and selects24
natural episodes, with no excess, zero updates/truncations and retained partial
tails. Final coverage median9.5, range5–14. All49,939 numeric reports, full
config/source/counters, finite checkpoints,346 unchanged frozen tensors per arm,
complete CPU replays/videos and three host guards pass. Service exits16:26:25
UTC,27m18.195s total wall time. Curves, coverage and completed comparison pass
and are visually reviewed. Reused development seeds do not establish reliability.

**Protocol deviation:** the declaration required a100k recovery, but the pinned
queue used`--checkpoint-every 200000`; no100k checkpoint was written. This was
recorded at16:16 UTC, before the outcome. The original recovery watcher exits1
when it sees the200k checkpoint; the original postprocessor exits1 on the absent
100k review. Both failures remain. A separate CPU completion review verifies
the successful final service/audits and unchanged original artifacts, then runs
the existing curve, coverage and comparison scripts. It explicitly records
`protocol_complete=false`. This does not repair the missing milestone. No GPU
stage repeats, cap increase, resumed learning or earlier-checkpoint selection.

[Declaration and deviation](../experiments/2026-10-26-cdp-qbert-long-imagination.md).
Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-h30-r32-200k`.
Learning review SHA256`bb75dc79d593f4ac3dcf6da74f983edb212f9c5ce513ce1bccadd247a89db592`;
curve`e06383ee68998a290adc36358a7c80d9ee2983fbe459e25d57fa53bd1b084f0c`;
coverage`a02d3de03631469a9dee0880527b5a2843ea05cc9c337025f24567834a4f62f1`;
comparison`2a59a1c630cf619aa282ca2557482c5bde0f129f46f57dae5fd125ad04494262`;
completion with deviation`b8bf2db1c22919295d368c6ef00619ef7dfc1688109bf23e6e0112c2fe55357d`.
Launch source`c1e00c2`, native`bfb8c31`, CLI`22bd61b`, replay`e853818`.
No active or selected next GPU allocation after this review.

## RGB64 compact option: CPU preparation only

After the completed H30, capacity, bonus, action-vocabulary and random-aid
results, prepare a different compact option:4M/RGB64 with the original H15,
full18 actions, coefficient1/actor0, Auto/N8/B8/T16/R32 and original rates.
It was prepared unselected while Breakout ran. After reviewing Breakout's
completed11.208/0of24 result and corrected CPU comparison, **select this fixed
200k test**. It ran from18:16:53 UTC and is now complete; the result follows
below. Breakout receives no extension or confirmation.

The saved native/RGB48/RGB64 scene at action256 retains a visible board at both
sizes; it does not show that RGB48 caused the missing-cube problem. Testing64
keeps the intermediate4M preset but changes CNN/target geometry, parameters and
initialization:3,578,225→4,131,729 trainable parameters. Keep own actual-initial
controls; do not claim matched initial weights versus RGB48 or an isolated
resolution effect. The old/new kernel/module difference also remains.

The candidate's complete native config equals the already completed full18
Breakout4M64 configuration. That run passed200k/49,939 updates and frozen checks
in22m30s; Qbert48 took19m45s. This is existing execution evidence and cross-game
cost context, not a Qbert timing prediction or learning result. No new native
build, GPU qualification, agent or game collection is performed for preparation.
Native max-pooled frames still receive one GPU resize.

The selected run keeps fresh1009/eval4000001009,200k/49,939 updates,
6,392,192 sampled positions,95,882,880 imagined actor rows, first248, zero debt
and30-minute training cap. Retain100k recovery and fixed200k final, fixed
final/own initial24 natural episodes,600k-action/300-second evaluation caps,
zero updates/truncations, excess/tails, full346-tensor/replay/guard/coverage
checks and the original15k/90%-first-pyramid gate. No aid, teacher, RAM policy,
reward shaping, checkpoint selection, cap increase, extension or automatic run.

CPU preparation passes the completed-control reader, wrong-geometry refusal,
unselected-launch refusal, manifest pins, native config identity and the actual
checkpoint schedule. The inherited old Qbert queue saved every200k; the prepared
queue explicitly saves every100k. An inherited H30 actor-row count and unrelated
predecessor metadata are corrected before validation; the earlier preparation
is retained. Active Breakout files remain unchanged.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-64-r32-200k`.
Feasibility SHA256`841f5a05a2ccd58975e7343930e05ea3fa0d6157f203e87132ae70870eb8cfa0`;
CPU preparation`88e1990c91b8e28a2a022dddf50ba247e316658cda771b08f9f3e9f876feac70`;
unselected declaration`bdebd93a10bf72db4359a1a6f7827b3012d1afe80a47556095f2cb6dddbcc0f1`.
Prepared source`3e97df0`; controller`run_qbert_rgb64_200k.py`, SHA256
`39f76bc6c8e4524e4fb24f60f93b079a32152d941ea4661bb4a1e99a9fb0540e`;
queue`89d76a622f02d362d8623fe548846626858651cd362a86844f421871d0cb7ef2`.


The selected run starts18:16:53 UTC/source`093e5c6`, owner
`kindle-cheap-qbert-rgb64-200k-20261026`, invocation
`6794e50f12c74f43992c429ce99b8169`. Startup verifies the full config differs from
Qbert48 only in native geometry,4,131,729 parameters, all346 finite initial
tensors matching the completed full18/4M64 native shape, fresh counters,
first248:1/256:2/256:3 updates and native frames/one GPU64 resize. This is not
matched initialization versus Qbert48 or evidence of useful Qbert learning.
100k recovery, fixed evaluations, replay/video, curves/coverage and comparison
monitoring are attached. Selected declaration SHA256
`11938b30a7c2d36fdd65fcc4756f8b35fda7e0a8e8c55c1e413ebfa6b680d7e3`;
selection review`73e83853d20880adbbb726fa166507de4efd4ca8c10bb9e7bdb3d1fa7a7ddda9`;
startup`4fe8c88269196ee1c5a94754978d1f10c4ab008135916804fc20a57661570224`.
Selection uses Breakout's completed CPU-repair receipt; its original failed
postprocessor and unchanged GPU artifacts remain preserved. Native`bfb8c31`,
runner`fbdaee3`, replay`5043b7a`; no cap increase or automatic next allocation.

The100k recovery passes24,939 updates/all346 finite tensors. The stable
104,160-action prefix has25,977 finite reports;102,400 progress is683.034s with
25,539 updates and zero debt. Recovery watcher exits successfully; learning
continues to the fixed200k endpoint. Recovery review SHA256`e3c0d856e7556160b6c66f6408e1cca6d979b6cfdc4d59df73703169ce48769a`.
At this recovery, frozen quality was pending; the completed result follows.


## RGB64 costs more and lowers frozen quality

The fixed200k RGB64 test completes in **22m20s**,13.14% more training time than
RGB48's19m45s. Frozen score falls **809.375→718.750**, and mean maximum initial
pyramid coverage falls **16.625→15.083 of21 cubes**. Both have0/24 first pyramids
and miss the original15k/90% gate. Retain RGB48 for this compact recipe; no
extension or confirmation is selected. This reused seed does not isolate image
resolution: parameter shapes, initialization and the host kernel/module differ.

Fresh1009/eval4000001009 keeps4M/Auto/N8/B8/T16/H15/R32, full18, coefficient1/
actor0, original rates and no aid. Actual4,131,729 parameters. Training completes
200,000 actions /799,444 emulator frames /49,939 updates in1,340.138s, with first
update248,6,392,192 sampled positions,364 natural episodes and zero debt.
Construction1.975s is separate. Online last50 mean842 exceeds RGB48's677.5;
the reviewed curves' online advantage does not survive fixed-final evaluation.

Frozen final/own initial score718.750/180.208, coverage15.083/5.833, both0/24
pyramids. Final coverage range11–18, median16. Evaluations collect18,344/9,232
actions and73,336/36,891 frames in7.572/3.825s. Exactly24 natural episodes per
arm, no excess, all partial tails retained, zero updates/truncations. All346
frozen tensors remain unchanged. The initial tensors match the already completed
full18/4M64 native geometry; no matched-weight claim versus QbertRGB48.
Every49,939 numeric report, full config/source/counters,100k recovery, CPU
replay/video, three guards, curves/coverage and comparison pass. Service exits
successfully18:39:49 UTC,22m55.905s total. No retries, cutoff or GPU repetition.

Artifacts remain in `qbert-4m-64-r32-200k` under the runtime directory above.
Learning review SHA256`ff2f9cf1f2d009e82937e6cafd461f401a93168e01ce3e7d7f7b1785270fb126`;
curve`f4d56a3582b5081417670779591d95a748816c08e7ad34ce59ada564e3d6d9a4`;
comparison`b1636a1fd64d7567f2dd938a03a9283a168005a211cc4f75408fd0b168f92da6`;
completion`83b45cdaf2af61071e3fee08c0ec5a3037ed1452cd41f9a0e6a48b923d7d600d`.
Launch source`093e5c6`, native`bfb8c31`, runner`fbdaee3`, replay`5043b7a`.
After review, select the separately prepared frozen-only Pong greedy assessment;
this adds no new Qbert allocation or training extension.

## Pixel-only CPU rule clears the first pyramid

A fixed hand-coded rule clears the first pyramid in **24/24 natural episodes**,
with mean score **3764.583** and mean maximum coverage **21/21**. It still misses
the original mean15000/90%-pyramid gate. This supplies a concrete candidate for
training guidance; it is not learned Kindle competence or evidence of transfer
to an unaided policy. No training aid or GPU allocation is selected here.

The rule receives only native210×160 max-pooled RGB. It recognizes visible
blue/yellow cube roofs and the orange player, remembers occluded roof states,
and follows a shortest board path toward an unlit cube. Single four-frame
direction pulses alternate with visual landing waits. It has no reward, RAM,
task-observer output or environment handle. The existing RAM task observer
measures outcomes separately. No enemy avoidance or learned model is added.

Fixed-action calibration and one development episode use seed9001. That episode
scores3650, clears the first pyramid at frame2129 and ends naturally after1092
actions/4367 frames. The policy stays unchanged for the subsequent assessment:
N8, eval4000001009 with the usual per-lane seeds, full18/sticky.25/repeat4/noops0,
first3 natural episodes per lane. These evaluation seeds are reused from the
compact learners, not held-out reliability evidence.

Collection takes29.622s for29,552 aggregate actions/118,160 emulator frames;
an independent fresh CPU rollout reproduces every action, RGB-frame hash,
reward, boundary, executed-frame count, episode and partial tail in29.587s.
There are exactly24 natural episodes, no excess or truncation, and seven
nonempty partial tails. Selected first-pyramid milestones span1619–2104 frames.
Both passes finish within the declared60k-action/120-second caps. No Kindle
agent, GPU work or learner updates occur. There is no video audit for this
reference; complete pixel hashes and trajectories are retained.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-pixel-reference-v1`, with policy
`qbert_pixel_policy_v1.py` and driver `run_qbert_pixel_reference_v1.py` in its
parent. Development artifacts remain in `qbert-pixel-guidance-dev`.
Review SHA256`6956fd1329769fb1c338e63bb436d833a4cb1b62047cf89cebd6e600d96f4ff5`;
declaration`b8815003ccd60a196d088385ab22c035b7aee47201be135edea30106b6dbbb8b`;
trace`e2e8ae5c225256df15eeeab0075e128dfb4a71b81d7616579cb0dc2954d80c5c`;
policy`d8e47b4b0815fbcdbf2ff19d2e06fb06ca3e4a37e0033310585113d3892a4f59`;
driver`65a47fb30ce0ea669f3eeb5fc8cc174dcb443a932420545b9e602e04152aeb2e`.
ROM/emulator/wrapper pins and fixed cohort are in the declaration. Active Pong
learning remains unchanged; native guidance integration and unaided transfer are untested.

The optional episode-guidance adapter is now staged outside the active source
and runtime.420 CPU tests pass. Its formatted policy reproduces all29,552
reference actions, pixel hashes and outcomes exactly in29.391s. A separate
real-Atari integration was originally labelled a CPU stub, but the later
isolation audit below corrects that claim: lazy imports replaced the stub with
native inference on the integrated AMD GPU. It completes16,384 actions/40 natural
episodes with3,032 overrides, collection15.969s and complete CPU replay7.268s,
with zero learner updates. It is not CPU-only collection or a trained policy.
The5% episode probability gives18.506% of actions in that trajectory; this is
not a prediction for the learner. Original artifacts and the correction remain.

The adapter draws independently per lane at episode start, preserves selected
episodes until their natural boundary or timeout, and resets only that lane's
pixel-rule state. Native action overrides retain normal observation/replay/
learning cadence. CLI rejects frozen use, restore overrides, other games,
non-native pixels, reduced action sets and combined random aid. Ordinary
accounting checks the episode schedule and counts; a separate complete CPU
environment replay verifies the policy's actions and input-frame hashes.

Preparation: `qbert-pixel-guidance-adapter/validation.json`, SHA256
`4cb8edabc23eb5e736ac928bc19a9930af6a4c054ed5406c4061b7c04f1cd1fc`.
The separate `runtime-qbert-guidance` Python overlay uses the unchanged native
binary and preserves the original runtime. A bounded4096-action/963-update
native check plus1024 frozen actions is prepared **unselected**, with300s/60s
caps and exact4M48/B8/T16/H15/R32/coefficient1 control config. It retains actual
initial weights,346-tensor comparisons, finite reports, both guards and full
guidance replay. Active Pong must finish and be reviewed before applying the
staged source or selecting GPU work. No full learning allocation is selected.


After reviewing completed Pong actor/critic learning (−0.542/10of24 wins,
all audits/comparison passed), select this bounded native check. Apply the exact
420-test source, with application receipt SHA256
`740d05076346e1dbc058a15b3c2c8c66b55b9e6f3ce8d2d550da6b0375be7828`.
The original runtime and native binary remain unchanged. The check starts
19:39:44 UTC and completes as recorded below.
No evidence of native guidance execution or unaided learning is claimed yet.

A separate fresh200k/49,939-update follow-up is prepared unselected with the same
native config,100k recovery,25-minute cap and fixed final/own initial24 natural
episodes with aid disabled. Its plot reader accepts the completed control,
rejects it when labelled guided, and imports no ALE. Queue recovery cadence/caps
and unselected refusal pass. It cannot run before the bounded check completes
and is reviewed. Preparation SHA256`ab29eea77e0514d058926ffd13e7f97fb79f3ccaa914129c25a66a67a2559fd3`.


## Bounded native guidance passes after a CPU reader correction

Started19:39:44 UTC/source`137ed77`, the bounded check completes4096 actions/
16,375 frames/963 updates in22.541s and1024 frozen actions/4096 frames in0.452s.
There are512 verified guided actions (12.5% of this short run), from5% episode
selection. Initial346 tensors match the Qbert48 control; all346 frozen tensors
remain unchanged. All963 finite reports, exact config/counters, selected CNN/
reward weight changes and both guards pass. This verifies native execution and
restore, not useful unaided learning or a stable guidance action fraction.

The original service fails19:40:14 UTC in the subsequent CPU audit. The new
`replay_qbert_guidance.py` reused a frozen-oriented reader that rejects nonzero
intrinsic rewards during training; the earlier integration disabled the
disagreement bonus and had zero intrinsic reward, missing this case. Correct the new replay command to read completed
training logs through the vector accounting audit.421 CPU tests pass, including
that regression. The saved4096-action trajectory then reproduces every action,
reward, boundary and guided input-frame hash, with512 verified overrides.
Both GPU stages had already succeeded and are not repeated. Original controller,
declaration, source reader, service failure and journal remain retained.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-pixel-guidance-smoke`.
Corrected review SHA256`a4fc31556ac9234606eb237e48d8cf755be184fddb02e88f9f9af0e6fa751c22`;
guidance replay`7741b43414d34676609e867a04dba41e2221fbe27a318939fe7bc3b4e9023fd0`;
completion`b67964cb947ae2a7a4f241f871655480b779951d701efc327f0ce598438276b9`.
The original service retains its failed state; separate CPU completion records
all checks and the reader repair. No frozen reward-recipe restriction is relaxed.

## Selected fresh pixel-guided learning at200k actions

After reviewing the bounded result and CPU repair, select a fresh200k-action/
49,939-update comparison. No smoke experience is reused. Keep the exact unaided
control's4M/RGB48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original-rate
native config, fresh1009/eval4000001009 and first248/zero debt. Add only the
optional5% per-episode pixel guidance during training; measure actual action
share and retain complete training-action/pixel replay. RAM observer outputs
remain separate from policy inputs. Initial346 tensors must match the control.

Keep6,392,192 sampled positions/95,882,880 imagined actor rows,100k/24,939-update
recovery, fixed200k final,25-minute training cap,600k-action/300-second frozen
caps per arm, and fixed first3 natural episodes/lane with all excess/tails.
Both final and actual-initial policies disable guidance, with zero updates/
truncations and full346-tensor/config/source/counter/replay/video/guard/curve/
coverage comparison. Original mean15000/90%-first-pyramid gate stays unchanged.
Aided online returns are distinct from unaided frozen results. Reused development
seed and older-kernel control limit attribution and reliability claims.

Preparation after reader correction SHA256
`3d0c582057b64ae67719479d7e0aad996bdd2eb5a2d5f2e6b2b3665e859e0571`.
Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-pixel-aid-200k`;
owner`kindle-cheap-qbert-pixel-aid-200k-20261026`. No cap increase,
extension, checkpoint selection, automatic next allocation or unchanged retry.


The fresh run starts19:45:05 UTC/source`53ffc6b`, invocation
`d71773c5847b4575b7830a99e2b78823`. Startup passes exact native config versus
control,3,578,225 parameters, all346 matched initial tensors, fresh counters,
5% episode-guidance declaration and first248:1/256:2/256:3 finite updates.
100k recovery, fixed frozen evaluations, complete action/pixel/video replays,
curves/coverage and comparison monitoring are attached. Unaided transfer remains
pending. Selected declaration SHA256
`e451b20379c301a655f6583d656cad566ddf907cf32aa5945f7e2f55a09bc2bf`;
startup`6b5b1abc1fa768ccce000de5d91fd0341a35c898c50993266cd14ffac33aad12`.
Native`bfb8c31`, runner`137ed77`, frozen replay`5043b7a`, guidance replay`53ffc6b`.

The100k recovery passes24,939 updates and all346 finite tensors. A complete
104,416-action prefix contains26,041 finite reports;102,400 progress is624.690s
with25,539 updates and zero debt. The recovery watcher exits successfully;
fixed200k learning and unaided evaluation remain pending. Recovery review
SHA256`7e9ad619807969f8b5abf9fc2019f4a9e833a9f9f35e088d776a3c73f1532cf1`.


## Pixel-guided training lowers unaided frozen quality

The fresh 200k comparison finishes in **20m13s**, but unaided frozen score falls
**809.375→633.333**, and mean maximum first-pyramid coverage falls
**16.625→11.667/21**. Both have **0/24 pyramids**; own initial scores 180.208.
Retain the unaided recipe. No extension or seed confirmation is selected.
The online curve includes guidance and its higher returns do not establish
unaided transfer. The original mean15000/90%-pyramid gate remains unmet.

Training completes 200,000 actions/799,409 emulator frames/49,939 finite updates
in 1213.509s, first248, zero debt and 378 natural episodes. The fixed 5% episode
selection chooses 23 of 386 started episodes and overrides 23,428 actions,
**11.714%** of the action budget; episode probability is not action fraction.
Complete CPU replay of the saved training trajectory takes 69.236s and verifies
every guided action/input-frame hash, executed action, reward and boundary.
The native config is unchanged from the unaided control; all 346 actual-initial
tensors and semantic frozen-initial traces match. Native source `bfb8c31`, CLI
`137ed77`, guidance replay `53ffc6b`; launch source `53ffc6b`.

Guidance is disabled in both frozen arms. Final/initial collection uses
16,624/9,232 actions, 66,452/36,888 frames and 6.701/3.748s, selecting the fixed
first three natural episodes per lane. One excess final episode and all partial
tails are retained; no truncations or learning updates. All 346 tensors remain
unchanged. Full config/source/counter checks, 100k recovery, complete frozen
replays/videos, three guards, curves, coverage and comparison pass. Service
exits successfully at 20:07:01 UTC, 21m55.901s total. No GPU stage repeats.
One reused development seed and the control's older kernel limit attribution.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-pixel-aid-200k`.
Learning review SHA256 `5e71127f30573ad88dc9e3d7dcd6979fa4470ca7249d3fce99400d616be7c62e`;
coverage `82b617a90d7abae2acb80228b0367d661cfc61483bd9a2228708780ea35e3b74`;
guidance replay `f78ae6ab75e37768ecfb34fb0f88841d2700859f0a47a3c3f51a668cd7c71433`;
comparison `802ffc6bdf5a30146ed464c724a05fdeb79e53866e7cf83a992253cb65f9b656`;
completion `583b2d4cc4f65c36a7b64c3263753589d75d9b4981205021c7473c309e0b1461`.
After review, select only the separately prepared bounded Pong RGB32 check.


## Guidance exposure motivates an optional action limit

The completed 5% episode-guided run contains 23 guided natural episodes, all
clearing the first pyramid, and 355 unguided episodes, none clearing it. Guided
mean return is3709.783; unguided mean364.296. The last50 unguided episodes average
605.0, below the unaided control's677.5. The displayed aided last50 mean786.5
includes three teacher successes; it is not the learner's unaided performance.
These are post-hoc training groups with different trajectories, not frozen or
independent policy comparisons.

Of23,428 overrides,11,672 occur through the first-pyramid milestone and **11,756
are NOOP after it**, continuing until death. This identifies wasted guidance,
not the cause of the frozen regression. The latest observed first pyramid is
at2439 emulator frames. A640-action limit (2560 frames) would preserve all23
recorded successful prefixes, cap their overrides at14,720 and reduce their
post-milestone overrides to3048. Later learner-controlled trajectories are
unmeasured. The limit is selected from existing development evidence, not a
held-out evaluation; no RAM/task signal enters the policy or its release rule.
Exposure review SHA256 `a6a98db77593e95431ffaa611238cd0db6c5b14b0511022452d9862a6531827a`.

An optional fresh-only `--qbert-guidance-max-actions` is staged outside the
active source/runtime. The whole-episode default is preserved; the new bounded
kind releases each selected lane after the declared number of actions, without
a new selection draw, and resets the allowance only at that lane's next episode.
Released frames are not read by the teacher, native policy actions are used,
and ordinary observation/learning continues. Accounting rejects early/late
release and hashes after release; replay retains the episode's original guided
label even when the teacher stopped before the natural boundary.

**444 CPU tests pass.** A corrected real-Atari integration explicitly verifies
CPU-stub device/provenance/zero parameters before accepting its result. It
collects16,384 actions/48 natural episodes in6.382s with2560 overrides and zero
updates.137 actions after guidance release execute the stub's UP choice.
Complete pixel/action/reward/boundary replay passes in6.797s; this is adapter
validation, not learned or native execution evidence. The original PixelPolicy
and connected-components ASTs are unchanged; the patch applies cleanly.
Source and active runtime remain untouched during Pong; no next GPU allocation
is selected. Stage:`/x/Temp/kindle-cheap-20261025/qbert-guidance-cap-adapter`.
Validation SHA256 `1f2e843a50036b2367ebe4bc7f4c4a933507ddeda0a89941f39af404c501fdb0`;
verified CPU review `ae33a53613a04c77ee7c7cd0c2ac2e5511316cc1ff43c370f7a616059620f785`.

### Integration isolation correction

The first new integration attempt accidentally instantiates native inference on
AMD Ryzen5 9600X/RADV RAPHAEL_MENDOCINO, the integrated device. Accessing lazy
`kindle.default_config` overwrites a directly assigned `VectorAgent` stub. The
attempt completes16,384 actions/65,480 frames in13.690s with **zero updates**;
full CPU pixel/action replay passes, but the added released-action assertion
fails. Its artifacts are retained as a failed CPU-isolation attempt, not counted
as CPU-only validation or a selected GPU experiment. The RX7900XT Pong process
keeps the same PID/invocation; no restart, cap change, driver reset or native
rebuild occurs. This unplanned co-host activity is an additional timing caveat.

Inspection of the original whole-episode integration finds the same error:
its observed header names the integrated GPU and2,697,457 native parameters,
contradicting the review's hard-coded `native_agent_constructed=False`. Original
review/logs remain unchanged; `qbert-pixel-guidance-adapter/cpu-integration/
isolation-correction.json` records the correction, SHA256
`066d56c7acd906b04b8ff429656c42f079454eb63ddc54eb4c552534930630f7`.
The separately guarded RX7900XT training/frozen reviews remain valid. The second
new harness revision fails on a default-config argument before any environment;
the third uses a fixed adapter-facing API object and verifies observed stub
identity. Both failed scripts/logs remain beside the successful CPU result.
First-new failure review SHA256
`8a06389ba2a76dfb4b41430a8beafb6f9d3a886aa085bdf6a0f93f9c6d27ce17`.


## Prepared640-action guidance comparison, unselected

Prepare one fresh200k-action/49,939-update comparison with the same native
4M/RGB48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original-rate config,
1009/eval4000001009 and5% independent per-episode selection. Change only guidance
duration versus the whole-episode candidate: release each selected lane after
640 actions and let the native learner act for the rest of that episode.
Keep first248, zero debt,100k recovery and25-minute training cap. No experience
is reused. The image-based teacher itself is unchanged; it receives no RAM,
reward or task-observer input, and the release rule uses only action count.

Both frozen arms disable guidance and use the fixed first3 natural episodes per
lane,24 each, with600k-action/300-second caps, no updates/truncations, retained
excess/tails and all346 tensors unchanged. Keep the original mean15000/90%-first-
pyramid gate. Compare final/actual-initial scores, coverage and learning curves
against both unaided809.375 and whole-episode-guided633.333, checking all346
initial tensors against both. Full config/source/counter/report/replay/video/
guard audits and complete training-action/pixel replay remain required.

The queue pins1500-second training,100k checkpoint cadence and the640-action
flag; frozen commands omit all aid. Compilation, unselected-launch refusal and
both completed-control readers pass; labeling the whole-episode control as
capped is rejected. An unused `runtime-qbert-guidance-cap` overlay carries the
staged Python changes and a symlink to the unchanged native binary. Existing
runtime/source files retain their exact hashes. The controller additionally
requires the exact source application and its applied validation receipt.

This is **unselected** until current Pong learning, frozen evaluations and
postprocessing complete and are reviewed. A promising RGB32 Pong result may
instead receive focused seed confirmation. No GPU queue, source application,
cap increase, extension or automatic next allocation follows this preparation.
The640 limit was chosen from existing teacher traces; useful unaided transfer
and generalization remain unmeasured. Unaided control uses the older kernel;
whole-episode control uses the current host generation.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-pixel-cap640-200k`;
controller`run_qbert_pixel_cap640_200k.py`, planned owner
`kindle-cheap-qbert-pixel-cap640-200k-20261026`.
Preparation SHA256 `a51807fd102f6f295bf7ef1924d4c739116a333f7f1e448f1b1bc2e3f6677b8b`.


## Selected fresh capped-guidance learning

After completed Pong RGB32 audits/comparison show lower frozen quality, select
the prepared fresh200k/49,939-update capped-guidance test. Keep the exact declared
5% episode selection,640-action release, native4M48/B8/T16/H15/R32/coefficient1
recipe, original rates, full18 and1009/eval4000001009. Preserve25-minute cap,
100k recovery, fixed final/actual-initial24-natural unaided evaluations and the
original15000/90%-pyramid gate. No prior experience, extension or selected
checkpoint is reused. Unaided transfer remains unmeasured for this cap.

Apply the exact staged patch only after Pong's terminal success and completed
postprocessing. **444 applied CPU tests pass**; the native binary is unchanged.
The existing whole-episode kind/default remains supported. The unused prepared
`runtime-qbert-guidance-cap` overlay now supplies the matching Python modules.
Application receipt SHA256`e6939174c05527e2b152a00850b707b95552148fca7ee03a757c0bf4965e8da5`.
Its source hashes match staged validation byte-for-byte. The controller requires
this receipt, the completed Pong review and every declared source/runtime pin.
No additional GPU qualification sweep or native rebuild is needed for the
CPU action-limit schedule, already covered by real-Atari CPU integration and
existing native forced/unforced-action execution; this200k run tests learning.


The fresh run starts21:08:46 UTC/source`8f1f7ef`, invocation
`732153dabd2646c082fc55e08ca75cc7`. Startup verifies RX7900XT, exact native config,
3,578,225 parameters, fresh counters,5%/640-action guidance and first248:1/
256:2/256:3 finite updates. All346 actual-initial tensors match both unaided and
whole-episode controls. Progress,100k recovery and fixed-final postprocessing
are attached. Selected declaration SHA256
`0dd481e4695c31e3d5e35751ea5b275b8dd5fce638b91a08762c2d92ba992e61`;
startup`7986c895c34bf8a6157b36008e4e6f09a821d81604a16797ca95f144b2fc030d`.

A read-only complete22,552-action prefix verifies577 actions after guidance
release, with no forced action or teacher-frame hash. First release is aggregate
5128/lane5 after exactly640 guided actions; native choices begin17,16,13,3.
The first5120 transitions match the whole-episode control for actions, rewards,
stored rewards, boundaries, executed frames, overrides and teacher input hashes.
This is execution/schedule evidence; full pixel replay and unaided competence
remain pending. Prefix review SHA256
`6d03d0be51ebf0b6ac3700fcf5d3b2002de9ee8c5ed67dd49bfb947d4aed6111`.


## Capped guidance recovers some quality but remains below unaided training

The selected fresh run completes **200,000 aggregate actions / 799,403 emulator
frames / 49,939 updates in 1199.665 seconds (20m00s)**, within its25-minute cap.
Service `kindle-cheap-qbert-pixel-cap640-200k-20261026` exits successfully at
**21:30:21 UTC**, after21m35.325s including construction, frozen evaluation and
audits. Source remains`8f1f7ef`; native remains`bfb8c31`. There is no restart,
extension, cap increase or checkpoint selection.

| Fixed200k recipe | Training wall | Frozen final / own initial | First pyramids | Mean first-pyramid coverage |
|---|---:|---:|---:|---:|
| Unaided4M/RGB48 control |19m45s|809.375 /180.208|0/24|16.625/21|
|5% whole-episode pixel guidance |20m13s|633.333 /180.208|0/24|11.667/21|
|5% episode selection,640-action guidance cap |20m00s|750.000 /180.208|0/24|16.208/21|

All three share native4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18,
original rates and1009/eval4000001009. The cap improves this one run relative to
whole-episode guidance, but both score and coverage remain below unaided.
**Retain the unaided recipe. No extension, confirmation or next GPU allocation
is selected.** The original mean15000/90%-pyramid gate still fails. These reused
development seeds/cohorts do not establish reliability or isolated host effects;
the unaided control used the older kernel. The cap was chosen from prior teacher
traces, not held-out evidence.

The recovery checkpoint contains100,000 actions/24,939 updates. All49,939 reports
are finite, first update remains248, training debt is zero, and training finishes
389 natural episodes without truncation. Config/source/counter audits, all346
matched initial tensors against both controls, identical semantic initial traces,
all346 unchanged tensors per frozen arm, complete action/reward/boundary replays,
videos, host guards, coverage, learning curves and the three-arm comparison pass.
The final frozen arm collects14,712 actions/58,815 frames in5.956s and25 natural
episodes, selecting the first3 per lane (24 total). Its one excess episode and
all partial tails remain recorded. Own-initial evaluation collects9,232 actions/
36,888 frames in3.759s and exactly24 natural episodes. Both use zero updates and
no guidance; neither action/time cap is reached.

Full training pixel/action replay verifies15,360 overrides (**7.680%** of actions),
from24 selected episodes among397 started. The completed23 selected episodes
all clear the first pyramid before guidance release, with mean3900; none of366
completed unguided episodes does. Unguided mean return is420.697 overall and727.5
for its last50 episodes. A separate read-only analysis verifies3,561 native
policy actions after release, with no override or teacher input hash. Teacher-
assisted successes remain distinct from unaided learning; the frozen score is
750, despite a higher late online curve. Post-hoc task labels never enter policy
inputs. The final incomplete selected episode remains in the replay tails.

Evidence remains under
`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-pixel-cap640-200k`.
The viewed comparison plot is`guidance-cap-comparison/comparison.png` (SVG and
receipt retained alongside it). SHA256 receipts:

- Learning:`69222e771b73dfcdb11e892dd0150d62bb6e7b19fd476dad5fdeb718f2ea6ffa`.
- 100k recovery:`935b69de96f348dff9e897d597becd426147b8fa33608bd26f1bc97334670b4c`.
- Guidance replay:`71711e6d9b037e1c10d415747f33e77291b391002dcfd79d8727ee443768a703`.
- Coverage:`8b4621838b78afdcc51f4f697d8f7e36aeb314bd5f39b5205c22a1ec4bcf4e94`.
- Curves:`7c16526056fa7839550a880214219a6cf63a35bf12b2508104c715085c743efe`.
- Comparison:`2097fb6849dae6fbdfbebbf03d2c3554b8e4684f994da2a2dda493bfbba2e309`.
- Completion:`e1f9a995a09f105a8ff677270319d2843ae85638915eccddc3ea12f2ea17d268`.
- Exposure analysis:`d7702449eb1f99126f643f6a3c8878082aafe1fae3891dd1fc33065b3de8273c`.


## RGB48 reward-forecast diagnostic prepared, not selected

The compact Qbert H30 result is weaker than H15, and the existing cheap runs do
not measure multi-step prior reward quality. Prepare a small frozen diagnostic
before proposing another rollout-length change. It will condition forecasts on
saved actions from the retained unaided4M/RGB48 final evaluation, using the same
recorded pixels/actions/outcomes for both actual-initial and trained models.
This does not allocate GPU work; Pong behavior-rate learning remains active.

The existing `probe_policy_forecasts.py` has a hard-coded RGB64 acceptance check.
A **staged, unapplied** adapter admits the already-supported32/48/64 geometries
while preserving full source/checkpoint configuration equality. It also checks
metadata before constructing the native agent. All24 focused CPU tests pass,
including geometry/config refusals, reset handling, forecast alignment and
calibration/test separation. Native code and every active Pong pin are unchanged.

A CPU review verifies the source replay, both finite346-tensor checkpoints and
all15,376 recorded actions/61,462 emulator frames. The first natural episode per
lane supplies calibration; the next two supply test data. Selection remains the
original24 natural episodes, with excess/tails excluded from metrics. No new
learner, acting policy or game trajectory is created by this preparation.

| Forecast target | Calibration examples / positives | Test examples / positives | Test terminal events |
|---|---:|---:|---:|
| One-step reward |4829 /155|9328 /299|16|
|15-step endpoint reward, stride16 |298 /5|567 /17|0|
| Sum of rewards over those15-step windows |298 /102|567 /193|0|

Twenty-two long-horizon origins cross resets or lack enough remaining actions
and are excluded. The zero terminal support at15-step endpoints means that
sampling cannot assess15-step terminal prediction; one-step terminal evidence
is also limited. The proposed comparison should emphasize one-step positive
reward discrimination and15-step reward sums, retaining zero/calibration-mean/
persistence baselines and unrelated-action conditioning. It must not interpret
low aggregate error on mostly zero rewards as useful event prediction.

A concrete **unselected**128-action/one-stream frozen check has a60-second cap.
Its controller refuses unselected launch and requires the active Pong service
and postprocessing to finish before any separately selected start. It checks
native/source/device identity, deterministic forecast prefixes, all346 frozen
tensors, zero learning and the host guard. No full capture is queued. A later
full comparison, if selected after reviewing bounded cost/execution, would have
a300-second cap per model and retain the exact same recorded trajectories.

Scalar re-observation uses different categorical draws from the original vector
actor. Unrelated actions are model-conditioning controls, not observed alternative
outcomes. Calibration/test episodes are reused development evidence, not held-out
learner reliability. Forecast errors alone cannot identify the cause of failed
Qbert learning. No checkpoint training, extension or native rebuild is selected.

Staging and CPU evidence:
`/x/Temp/kindle-cheap-20261025/qbert-48-policy-forecasts-preparation`.
Review SHA256`7ecb7664fe7569286666ccd81e7f104e8c6c73ba5444022ace520f54d3954cfa`.
Bounded queue:
`/x/Temp/kindle-cheap-20261025/qbert-4m-48-policy-forecast-smoke`;
controller`run_qbert48_forecast_smoke.py`. Unselected declaration SHA256
`ef5aee5b7947cbdfd2a1c1d7259ad59d6ce885ac8c22fb45ba7baaad4a599455`.


## Guidance changes experience; supervised actor imitation is not implemented

A read-only source review verifies the relevant native files are unchanged from
runtime`bfb8c31`. The Python adapter forwards forced actions and then ordinary
observations/rewards/boundaries. Native action selection replaces the proposed
action and stores it as `pending_action`; replay retains that executed action
with its resulting observation, without a teacher/expert-label flag. The actor
loss consumes imagined actions weighted by discounted normalized advantages;
replay features feed the critic loss. Thus the completed guidance experiments
supply experience for world/critic learning, with policy transfer through
imagined returns. They do not perform supervised actor imitation.

The capped run's verified15,360 teacher actions contain14,447 NOOP and913 movement
actions. A constant NOOP predictor would match **94.06% of these saved labels**;
that is a dataset arithmetic result, not a game score or policy assessment.
Movement labels account for0.4565% of all200k collected actions. These counts
make aggregate label accuracy inadequate for judging a future imitation change.
They do not prove why the current unaided frozen result is weak.

If supervised guidance is later proposed, it needs an explicit trusted-label
path, separate from random exploration and ordinary actions. Replay's incoming
`previous_action` belongs to the preceding state: a state-at-t imitation target
must use the action stored at t+1, within the same lane and valid episode boundary.
A native auxiliary actor loss would be a new algorithm intervention requiring
alignment/gradient checks, unchanged default behavior and unaided frozen task
evaluation. At this review, no imitation implementation or allocation was
selected. The completed forecast check and later optional implementation are
recorded below; this source-path review describes the retained original runtime.

Source locations: [override path](../../kindle/src/dreamer/agent/vector.rs),
[replay frame semantics](../../kindle/src/dreamer/replay.rs),
[behavior targets](../../kindle/src/dreamer/agent.rs),
[behavior loss graph](../../kindle/src/dreamer/behavior.rs), and
[Python binding](../../python/src/vector.rs). The review constructs no agent,
changes no code and verifies every active Pong pin remains unchanged.
Evidence:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-pixel-cap640-200k/guidance-learning-path-review.json`;
SHA256`a2a4e096462169e4d5670e6f7ab067a0fdc934f1e12480bc7dfe79e430b898b7`.


## Selected bounded RGB48 reward-forecast check

Pong behavior-rate learning and all postprocessing now finish successfully;
frozen12.375 remains below B8 and the mean15 gate. After reviewing that result,
select only the previously prepared128-action/one-stream frozen Qbert check.
Use the pinned staged RGB48 helper and retained unaided final checkpoint on
its recorded final-policy trajectory, with60-second process/180-second service
caps, zero learning, all346 tensors unchanged and the existing host guard.
The prepared declaration is preserved before selection; exact checkpoint files
and the completed Pong review are added to its pins. No native rebuild or new
training trajectory is involved. A full initial/final forecast comparison
requires a separate cost/execution review after this check; it is not queued.


## RGB48 bounded success and selected fixed forecast comparison

The bounded check starts22:25:43 and exits successfully22:25:46 UTC. It replays
128 recorded actions in2.674757s total, including0.392923s collection, with zero
updates, all346 tensors unchanged, repeatable forecast prefixes, unchanged
belief/counters during forecast calls, finite predictions and a passing guard.
Review SHA256`e3f485033cc6e1e7351315f6fdbf57fe0f76e0cd936d33302912e27cf2f2c313`.
This validates execution and cost, not predictive quality. The exact staged
geometry helper is applied, with24 CPU tests passing in0.10s; native unchanged.

After reviewing that cost, select the **fixed final and actual-initial** models
on the same15,376 recorded actions/61,462 emulator frames per model. Each frozen
capture has a300-second cap, with720 seconds for the service. Zero updates,
all346 tensors unchanged, exact source/replay/checkpoint/runtime identities,
all8 stream pixel hashes and action/reward/boundary/split arrays must match.
Retain horizons1/15 withstride16 for the latter, first natural episode per lane
for baseline calibration and next two for test. Report zero/calibration-mean/
persistence and unrelated-action baselines, emphasizing sparse positive events
and15-step reward sums. The existing299 one-step test reward events and193
positive15-step windows support that assessment;16/0 terminal events do not
support a terminal-quality conclusion. No new trajectory or training allocation.

Controller`run_qbert48_forecasts.py` compiles and refuses unselected launch.
Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-4m-48-policy-forecasts`;
owner`kindle-cheap-qbert-48-forecasts-20261026`. The declaration pins source,
checkpoints, retained replay, completed smoke and exact applied tests. Forecast
errors remain descriptive, not a causal explanation for poor policy transfer;
no extension or next GPU allocation follows automatically.


## Compact Qbert learns reward forecasts despite weak task performance

The fixed comparison starts22:28:23 UTC/source`915e442` and exits successfully
at **22:30:05 UTC**, total service wall1m41.711s. The trained and actual-initial
captures replay the same15,376 aggregate actions/61,462 emulator frames each,
in50.329847s and49.267957s. Both perform **zero learning** and preserve all346
checkpoint tensors. All8 recorded stream pixel hashes and action/reward/boundary/
episode/split arrays match between models. All source/checkpoint/native/device
pins, forecast-prefix invariance, finite outputs, sample-support checks and
host guards pass. Neither process approaches its300-second cap.

The first natural episode per lane calibrates baselines; the next two provide
9328 one-step test examples (299 positive rewards) and567 valid15-step windows
(193 positive sums). The initial reward heads predict zero. Errors below are
mean absolute errors in reward points; lower is better. AUC measures ranking
of reward events; higher is better and.5 is chance/constant ranking.

| Test metric | Actual initial / zero | Trained, recorded actions | Trained, unrelated actions | Calibration mean | Persistence |
|---|---:|---:|---:|---:|---:|
| One-step reward MAE, all |1.431|1.190|1.149|2.609|2.836|
| One-step reward MAE, positives |44.649|18.398|19.178|43.391|45.819|
| One-step positive-reward AUC |.500|.969|.968|.500|.492|
|15-step reward-sum MAE, all |22.487|12.400|13.274|28.753|36.287|
|15-step reward-sum MAE, positives |66.062|31.089|34.724|46.431|79.404|
|15-step positive-sum AUC |.500|.899|.869|.500|.519|

After observing the resulting frame, trained posterior reward prediction has
positive-event MAE15.124 and AUC.979. This is a separate, easier measurement
than forecasting before the action. At the15-step endpoint there are only17
positive events: recorded-action AUC.956 versus unrelated.546; retain that
small support alongside the broader reward-sum comparison. The actual-action
sum forecast improves MAE but has worse RMSE than unrelated conditioning
(49.975 versus48.071), so it does not dominate every metric. One-step unrelated
actions also have slightly lower all-example error; action sensitivity is not
uniformly useful. Unrelated actions have no observed counterfactual outcomes.

The4M48 model learns useful reward information, including15-step sums, on this
retained final-policy cohort. That does not establish accurate alternative-action
ranking, exploration, actor optimization or the cause of failed first-pyramid
transfer. It provides no independent reason to enlarge the model or change
H15 again; retain the unaided4M48/H15 control. Guided-training results and the
absence of a supervised actor-imitation loss remain separate evidence. No new
training, unchanged extension or next GPU allocation is selected here.

There are only16 one-step terminals and zero15-step endpoint terminals. No
terminal-quality conclusion follows; the one-step learned terminal Brier score
.002020 is worse than the calibration-frequency baseline.001712 despite a high
ranking AUC. Scalar re-observation uses new categorical draws, not the original
vector actor's exact latent states. All trajectories and learner/evaluation seeds
are reused development evidence. The frozen task score remains809.375/0-of24
pyramids; this diagnostic changes no five-game gate or competence claim.

Artifacts remain in`qbert-4m-48-policy-forecasts` under the declared runtime root,
including both captures, comparison JSON/PNG/SVG, source pins and terminal journal.
Comparison SHA256`1421a339edeb1fdff27086e506f5de92862107515c090de19461fea7bdb0c057`;
completion`9a0107f304ea3dbf42dc9b88f2ccf9f47167fdf4ccdf8a2d78a43c94e7c777d9`.
The geometry helper passes24 applied CPU tests; native binary is unchanged.


## Optional native expert imitation: implementation and bounded validation

The retained compact model predicts rewards, but guided experience has not
improved its unaided task performance. Select an optional supervised actor term
on explicitly trusted replay actions as the next bounded learning intervention.
This adds no parameters and keeps the existing CDP world/imagined actor/critic.
No general backend optimization, pretrained encoder or Python learning is added.

`expert_imitation = {scale, action_weights}` is absent by default and omitted
from serialized default configs. When enabled, the shared native actor also
consumes detached replay features. The auxiliary cross-entropy is averaged over
all replay rows after applying explicit per-action weights, then multiplied by
scale. It does not normalize by the current minibatch's expert count; that would
cancel NOOP downweighting on batches containing only NOOP labels. The default
graph retains its original six outputs and parameter layout.

Only the explicit `expert_actions` vector API marks labels; ordinary overrides
and action masks remain untrusted. Targets pair state t with the executed action
stored at t+1 in the same lane. Last/reset crossings and sequence tails are
excluded; an action leading into a true terminal can still be supervised.
The Atari CLI exposes fresh-only `--guidance-imitation-scale` and optional
`--guidance-noop-weight`, requiring Qbert pixel guidance. Frozen restore keeps
the checkpoint config and supplies no expert labels. Logs retain an explicit
expert mask; the CPU audit rejects missing, forged or undeclared masks.

127 Rust CPU tests and strict library/test Clippy pass;80 GPU tests remain
ignored in that suite.477 relevant Python tests pass in1.11s. New checks cover
legacy/default config, weight validation, incoming/outgoing alignment, resets,
stream sampling, actor parameter reuse, graph compilation, CLI forwarding and
trusted-label logging. The first replay test incorrectly assumed sampling would
select each lane once; the corrected test checks tags across sampled identities
and confirms both lanes are exercised without imposing that false assumption.

Select only two bounded GPU checks under the existing host guard: analytic
weighted/masked cross-entropy gradients, and explicit vector expert labels
reaching one native replay actor update. Each has a120-second process cap;
service cap300 seconds. These are correctness checks, not game performance.
A separate release wheel will preserve the previous native runtime. A bounded
Atari learning/restore check follows only after reviewing this result; no full
training allocation, extension or seed campaign is selected.

Implementation/build/check artifacts:
`/x/Temp/kindle-expert-20261026`. Original runtime and all completed learner
checkpoints remain under`/x/Temp/kindle-cheap-20261025`.


Both bounded native checks pass; service exits22:44:40 UTC in4.947s. The analytic
check verifies all120×18 logit derivatives with zero/.05/1 target weights and
actor mixture0/.01. The vector check exercises invalid-before-mutation refusals,
ordinary versus trusted overrides, tag consumption and one native learner update
(3 expert rows, weight sum3, loss.5461493); actor parameters change. Both guards
pass. Review SHA256`a735cae7c76459a72657206f2530285b05301641e3de83e95f38677ddebc34b9`.

Release wheel/source`79c9a7f` is packaged separately at
`/x/Temp/kindle-expert-20261026/runtime`; native SHA256
`7c07d6702e37ca11aa20c1b35b48951f0ed01e0e1dfe72e187eac52253f8455e`.
The prior runtime remains unchanged.477 Python tests pass against the packaged
binding, and strict Python-binding Clippy passes. No additional native parameters.

After that review, select only a fresh4096-action/963-update Qbert smoke and
1024-action frozen restore:4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18,
1009/eval4000001009,5% guided episodes capped at640 actions, expert scale1,
NOOP weight.05 and other action weights1. Keep300s training/60s frozen caps,
600s service cap; first update248 and zero debt. Require346 matched initial and
unchanged frozen tensors, world/actor/critic weight changes, nonzero trusted
movement labels and auxiliary reports, complete pixel/action replay, declared
expert masks, config/source/counter checks and guards. This does not assess
natural-episode competence and its experience will not seed a full learner.
Owner`kindle-expert-qbert-smoke-20261026`; controller`run_qbert_expert1_smoke.py`.
Artifacts`qbert-4m-48-expert1-smoke` in the new root. No full allocation selected.


## Native expert-imitation smoke passes; task benefit remains unmeasured

The selected fresh check/source`0bb6939` completes4096 aggregate actions/16,378
emulator frames/963 updates in **22.662s**, first update248 and zero debt. Frozen
1024 actions/4096 frames take0.452s, with zero learning and zero natural episodes.
The service exits successfully **22:49:20 UTC**, total32.565s including setup,
frozen restore and CPU audits. No cap increase or GPU repeat occurs.

All963 reports and346 checkpoint tensors are finite. All346 actual-initial
tensors match the retained unaided4M48 control, and all346 trained tensors remain
unchanged after restore/frozen evaluation. The selected CNN/reward/actor/critic
weights change; parameter count remains3,578,225. Both guards, config/source/
counter checks, explicit trusted-label masks and full training pixel/action
replay pass. The teacher executes512 actions (12.5% of this short run):474 NOOP
and38 movement actions. The schedule still selects episodes at5%; actual action
share is measured separately.

611 of963 updates contain expert labels. Across sampled replay,13,785 labelled
rows carry total weight1524.300 and accumulated unscaled auxiliary loss33.4449.
These are repeated replay exposures, not additional environment actions. Zero-
label updates have zero auxiliary loss. No aggregate label accuracy or task
competence is claimed; frozen natural-episode assessment is still required.
The short timing is execution cost, not an isolated performance comparison.

A strengthened vector test disables the imagined policy loss entirely; the
expert loss still changes actor parameters (3 labelled rows, weight3, loss.5461493).
Its separately guarded service passes and exits22:51:05 UTC. The analytic test
already checks weighted/masked logit derivatives. This one-line test-only change
does not change the packaged release native. Review SHA256
`e67d9dea6938284ce2cd876caf4f17cceeaa95de899db199698ad413bfc98e2b`.

Smoke review SHA256`cd942380d147d839d01ec18457075a48909d72de13a1fc06612c7d10bdc3b537`;
completion`fcaecb6bd346117cebd7fc38374f8d6752e18018dee72ce68997ccfbf3631f45`.
The isolated wheel/runtime, validation executables and all evidence remain in
`/x/Temp/kindle-expert-20261026`. The2.7GiB disposable build cache is released
after packaging. Original native and learned checkpoints remain unchanged.

The next learning decision can use this qualified optional path in a fresh,
capped comparison against the completed640-action guidance and unaided controls.
Its benefit is not established by finite losses or a short execution check.
No full training allocation, unchanged extension or seed campaign is selected
by this smoke result. Never reuse the smoke's experience as a fresh learner.

## Expert imitation improves compact Qbert, but remains far below the gate

The separately selected fresh200k comparison completes **200,000 aggregate
actions /799,509 emulator frames /49,939 updates in20m11s** (1210.768s). The
fixed final policy, with guidance disabled, scores **1083.333 and clears2/24
first pyramids**, versus809.375/0 for unaided training and750.000/0 for capped
guidance without imitation. Mean first-pyramid coverage reaches18.625/21.
This is useful transfer on the reused development seed, not Qbert mastery or
evidence that the five-game goal is complete.

| Fresh200k recipe | Training | Frozen mean | First pyramids | Mean coverage |
| --- | ---: | ---: | ---: | ---: |
| Unaided4M48 | 19m45s | 809.375 | 0/24 | 16.625/21 |
| 5% episode guidance, cap640 | 20m00s | 750.000 | 0/24 | 16.208/21 |
| Same capped guidance + expert loss | 20m11s | 1083.333 | 2/24 | 18.625/21 |
| Actual initial, all three recipes | — | 180.208 | 0/24 | 5.833/21 |

All recipes use4M/RGB48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18,
original rates and fresh1009/eval4000001009. The candidate adds expert scale1,
NOOPweight.05 and other weights1, using the qualified native/CLI79c9a7f runtime.
The same346 initial tensors and initial semantic trajectories match both
controls. The new native build and older unaided-control kernel limit causal
attribution. Observed training time is0.93% above capped guidance and2.22% above
unaided, while mean score is33.85% above unaided; these are single-run recipe
comparisons, not isolated performance or seed-reliability estimates.

The original mean15000/90%-first-pyramid gate remains unchanged and fails.
The fixed final assessment runs18,808 actions/75,200 frames in7.605s;26 natural
episodes complete, with the declared first3 per lane selecting24 and retaining
the2 excess episodes and partial tails. Actual initial runs9232 actions/36,888
frames in3.754s and completes24 natural episodes. Both have zero updates and
zero truncations, and all346 tensors remain unchanged in each frozen arm.

All49,939 reports are finite; config/source/counter checks, explicit trusted
labels,100k/24,939-update recovery, all346 initial and frozen tensor checks,
initial semantic trajectories, full training and frozen pixel/action replays,
videos, host guards, curves and comparison pass. First update248 and zero
residual credit match the declaration. The main service exits successfully
**23:23:23 UTC**,21m48.267s including setup, evaluations and CPU audits. CPU
comparison completes23:23:46; the added exposure analysis completes23:24:02.
There is no GPU restart, cap increase or extension. The two CPU preparation/
startup-checker corrections remain recorded in the experiment record.

The5% episode selection rule chooses22 of338 started episodes and executes
**14,080 trusted actions (7.040%)**:13,247 NOOP and833 movements. All22 completed
guided episodes clear a first pyramid, mean3784.091. These returns include
teacher actions. Among308 unguided training episodes,7 clear a pyramid and the
mean is495.698; the last50 average762.000 with1 pyramid. The fixed frozen score
above remains the outcome measure. Full replay verifies every teacher action
from native RGB; RAM task observers remain post-hoc only.

24,936 of49,939 updates contain expert labels. Repeated replay samples contain
498,698 labelled rows with total target weight53,676.194; these are not unique
demonstrations. Across successive50k-action windows, weighted mean expert
cross-entropy is1.418,0.480,0.324,0.308. This shows fit to changing sampled
training targets, not held-out action accuracy or the cause of the frozen gain.
The trusted-action masks are absent after the640-action release; the native
policy supplies6831 recorded actions in those released episode suffixes.

Retain this candidate for a separately selected fresh-seed check before any
broader claim. Do not extend this learner or choose an earlier checkpoint.
No successor GPU allocation is selected or queued by this result.

Protocol: [expert-imitation comparison](../experiments/2026-10-26-cdp-qbert-expert-imitation.md).
Artifacts: `/x/Temp/kindle-expert-20261026/qbert-4m-48-r32-expert1-cap640-200k`.
Learning review SHA256
`3f1251fae80329393f4cdeac0725d5bdddd4a9f3c3feaae3edc94c68aa8dea79`;
comparison`7b4edd990733ac7754ec49a5c693d9af66058ba9344c5c2244da4e01776dab39`;
completion`bad2bcff8bd51ba954ee1bfefe457216e30ae3b0211fd3a97713e3ae9e00056b`;
exposure`240f1d5da7383aa1989db9f8472634672d7bcab07963b697e70578ce808de4c7`;
100k recovery`8b2c5a8311ab24e1c81916c34696de2a9042dae526b5733b12376c13531dd899`.

## Frozen expert policy leaves individual cubes untouched

A CPU-only replay of the completed1009 final policy reproduces every saved
episode and partial outcome, rewards, boundaries, frames and reset count. It
takes5.972s and constructs no agent. The observer includes the exact first-
pyramid completion frame, so both successful episodes retain all21 cubes.
This adds diagnostic detail to the existing frozen cohort; it is not another
policy evaluation or a new sample of learning reliability.

In all24 selected episodes, the number of cubes ever lit equals the maximum
simultaneously lit: mean18.625/21. Only the same2 successful episodes ever light
all21. Seventeen episodes peak at19 or20 and then last another804–2744 emulator
frames without improving that peak. Thus, for this cohort, missing first-pyramid
completion cannot be explained solely by lighting all cubes at different times
and losing coverage before they are simultaneously lit.

The cube tracked at RAM address42 is never lit in21/24 episodes; address32 is
never lit in14/24. The rest are each missed in at most4 episodes. These addresses
are post-hoc observer data and never reach the policy. Two decoded frames from
the verified stream0 video show the remaining bottom-corner tile tops still
blue: both corners near episode0's19-cube peak, and the bottom-right corner
near episode2's20-cube peak. The images are illustrative lossy video frames,
not exact native inputs or a complete RAM-to-position mapping.

This localizes a recurring incomplete-visitation pattern. It does not establish
whether perception, policy learning, planning, demonstration coverage or some
other mechanism causes it. The same CPU analysis subsequently completes after4013
without changing its learner, recipe or caps; the result follows below.

Artifacts:`expert-tile-coverage-review.json`, two`*-episode-near-peak.png` frames
and`expert-tile-visual-review.json` in the completed1009 trial directory.
Tile review SHA256`f9af1005d6acef474077a42b8eb83e09b7698d02790a6c9e4f664d6f09f67e67`;
visual review`da227a79f1853526e0e2f6a7703fd2e33bad3725d7b9ddb8ed4fbbccc9bb366b`.

## Pixel teacher can recover after a recorded learner prefix

A separate CPU-only feasibility check replays the completed1009 frozen policy's
first200 actions in each of its eight first episodes, then branches to the
**unchanged** native-RGB pixel teacher for at most640 actions per lane. All eight
original episodes are still live at the branch. Their initial partial coverage
is10–18 cubes, with no first pyramid cleared. The teacher receives only current
native max-pooled RGB, without RAM, rewards, task labels or environment handles.

All8 trajectories reach the first pyramid after176–291 teacher actions. The
declared640-action window still runs in full:1600 recorded-prefix actions plus
5120 teacher actions,26,880 emulator frames total. There are **zero natural
terminations and eight partial tails** at the cutoff. These are milestone
observations from hand-coded recovery, not8 eligible natural-episode successes,
not learned transfer and not a full Qbert gate pass.

The first pass takes6.732s and an independent fresh-environment replay takes
6.692s. Every action, pixel hash, reward, boundary and frame count matches.
The recorded learner prefixes also match their source log's rewards, boundaries
and frame counts; all ROM/ALE/wrapper/policy/source pins remain unchanged. No
Kindle agent or GPU is used, and the then-running4013 learner is unaffected.

This supports the feasibility of starting the existing teacher after the
learner has acted. It does not show that delayed guidance improves learning,
or that a fresh weak learner supplies the same distribution of prefix states.
The completed4013 result follows below. No delayed-guidance implementation or
GPU allocation is selected by this CPU check.

Artifacts:`/x/Temp/kindle-expert-20261026/qbert-delayed-teacher-prefix200`.
Review SHA256`a78ecf9b8d4bf2cde9aa6444872ec43051e160cc77cfee50207f63d1aeb6a275`.

## Expert imitation transfers on4013, with substantial seed variation

The fixed-recipe fresh4013 check completes **200,000 actions /799,532 emulator
frames /49,939 updates in20m07s** (1207.015s). With guidance disabled, its fixed
final policy scores **3153.125 and clears17/24 first pyramids**, versus its own
actual initial182.292/0. Mean coverage rises5.167→20.167/21. The same recipe on
1009 scored1083.333/2, so both learners show useful transfer at about20 minutes
each, with a large difference in outcome. Neither passes the unchanged
mean15000/90%-first-pyramid gate; the five-game goal remains unfinished.

| Fresh learner | Training | Frozen final / own initial | First pyramids final / initial | Mean coverage final / initial |
| --- | ---: | ---: | ---: | ---: |
| 1009 | 20m11s | 1083.333 /180.208 | 2/24 /0/24 | 18.625 /5.833 |
| 4013 | 20m07s | 3153.125 /182.292 | 17/24 /0/24 | 20.167 /5.167 |

Only learner/guidance/evaluation seeds and output paths change. Both use the
same native/CLI79c9a7f,4M/RGB48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/
full18/original rates,5% episode guidance capped640, expert scale1/NOOPweight.05.
The native config differs only in seed1009→4013. Seed4013 comes from the reused
development seed set, not a broad held-out test. There is no paired unaided
Qbert4013 training control; this checks repeatability of candidate learning,
not an isolated imitation effect on that seed.

All50 source/evidence pins remain unchanged. All49,939 reports and346 checkpoint
tensors are finite. The zero-experience initial matches all346 same-seed,
same-architecture reference tensors and differs from1009; nothing is restored
into training. First update248, zero debt,100k/24,939 recovery, config/source/
counters, explicit trusted labels, complete training/frozen pixel/action replays,
videos, host guards, curves, exposure analysis and two-seed comparison all pass.
No startup failure, GPU retry, extension or cap increase occurs in this life.

Final and own-initial assessments each complete exactly24 natural episodes
with zero updates/truncations; all346 tensors remain unchanged in each arm.
Final:16,608 actions/66,398 frames in6.716s. Initial:8728 actions/34,875 frames
in3.553s. Partial tails are retained. The main service exits successfully
**23:53:38 UTC**,21m40.264s total including setup, evaluations and CPU audits.
Postprocessing completes23:54:08 and the additional tile replay23:54:40.
All services and observers are terminal.

Guidance selects18 of325 started episodes and executes **11,520 trusted actions
(5.760%)**:10,855 NOOP and665 movements. All18 completed guided episodes clear
a pyramid, mean3894.444. Among299 unguided training episodes,95 clear a pyramid,
mean1500.167; the last50 average3484.500 with41 pyramids. Those online results
remain separate from the fixed frozen outcome. The different realized exposure
from1009's7.040% is a consequence of the fixed stochastic episode rule and
different trajectories, not a changed guidance probability.

21,735 updates contain expert labels; repeated replay contributes431,760 labelled
rows with total target weight45,420.645. Weighted cross-entropy over successive
50k-action windows is1.116,0.379,0.301,0.306, similar at the end to1009's0.308.
These changing training samples do not measure held-out action accuracy or
explain the large frozen-score difference.

The independent5.148s CPU tile replay reproduces every saved outcome. Mean
ever-lit coverage is20.208 versus simultaneous20.167; the same17 episodes ever
light all21. None of the7 failed episodes ever lights all21. Only3 episodes peak
at19–20, compared with17 on1009. Address32 is never lit in0/24 episodes and42 in
2/24, compared with14/24 and21/24 on1009. The earlier corner-omission pattern
therefore varies across learners. This does not diagnose its mechanism.

Retain the expert-imitation recipe and both learners as useful compact evidence,
without extending either life or claiming mastery, broad reliability or CDP
superiority. Any changed-budget or guidance test needs a separate selection;
none is active or queued by this result. The CPU delayed-teacher reference
remains feasibility evidence only; no delayed-guidance implementation is applied.

Protocol:[fresh4013 check](../experiments/2026-10-26-cdp-qbert-expert-seed4013.md).
Artifacts:`/x/Temp/kindle-expert-20261026/qbert-4m-48-r32-expert1-cap640-200k-seed4013`.
Learning review`cd1e51e85eebdb007951a5d8950ccb3798e36784dbb40cf9c057853bf9344c36`;
completion`0489d949361d7f645fa03f87c5e25ad38de1df0cd4db65e3f7fc45a1ef771044`;
comparison`87faaa70a7d1e19249d381992ab7b97e02a47e55fb9931465b308e57baef24da`;
exposure`ddcc3c669cc8ce495b55dd39213d88ded3f528d4d4ad50119595ac439b8f056a`;
tile replay`57ee54e4e3a3534ff8206d4e65a9c7a4a5aeea5f0a22f38bfe2c5a9ff29f4c0d`.


## After the first pyramid, movement continues but the teacher is limited

Read-only analysis of both completed expert lives verifies every completed
training/frozen episode's action count, reward sum and frame count against the
existing full replays. Among successful frozen episodes,1009 chooses NOOP for
554/602 actions after the first-pyramid milestone (92.03%);4013 chooses it for
only559/6384 (8.76%). Thus4013's weak full-game score is not explained by simply
stopping movement. The17 and2 episodes are selected outcomes within reused
cohorts, not independent learners or a causal comparison.

The capped teacher supplies only NOOP labels after that milestone:3136 on1009
and2703 on4013. Its source recognizes the first board's blue→yellow target.
Later reward totals include delayed completion payouts: in one exact CPU replay
of4013 lane0/episode0, score850 at frame1156 rises to3950 at1396 and ends4075.
Native RGB snapshots retain the completed yellow board, then show blue tiles
on subsequent movement. This motivates testing continuation behavior, without
claiming a particular cause of the learned policy's shortfall. The milestone
analysis excludes the action containing the milestone; it does not measure a
validated second-pyramid success count.

Two bounded CPU prototypes keep the existing movement rule and receive only
native max-pooled RGB. Neither constructs a Kindle agent, changes runtime code,
uses RAM/rewards as policy inputs or consumes GPU work. Each uses the first
natural episode of8 reused4013 evaluation lanes, followed by independent exact
saved-action/pixel/reward/boundary replay:

| CPU teacher prototype | Mean natural return | First pyramids | Actions / frames | Collection / replay |
| --- | ---: | ---: | ---: | ---: |
| Alternate blue/yellow target after a full-color board |6615.625|8/8|9116 /36,451|12.608s /3.439s|
| Infer board colors and pause64 actions after completion |5840.625|8/8|8637 /34,531|22.341s /3.345s|

Both complete8 natural episodes without truncations or partial tails, and both
remain below15000. The second prototype sees later gray palettes but performs
worse on this small reused cohort. These are hand-coded reference outcomes,
not learned success; repeated development on the same8 seeds is not held-out
validation. No production guidance change or guidance-training allocation is
selected from these prototypes.

Select instead one fresh500k/R16 test of the existing expert recipe:2.5 times
the experience,62,470 versus49,939 updates,30-minute cap, same4013 seed and
fixed final/own-initial controls. The stronger4013 policy already moves after
its first pyramid; additional experience may help it learn continuation, but
prior unaided500k/R16 was negative. Budget and ratio change together. See
[the separate selection](../experiments/2026-10-27-cdp-qbert-expert-r16.md).

All diagnostics live under`/x/Temp/kindle-qbert-continuation-20261027`.
CPU preparation corrected a missing training-replay frame-offset field and an
unset module search path before successful collection; no native agent or GPU
stage was involved. The exact single-episode RGB replay takes0.588s.
Action analysis SHA256:
`00a8abbc938b5a5a67874e9b6e8a90007a793a3897a4e6a52f4b270e58214c48`.
Native RGB replay:
`57d757e876a084c700956e40e865b7a91590532beef37be41f510828f16b5e72`.
Alternating prototype:
`be1ad5044aafab070bd79e9f85ef8b43500576ce27093e182765b0dfd2b20df9`.
Palette prototype:
`bd8807697dbcc4b657553be31c635cd6e8a195e42b9b7a8f49108d78a9968abd`.


## Score-color prototypes remain below the game gate

The original manual identifies the score/life-icon color as the current tile
color target and specifies a3100-point completion bonus. Thus post-milestone
reward totals include completion payouts, not exclusively progress on a later
board. [Parker Brothers1983 instructions](https://www.atariage.com/manual_html_page.php?SoftwareLabelID=379).

Five bounded CPU developments use only native max-pooled RGB and the existing
movement rule. Each collects the first natural episode of the same8 reused4013
evaluation lanes, then independently replays saved actions and checks every
pixel hash, reward and boundary. All complete8 natural episodes, zero
truncations/partial tails, and exact replay. No Kindle agent or GPU is used.

| Prototype | Mean return | First pyramids | Actions / emulator frames | Collection / replay |
| --- | ---: | ---: | ---: | ---: |
| Score-color reader without initial fallback |0.000|0/8|8304 /33,216|4.775s /3.144s|
| Initial yellow fallback, broad score crop |3684.375|8/8|8805 /35,201|21.876s /3.384s|
| Narrow digit crop |3684.375|8/8|8805 /35,201|21.286s /3.412s|
| Narrow crop plus a hop to a neighboring tile after full color |4825.000|8/8|8649 /34,581|21.010s /3.369s|
| Alternating transition target plus stable digit color |6743.750|8/8|9024 /36,091|26.185s /3.494s|

The first reader never obtains a target before scoring and stays idle. Initial
fallback restores first-pyramid behavior. The broad crop can admit the orange
character; narrowing it removes that observed color error but does not improve
scores. Continuing with a board hop exposes transition flashing, which the
three-observation reader incorrectly accepts as target changes. The final
prototype requires8 consecutive visible observations, preserves the original
base/target color handling, and alternates provisionally after a full-color
board. Its recorded targets reach the later gray palette in5/8 lanes, but the
mean is only128.125 above the earlier alternating reference's6615.625. No
validated later-pyramid success count or causal mechanism is claimed.

All failed attempts are retained. This is repeated selection on one small
reused cohort, not held-out validation or learned success. The best score still
misses15000 substantially; no production guidance change, seed campaign or new
GPU allocation is selected from these results. The existing500k/R16 learner
kept its original teacher and30-minute cap; its completed result follows.

Collection/replay totals111.935s on one affined CPU core at nice10 during the
active learner. This is recorded as a co-host timing caveat. All67 active pins
remain unchanged afterward; the service retains PID61973 and invocation
`66573c6168d640428dfcdafa9d82ca17`. No agent, native training source or runtime
is modified by the prototypes.

Artifact root:`/x/Temp/kindle-qbert-continuation-20261027`.
The aggregate`hud-teacher-review.json` pins every per-prototype review, script
and exact trace, preserving all scores, natural episodes, raw-frame counts and
CPU times. SHA256:
`61d328699bf25851672c55765196f14fb4dc235fb2335b85e3f6e3b494933355`.


A subsequent CPU check changes only the hybrid teacher's minimum hop age6→2.
All eight saved action/pixel/reward/boundary traces are byte-identical to the
hybrid control, including6743.750 mean and8/8 first pyramids. The minimum age
was not binding on these trajectories; this does not diagnose other timing
rules or native learning. Collection26.354s plus replay3.493s bring the measured
co-host CPU total to141.783s. No production change or allocation follows.
Receipt`hop-wait-check.json` SHA256:
`07a259da3b4286b06b97293a45b4ec3dab442df9417893b0035e25691a2e3abb`.

## More experience at R16 improves first-pyramid success, with higher cost

The fresh expert-imitation4013 candidate completes **500,000 aggregate actions
/1,998,773 emulator frames /62,470 updates in28m20s** (1700.485s), within its
30-minute training cap. Fixed frozen quality improves to **3752.083 /21 of24
first pyramids**, versus the completed200k/R32 control's3153.125 /17. Own
initial182.292 /0 is unchanged. The mean gains19.00% for40.88% more training
time. Both the original mean15000 and90%-first-pyramid requirements remain
unmet; the five-game goal is unfinished.

| Fresh4013 budget | Actions / updates | Training wall | Frozen final / own initial | First pyramids | Mean coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| R32 |200k /49,939|20m07s|3153.125 /182.292|17/24|20.167 /21|
| R16 |500k /62,470|28m20s|3752.083 /182.292|21/24|20.333 /21|

Only action budget and replay ratio change, along with their declared cap and
fresh output paths. Both use native/CLI79c9a7f,4M/RGB48/Auto/N8/B8/T16/H15/
coefficient1/actor0/full18/original rates and5% first-pyramid guidance/cap640/
expert1/NOOPweight.05. This is2.5 times the experience and1.251 times the
updates, not an isolated ratio effect. Seed4013 was selected after its stronger
result and the evaluation cohort is reused. No checkpoint, replay, belief,
game or RNG is reused, but this is not an independent seed confirmation.

All62,470 reports and346 tensors are finite. All67 source/evidence pins remain
unchanged. First update248, zero debt, four100k recovery checks, exact config/
source/counters, explicit trusted labels, full training/frozen pixel-action
replays, videos, guards, curves, coverage, exposure and comparison pass. Initial
tensors match all346 control tensors, and the initial evaluations have identical
1140-event semantic traces. No failed training/audit attempt, GPU retry, cap
increase or extension occurs in this life.

Each frozen arm completes exactly24 natural episodes with zero updates and
truncations,346 unchanged tensors and retained partial tails. Final evaluation:
20,256 actions /80,984 frames in8.244s. Initial:8728 /34,875 in3.565s. Final
coverage histogram is10:1,17:1,20:1,21:21; success87.5%. Scores range275–4600,
so improved first-pyramid frequency still leaves a large full-game score gap.

Guidance selects45 of817 started episodes and executes28,324 trusted actions
(5.6648%):26,669 NOOP and1655 movements. All44 completed guided episodes clear
a pyramid, mean3831.818; one selected episode remains partial. Of765 unguided
training episodes,264 clear a pyramid, mean1717.712. The last50 unguided average
3788.000 with44 successes. These online figures remain separate from the fixed
frozen outcome. The online curve dips markedly between about220k and330k,
then recovers; no earlier checkpoint is selected from that curve.

24,212 updates contain expert labels; repeated replay contributes464,475
labelled rows with total target weight48,926.944. Weighted cross-entropy over
successive50k-action windows falls1.601→0.316, with intervening variation.
These changing replay samples do not measure held-out action accuracy or
establish why the policy improves.

The service exits successfully **00:48:34 UTC on October27**,31m25.790s total
including setup, evaluation and CPU audits. Postprocessing completes00:48:42;
all services and observers are terminal. The independent CPU teacher checks
consume141.783s on one affined low-priority core during training, a co-host
timing caveat. The runtime, native learning and action guidance remain unchanged.

Retain both measured cost/quality points. The larger budget improves this reused
seed but still fails both gate components; no unchanged extension, seed campaign
or automatic successor follows. The CPU continuation rules remain unselected.

Protocol:[fresh500k/R16 comparison](../experiments/2026-10-27-cdp-qbert-expert-r16.md).
Artifacts:
`/x/Temp/kindle-qbert-continuation-20261027/qbert-4m-48-r16-expert1-cap640-500k-seed4013`.
Learning review`3cad7a4285536a7d484759f3f483250636a82dac9fb66d031749f2c8b1dcaa40`;
completion`391554dfef3c781c9a432088d4f59c64e5b622594ff3bb9eea1b2ee5a8e84622`;
comparison`58879af670c951c03fce61c5a07a2d50233a3643a4a6f1242eca4ff0e5ed9d8a`;
exposure`5e311a769d85f41e340e7ca23812ce658029bf265097b13ff53e2a37cbef1ef5`.


## Repeated movement and pixel avoidance strengthen the CPU reference

A CPU replay of the saved hybrid reference's lane0 reproduces1096 actions,
4382 frames and7550 points exactly. Its rule releases29 pending moves at age16
while the character remains visible; many intervals between move requests are
17 actions. Native RGB snapshots place purple sprites near recorded life losses.
Lives are read only after each step for analysis and never passed to the policy.
This motivates two focused CPU candidates, not a native-learning diagnosis.

| Same eight reused4013 first episodes | Mean natural return | First pyramids | Actions /frames | Collection /replay |
|---|---:|---:|---:|---:|
| Existing hybrid color rule |6743.750|8/8|9024 /36,091|26.185s /3.494s|
| Repeat a new hop while its origin stays visible, at most3 further actions |14703.125|8/8|11,581 /46,306|33.400s /4.242s|
| Same, reroute a new hop away from nearby purple pixels |21825.000|8/8|24,733 /98,913|71.038s /8.002s|

Both new candidates complete all8 natural episodes without truncations or tails;
every saved output pixel, action, reward, frame count and boundary replays exactly.
No native module is loaded. The avoidance rule uses current RGB, the existing
route graph and no RAM, reward or task inputs. The repeated-command candidate
still misses15000; all8 avoidance episodes exceed15000 (19100–22500). These are
hand-coded reference scores on repeatedly selected development episodes, not
Kindle learning or independent confirmation. Provisional color-flip counts are
not validated later-pyramid counts.

Select only a fixed CPU standard24 assessment of the unchanged avoidance rule:
first3 natural episodes per8 lanes, retaining excess episodes and partial tails,
120k-action/360-second cap per collection/replay pass, exact input/output/reset
pixels and outcome replay. The earlier8 are included in24. That assessment is
active; no production guidance patch, native training allocation or successor
is selected. Breakout's active learner retains the same PID/invocation and all46
input pins. These CPU jobs run on one affined core at nice10; measured successful
diagnostic/collection/replay portions total123.747s before the24 assessment and
remain a co-host timing caveat.

Artifacts:`/x/Temp/kindle-qbert-continuation-20261027`.
Aggregate`movement-avoidance-review.json` SHA256
`b71b9391ea0e6ea380c9081af23afa94636607b2d577b36808523e8b60a7e0b2`;
repeat review`595125ad6b661c9aa69ff4b67cf94a67dc40eb5d408d6d996927ca07fa5e8395`;
avoidance review`99afab5535aec6b2dd311fd08b3cb5bed7bf2ac945ba076e50ca5fe14b033c60`.


## Fixed24 CPU reference passes and optional adapter remains staged

The unchanged avoidance reference completes **80,496 aggregate actions /321,923
emulator frames**, with first3 natural episodes per lane selecting exactly24.
Mean **22050.000**,24/24 first pyramids and24/24 individual scores at least15000
(range19100–22500). All24 completed episodes and8 partial tails are retained.
Collection233.913s and independent replay27.403s both stay within their declared
360-second caps. Every input/output/reset pixel hash, saved action, reward,
frame count and boundary matches. The first8 outcomes equal the earlier cohort
and are included in24; this is not independent confirmation or native learning.
No native module is loaded. Standard review SHA256
`87e852aa2acdc8d7e48a36ea102c6e520bfae61a44857dbcf118497351790986`.

An optional continuation adapter is staged externally at
`/x/Temp/kindle-qbert-guidance-20261027`, without modifying the active checkout
or runtime. Its explicit `--qbert-guidance-policy continuation` requires a
positive guidance probability and an action cap, and remains fresh/Qbert/full18/
native-RGB only. The default first-pyramid class has identical AST; the shared
selection/reset schedule and native expert API stay unchanged. The continuation
rule is a separately declared guidance kind for audit/replay.

**456 CPU tests pass**,10 unchanged native-constructor validations deselected,
zero native construction attempts, in1.341s. Replaying all **24,733 actions
/98,913 frames** through the staged policy reproduces every saved action, output
pixel, reward and boundary in74.678s. No native module loads in this parity run.
All46 active Breakout pins and original Python sources remain unchanged;
`git apply --check` accepts the exact three-file patch. These are CPU adapter
checks; native execution and learned benefit remain unmeasured. No Qbert GPU
allocation is selected yet. Original runtimes/logs are retained.

Patch SHA256`aeb61a4510b9122ae0fbf4ee8777eb86e15d2d082ebc7ee7dccbe2a3351ebfbc`;
validation`82b534c911aa4499cafd14f8d50d146f22d8e637ffd5f90145b3405a25bd4563`;
policy parity`2f2d4d44f9b90412fad4234c5ee9531f790d9e857f32af33e63fa357db3c29dc`.
Measured standard24 and adapter portions add337.336s to the earlier123.747s
of affined CPU reference work during Breakout. Retain the co-host timing caveat;
no causal wall-time correction is claimed.


The exact validated adapter is subsequently applied after Breakout's complete
terminal audits and pin checks. Select only the [bounded native continuation
check](../experiments/2026-10-27-cdp-qbert-continuation.md); no full Qbert learner
is selected. Historical staged descriptions above preserve preparation order.


## Native continuation check passes

Fresh4013 completes **8192 actions /32,739 emulator frames /1987 updates in
70.493s**. First248, zero residual credit,3,578,225 parameters, all reports/tensors
finite. All346 initial tensors match the retained same-seed expert control;
CNN/reward/actor/value weights change. The check uses25% episode selection and
cap640, actually executing2987 trusted actions:2300 NOOP and687 hops.
1968 updates report nonzero imitation loss. Complete guided-input/action/outcome
replay and both host guards pass, including capped release. All32 declaration
pins remain unchanged. Native source79c9a7f is unchanged; adapter source3f3f7b5.

Frozen1024 actions /4096 frames take.531s, zero updates, all346 tensors invariant.
No natural frozen episode completes, so this establishes execution and restore,
not task competence. Service starts03:21:37 and exits03:23:09 UTC on October27,
1m31.375s total. No GPU stage repeats. The25% guidance check and its early cost
are not a full-run speed prediction. No full learner, cap increase, extension
or automatic promotion is selected by this result.

Artifacts:`/x/Temp/kindle-qbert-guidance-20261027/qbert-4m-48-continuation-check`.
Review SHA256`6f91901f01eb2101735282f6c2fb2a4bcff754863de1160edb3c29e393b5943c`.


## Fresh continuation comparison selected

The passing8192-action check is followed by a separately selected fresh200k
comparison with5% episode guidance, continuation policy and cap1920. Model,
learning settings, native binary and host match the retained4013/R32/200k
first-pyramid control; teacher and cap change together. Saved reference prefixes
at640/1280/1920 actions average4875.000/11311.458/15026.042, motivating1920
without claiming those are natural-episode or learned scores. Keep30-minute
training cap and fixed final/own-initial24-natural unguided evaluations against
the original15k/90% gate. See the [selected protocol](../experiments/2026-10-27-cdp-qbert-continuation.md#selected-fresh-200k-comparison).
No training outcome is available yet; no extension or automatic successor.


## Continuation guidance improves frozen learning but misses the full gate

The fresh4013 continuation life completes **200k aggregate actions /799,570
emulator frames /49,939 updates in25m23s** (1522.860s), within30 minutes.
Fixed final frozen mean **4730.208 with23/24 first pyramids**, versus its own
actual initial182.292/0. Coverage rises5.167→20.583 of21 cubes. The95.83%
first-pyramid rate passes that part of the gate, but **mean15000 still fails**;
none of the24 selected returns reaches15000. This is useful development
learning, not completed Qbert success or five-game reliability.

| Fresh recipe on4013 | Actions / updates | Training wall | Frozen mean | First pyramids |
| --- | ---: | ---: | ---: | ---: |
| First-pyramid teacher/cap640/R32 |200k /49,939|20m07s|3153.125|17/24|
| First-pyramid teacher/cap640/R16 |500k /62,470|28m20s|3752.083|21/24|
| Continuation teacher/cap1920/R32 |200k /49,939|25m23s|4730.208|23/24|

Against the primary200k control, mean improves50.02% for26.17% more training
time. Against the secondary500k/R16 control it improves26.07% with10.45% less
time. All use the same native binary and host; primary model/learning settings
match. Teacher behavior and guidance cap change together; the secondary also
changes ratio and budget. Seed4013 and the cap were selected on reused
observations. Co-host CPU work is recorded in the protocol. No isolated causal
effect, hardware speedup or independent seed confirmation is established.
Retain the recipe as promising; no extension or next allocation is selected.

Configuration remains4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18,
original rates, expert scale1/NOOPweight.05,5% episode selection/cap1920;
evaluation4000004013, no restored training state. Source8761ff2/native79c9a7f.
Actual guidance is34,468 actions (17.234%):26,336 NOOP and8132 hops.18 of295
episode starts are selected. The17 completed guided episodes average15110.294;
270 unguided online episodes average1754.444, last50 mean3983.500 with43 first
pyramids. These changing-policy online scores are distinct from frozen learning.
38,378 reports contain1,148,366 repeated labeled replay rows. Weighted label
cross-entropy falls1.036→.209 across50k windows on changing samples, not held-out
accuracy; lower imitation loss alone does not establish the remaining mechanism.

All numerical/config/source/counter checks,100k recovery,346 initial tensors
matching control/check,346 unchanged tensors per frozen arm, identical initial
semantic traces,56 declared pins, complete guided input/action/outcome replay,
frozen recorded-outcome replay/videos, host guards, curves, coverage, exposure
and comparison pass. Final evaluation takes26,768 actions /107,020 frames /
12.511s and selects first3 natural episodes per lane from32 completed. Initial
uses8728 /34,875 /4.086s,24 completed. Zero updates/truncations; excess episodes
and partial tails are retained. Ordinary frozen logs have no live pixel hashes.

Main service exits successfully04:07:44 UTC,28m54.691s total including frozen
assessments/audits; CPU postprocessing completes04:08:08. No GPU stage repeats.
Artifacts:`/x/Temp/kindle-qbert-guidance-20261027/qbert-4m-48-r32-continuation-cap1920-expert1-200k-seed4013`.
Learning review SHA256`bc536f1deb00295678dbb9765ecc9f6ab4d1a08dc426b519291bbac5a1ac095c`;
completion`6e35aa36ee74291debec6a2635ee3aa6f003a3176cbf0efaa7543b7c8c45da59`;
comparison`ef10ab0f268d3385f82238625b97010a657db0d09f0e4b322f7229640ee91e8d`;
exposure`0e2025a544e7a6d5f9acc3c23e98e975307ab80977c3457c7db437f5a8f1cba3`.
The comparison PNG/SVG and curves remain outside Git with their receipts.

## Saved replay locates the remaining gap after the first pyramid

CPU analysis of the completed continuation4013 frozen cohort finds23/24 first
pyramids but only **3/24 second-board color-cycle signatures**: all21 cube bytes
become148, followed5–7 raw frames later by a uniform completion flash. Among
the23 episodes reaching the second board, peak target-color coverage averages
16.739/21; four stall at20 and two at19. This is a post-hoc color diagnostic,
not a newly qualified general level detector or a changed success gate.

Every first completion adds **3100 automatic bonus points** before the next
board starts162 raw frames later. Thus reward after the first milestone cannot
be read as later-board tile progress. The native policy does make progress on
the second board, but most of these episodes end before its completion. Lane0's
first episode reaches19/21 second-board colors, then loses its last three lives
without further tile progress; its third episode completes the second color
cycle and ends at7875 on the next palette. Saved native210×160 snapshots are
illustrations from lane0 only; the counts use all24 selected episodes.

After the first milestone, the continuation policy executes13,407 actions:
8703 NOOP (64.91%),4703 cardinal commands and one other action. The old200k
policy uses all18 actions, with8.76% NOOP among6384 post-milestone actions;
old500k uses31.31% NOOP among11,672. Restriction to the teacher's five-action
vocabulary shows behavioral transfer, not correct timing or a causal diagnosis.
The changing state distributions prevent treating these fractions as accuracy.

Replaying the26,768 saved frozen actions /107,020 frames matches every recorded
reward, boundary, reset/frame counter, completed episode and partial tail.
RAM cube colors and lives are observer outputs only; no native policy receives
them and no new action decisions or learner updates occur. The12.301s replay
and.025s summary run on one affined low-priority core during the1009 confirmation.
The preceding2.518s log-only analysis runs as a single unpinned CPU process.
All60 active pins remain unchanged. Ordinary frozen logs lack live pixel
hashes, so reconstruction hashes do not establish direct live-image equality.

Artifacts:`/x/Temp/kindle-qbert-transfer-20261027`.
After-pyramid review SHA256`f0619ce871fe9227c4e0bdcc638d7a68963dd0d69213895f9692ff1991d0f21a`;
raw-frame replay`854cda3ec027729e709a47085aa0a7e210001c51ad6351b23e3a6ef74c85d5d8`;
color summary`bb8dc27163ea03825b8633e3201f0f33ef76b2838c0691270600d5de30f543ab`.
No runtime, guidance policy, allocation or evaluation criterion changes.

## Continuation improves the weaker seed but both miss the full gate

The fixed-recipe fresh1009 confirmation completes **200k aggregate actions
/799,538 emulator frames /49,939 updates in26m38s** (1597.708s). Frozen mean
**3137.500 with17/24 first pyramids** improves its paired old-teacher control's
1083.333/2, versus the identical own initial180.208/0. Coverage rises5.833→20.167
of21 cubes. Mean improves189.62% for31.96% more training time than20m11s.
Both the original mean15000 and90%-pyramid requirements fail on this seed.

| Seed | Old teacher/cap640: mean / pyramids / training | Continuation/cap1920: mean / pyramids / training |
| --- | --- | --- |
|1009|1083.333 /2 of24 /20m11s|3137.500 /17 of24 /26m38s|
|4013|3153.125 /17 of24 /20m07s|4730.208 /23 of24 /25m23s|

All four lives use200k actions/49,939 updates. The two continuation lives have
identical4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original rates,
5%episode continuation/cap1920/expert1/NOOPweight.05, native binary, Python
runtime and host; only learner/evaluation seeds differ. Each starts fresh and
matches all346 initial tensors and frozen initial semantic traces of its paired
control. Different learner seeds have different initial weights. Teacher and
cap change together against the old controls. Both seeds/cohorts were reused
in development; two improvements do not establish held-out reliability or full
Qbert success. No extension or third seed is selected.

Actual1009 guidance is38,400 actions (19.200%), from20 of319 episode starts:
29,393 NOOP and9007 hops. All20 completed guided episodes clear a first pyramid
and average15022.500, but these returns include teacher actions. The291
unguided online episodes average1110.052 with58 first pyramids; their last50
average2617.000 with29 first pyramids.42,319 supervised reports sample1,284,883 repeated
labeled replay rows. Weighted label cross-entropy falls1.260→.394 across50k
windows on changing samples; this is not held-out imitation accuracy.

All numerical/config/source/counter checks,100k recovery,346 matched initial
and unchanged frozen tensors, paired initial semantic traces,60 pins, exact
guided input/action/outcome replay, frozen outcome replays/videos, host guards,
curves, coverage, exposure and four-arm comparison pass. Final frozen uses
19,040 actions /76,121 frames /9.015s; initial9232 /36,888 /4.384s. Each selects
the first3 natural episodes per lane,24 total, with zero updates/truncations;
excess episodes and partial tails remain retained. Ordinary frozen logs lack
live pixel hashes. The comparison figure is visually checked and stays external.

Main service exits successfully04:47:41 UTC on October27,30m09.082s total;
training itself stays inside30 minutes. CPU postprocessing exits04:47:57.
No GPU stage repeats. Source`7560d2d`/native`79c9a7f`; co-host CPU diagnostics
and preparation are recorded in the protocol and are not wall-time corrections.

Artifacts:`/x/Temp/kindle-qbert-guidance-20261027/qbert-4m-48-r32-continuation-cap1920-expert1-200k-seed1009`.
Learning review SHA256`18e4a3a5bb0ac330fd04c6d1e12dfa2baa53ff51dac13aa8c56261cfb0a657ff`;
completion`cec2e9ccb4fbb75c8de4b8c716430f7d83e0555219f25bddcd037410c48d0bab`;
comparison`2fc6eca1584e980c8078b2afac9b3b572298c3870afb9af6661a3d6ccdf2a749`;
exposure`9d9d18f654e9276ff0df70d7458bc42a17901b0b6312d42dc6280f12e420e1cb`.

## Higher-exposure R16 bounded check passes

Fresh4013 at4M48/Auto/N8/B8/T16/H15/R16,20% episode continuation/cap1920,
expert1/NOOPweight.05 completes **8192 actions /32,752 frames /994 updates in
52.742s**. First248, zero residual credit; the native model, rates, coefficient1,
actor0 and full18 actions remain unchanged. All reports/346 tensors are finite,
all346 initial tensors match continuation4013, and CNN/reward/actor/value
weights change. Actual guidance4527 actions (55.261%):3491 NOOP and1036 hops;
990 reports contain expert labels. Exact guided input/action/outcome replay and
both host guards pass; all33 pins remain unchanged.

Frozen1024 actions /4096 frames take.524s, zero updates and346 invariant tensors.
No natural frozen episode completes; this is execution and early-cost evidence,
not task competence. The check does not reach cap1920 in any lane; prior full
continuation lives cover its unchanged release schedule. No check experience
is reused. Service starts04:51:06 UTC/source`e920f7b`, exits04:52:23 successfully,
1m16.753s total. No GPU stage repeats.

The measured early rate projects to21.46 minutes for200k, with uncertain later
guidance fraction and update cost. Therefore the previously prepared, unselected
20-minute full-run draft is archived; the separately selected fresh comparison
declares a25-minute training cap before launch. Neither bound is extended.
Probability and replay ratio change together, and the original mean15000/90%
gate remains unchanged. See the [selected protocol](../experiments/2026-10-27-cdp-qbert-continuation.md#selected-fresh-higher-exposure-r16-comparison).

Artifacts:`/x/Temp/kindle-qbert-exposure-20261027/qbert-4m-48-r16-p20-check`.
Review SHA256`9b03f5c7dc686b554b821d9a3fdbc6912dfa73c38d924c7a261730db575871ef`.

## More guidance with less replay saves time but lowers frozen quality

The fresh4013 comparison completes **200,000 aggregate actions /799,662
emulator frames /24,970 updates in21m59s** (1318.663s). Relative to the matched
5%/R32 continuation control, training takes13.41% less time but frozen mean
falls29.53%, from4730.208 to3333.333. Retain5%/R32; no extension or confirmation
of this candidate is selected.

| Recipe | Frozen mean | First pyramids | Mean first-board coverage | Training |
| --- | ---: | ---: | ---: | ---: |
| Continuation5% /R32 |4730.208 |23/24 |20.583/21 |25m23s |
| Continuation20% /R16 |3333.333 |18/24 |20.750/21 |21m59s |
| Own actual initial |182.292 |0/24 |5.167/21 |0 |

Keep4M/RGB48/Auto/N8/B8/T16/H15, coefficient1/actor0/full18/original rates,
continuation/cap1920/expert1/NOOPweight.05. Only episode probability5%→20% and
ratio32→16 jointly change against the primary control; same native/runtime/host,
all346 initial tensors and paired initial semantic traces match. This reused
seed and cohort are development evidence, not isolated attribution or an
independent confirmation. Both trained means remain below15000; the candidate
also misses90% first-pyramid success. No selected episode reaches15000.

The20% episode schedule selects49/225 starts and actually overrides91,913/200k
actions (45.9565%):70,218 NOOP and21,695 cardinal hops. All46 completed guided
episodes clear a first pyramid, mean15127.174;171 unguided completed episodes
average1305.994 with45 first pyramids. The final50 unguided average2585.5 with29
first pyramids. There are8692 native actions after guidance releases. These
online returns contain different trajectory distributions and are not frozen
competence.24,835 reports contain labels, with1,519,459 repeated replay rows,
weighted target mass416148.910 and summed reported loss1733.083. Weighted
cross entropy across50k-action windows is1.2024/.3567/.1955/.1610; these changing
replay samples are not held-out accuracy. Lower imitation loss does not establish
better task behavior.

All24,970 reports are finite; first update248 and zero residual debt match.
The100k recovery contains12,470 updates/346 finite tensors. Both fixed frozen
arms select first3 natural episodes per8 lanes,24 each, with zero updates or
truncations and all346 tensors unchanged. Final uses22,112 actions/88,401 frames
in10.516s; initial8728/34,875 in4.087s. All47 pins, configuration/counters,
matched initial weights/traces, full guided input-pixel/action/outcome replay,
frozen outcome replay/videos, host guards, curves, coverage, exposure and
comparison pass. Ordinary frozen logs have no live pixel hashes. The comparison
plot was visually checked. Main service exits05:23:55 UTC and postprocessing
05:24:19 on October27; no GPU stage repeats or cap increases.

Artifacts:`/x/Temp/kindle-qbert-exposure-20261027/qbert-4m-48-r16-p20-cap1920-expert1-200k-seed4013`.
Learning review SHA256`0ba07bfde70af02425b8d2b867f1727dbbe9a93d32cfe424c053fa7f4ad4d7a0`;
completion`eeccfd460ba6afe09618cfd21e250f24bcf44b294ab5c4c68cc84f8b8b6c0515`;
comparison`7a9f10c2d77afc845dfaadffbea2630fd5e5b769322fa8179978e427d3ffe182`;
coverage`8b9f3e4d1b6068ca5b39d8faf862500569d838d561b0ab9c094ac4906bbe891d`;
exposure`bebaffd8faff07c4443504ae313a970b9e6b88877c10542f4b3e95ffde227172`.
The preceding protocol retains the prelaunch25-minute cap, bounded check,
unselected20-minute draft and2.067s of co-host CPU preparation/analysis.

## Expert labels currently train the actor without direct world gradients

A read-only review verifies that`behavior.rs`, `agent.rs` and`world.rs` exactly
match native source`79c9a7f`. The [expert term](../../kindle/src/dreamer/behavior.rs)
uses replay features as graph inputs and trains the behavior actor. The
[agent](../../kindle/src/dreamer/agent.rs) builds separate world and behavior
training sessions and uploads expert targets only to the latter. The
[world graph](../../kindle/src/dreamer/world.rs) has no expert target or loss;
`actor_critic_gradient` routes the initial imagined policy/value losses through
frozen heads into world state. Enabling that existing option therefore does
not route expert-label gradients into the encoder/world.

Teacher actions still change experience and ordinary world learning. This
source-path fact is not a measurement of raw derivatives or a diagnosis of
weak policy transfer. A direct supervised representation objective would
require an explicit native implementation and bounded gradient/alignment
checks; none is selected or implemented here. No runtime/default changes.
The.447s affined CPU receipt check constructs no agent/environment and leaves
all58 active Pong inputs unchanged. Receipt:
`/x/Temp/kindle-expert-path-20261027/source-review.json`, SHA256
`0150815d50796637e2608c4b06961b24c1d071208aaba2704021b59148511159`.

## Saved action timing does not support a simple excess-waiting diagnosis

A log-only comparison uses all24 selected natural episodes from each retained
continuation learner and the stronger pixel teacher. All executed actions,
completed reward/frame accounting and source hashes reconcile. The4013
post-first-pyramid count exactly reproduces the earlier13,407 actions/8703
NOOP review; no emulator, new decisions or native updates are used.

| Policy | Selected actions | NOOP share | Median NOOP run | Cardinal action followed by same command |
| --- | ---: | ---: | ---: | ---: |
| Native continuation4013, score4730.208 |19,399 |65.81% |5 |61.36% |
| Native continuation1009, score3137.500 |15,147 |68.05% |5 |57.15% |
| Pixel teacher4013, score22050 |75,674 |76.22% |7 |60.36% |

Median same-command cardinal runs are2 actions in all three policies. After
excluding actions until162 frames after the first milestone, NOOP shares are
65.48%/56.70%/76.07% respectively. The162-frame transition duration was verified
in the earlier native4013 color replay; for the other rows it is only a fixed
time exclusion, not a qualified general level detector. The teacher therefore
waits more in these marginal summaries, while the learned policies repeat
cardinal commands at similar rates. The earlier64.91% post-milestone NOOP share
alone does not show excess waiting.

These policies visit different states, survive different lengths and complete
different numbers of boards. Requested commands are not actual movement under
sticky actions. Counts are descriptive, not matching-state teacher accuracy,
causal attribution, a NOOP-weight prescription or proof of a sampling problem.
They motivate avoiding an inactivity diagnosis from NOOP counts alone. A fixed
retained-weight greedy assessment can cheaply test action selection directly;
no further training is needed for that question.

The.162s affined low-priority CPU analysis leaves all47 active Breakout pins
unchanged. Artifacts:`/x/Temp/kindle-qbert-transfer-20261027/action-timing-review.json`,
SHA256`f3582c8d8883e6d25566cfd91927ef1337ae110f4ad0a916f337836d66b623ef`.
All observations are post-hoc on reused development cohorts; task gates stay
unchanged and broad reliability remains unproved.

## Prepared greedy comparison of both retained continuation learners

Prepare a fixed four-arm comparison: final1009/initial1009/final4013/initial4013
from the completed5%/R32/cap1920 continuation lives. Both original sampled
outcomes3137.500/17 first pyramids and4730.208/23 remain; neither meets the full
mean15000/90%-pyramid gate. Existing native `--greedy` changes only frozen
action selection/output locations. There is no new learner or additional
training, and no assumption that the narrow Pong improvement transfers.

Keep exact original Qbert runner SHA256
`9b7e6631b43cf4368cca902b16c9027a88cb11d11088e4199b42da4452f89b58`,
packaged runtime/native79c9a7f, all12 original runtime pins per source, wrapper,
ROM, host, seeds4000001009/4000004013, full18/sticky.25/repeat4/noops0/native
frames/one GPU48 resize. Each arm retains first3 natural episodes per8 lanes=24,
600k-action/300s caps, zero updates/truncations and the original mean15000/90%
gate. Require346 invariant tensors per arm, source invariance, full recorded-
action outcome replays/videos and guards. Audit1200s/service2460s. Ordinary
frozen logs have no live pixel hashes.

Four commands parse, five scripts compile, all four source checkpoints contain
346 finite tensors each, and sampled scores/criteria recompute exactly from
verified saved replays. The controller refuses while unselected. Preparation
and checks use.148s/.459s on one affined low-priority core, constructing no
native module, agent or environment; all47 active Breakout pins remain unchanged.
The declaration pins60 inputs. No greedy score existed at preparation.

At preparation the assessment stayed **unselected** until the active Breakout
life and all postprocessing finished, followed by a separate terminal-review
decision. Selection and completed results follow below.
It cannot launch automatically. No retry, cap increase, extension, checkpoint
selection or successor. Reused development weights/cohorts do not become new
independent learners or establish broad reliability. Retain original training
costs26m38s/25m23s; frozen work adds cost and cannot relabel sampled failures.

Artifacts:`/x/Temp/kindle-qbert-greedy-20261027/qbert-4m-48-continuation-two-seed-greedy-frozen`.
Prepared declaration SHA256`037b86f98297970bc1b4efeb343f6ca3def04e4fe38d1fcb925f5770558ca3b4`;
CPU preflight`84de1e79bf3e4b905a15deac7c1579829051595845c1ca91e77200ae8ef76652`;
controller`cbd80844532238fe51e5835cf3a4350428fc1803d41622a867d19b2b74a746b4`.

## Retained Qbert greedy comparison selected

After complete Breakout cap1920 audits and both main/postprocessing services
are inactive/MainPID0/Resultsuccess, select only the prepared four-arm greedy
assessment above. The preceding terminal receipt is
`cae88022cf55cd8d1461bc3a4b8781557980432d463d8516c7ff47666c373ae0`.
Keep all60 pins, fixed source checkpoints, original host/runtime, four24-natural
assessments,600k-action/300s caps per arm, original mean15000/90%-pyramid gate,
zero updates/truncations,346 invariant tensors and complete replays/videos/
guards. Audit1200s/service2460s. No retry, cap increase, extension, new learner,
checkpoint selection or automatic successor. Archive the prepared declaration
before marking selection. No new training or global inference default change.
Owner:`kindle-qbert-continuation-two-seed-greedy-frozen-20261027`.

## Greedy evaluation lowers both retained Qbert results

The fixed four-arm assessment completes with **zero new training or updates**.
Both retained continuation learners lose mean return and first-pyramid success
under greedy action selection. Retain sampled actions for this Qbert recipe;
neither greedy learner meets the original mean15000/90%-pyramid gate.

| Learner | Sampled final / own initial | Greedy final / own initial | Sampled / greedy first pyramids | Original training |
|---|---:|---:|---:|---:|
|1009|3137.500 /180.208|3058.333 /85.417|17/24 →16/24|26m38s|
|4013|4730.208 /182.292|3770.833 /263.542|23/24 →19/24|25m23s|

All four initial controls have0/24 first pyramids. These are the same two
retained200k/49,939-update learners and reused evaluation cohorts, not new
independent learners. Original sampled outcomes and training costs remain;
the narrow Pong greedy success does not generalize to this Qbert comparison.
No global inference default changes, extension, third learner or successor
allocation is selected.

Only frozen action selection/output paths change. Exact original Qbert runner,
packaged runtime/native79c9a7f, host, saved config, wrapper/ROM, seeds4000001009/
4000004013, full18/sticky.25/repeat4/noops0/native frames/one GPU48 resize remain.
Each arm selects first3 natural episodes per8 lanes=24, with zero updates or
truncations and no guidance. All60 input pins, source checkpoints/config/counters,
346 invariant tensors per arm, recorded-action outcome replays/videos and
host guards pass. All source files remain unchanged. Ordinary frozen logs
have no live pixel hashes; replay validates outcomes, not live-image equality.
No GPU stage repeats or declared cap increases.

| Arm | Aggregate actions | Emulator frames | Evaluation seconds |
|---|---:|---:|---:|
|final1009|20,072|80,247|9.374|
|initial1009|9216|36,827|4.244|
|final4013|25,384|101,497|11.774|
|initial4013|10,432|41,697|4.884|
|Total|65,104|260,268|30.277|

Evaluation times exclude construction and audits; the entire service takes
**1m17.712s**, starts06:12:49 UTC/source`62482da`, exits06:14:07, and terminal
postprocessing completes06:14:10 on October27. PID98877/invocation
`88c34726d7074c7a9b1351c58dd22a82`. Both services are inactive with successful
results. The fixed600k-action/300s per-arm,1200s audit and2460s service caps
are unchanged. Excess completed episodes (28/24/27/25) and partial tails are
retained, but only the declared24 per arm enter the table.

Two ad-hoc read-only status inspections accidentally included `.gpu-memory.jsonl`
files and failed header lookup; explicit four-arm names resolve the inspection.
The pinned controller/audits were unaffected and no GPU work repeats.

Artifacts:`/x/Temp/kindle-qbert-greedy-20261027/qbert-4m-48-continuation-two-seed-greedy-frozen`.
Selected declaration SHA256`876359cc525a1b3088daac595e8f9e1c7431aa2ff938abeb6c69926007ded156`;
review`c57c6b70854a5ad56bba496b7d086b063175c20e2313caf075d476eeff724102`;
postprocessing`ae63ad499af82718c2355593238f23803025877fa0c6bd4d6c86b55ae99e5489`.

## Expert world gradients give a small R16 gain below the stronger control

The [fixed expert-world comparison](../experiments/2026-10-28-cdp-qbert-expert-world.md)
completes **200,000 aggregate actions /799,662 emulator frames /24,970 updates
in21m28s**. Frozen mean improves3333.333→3670.833 and first-pyramid success
18/24→19/24 against the retained20%/R16 control. Training takes2.32% less measured
time, and mean return is10.13% higher. These are one reused development seed's
point estimates, not an established speedup, reliability or isolated causal effect.

| Recipe, seed4013 | Actions / updates | Training wall | Frozen final / own initial | First pyramids |
|---|---:|---:|---:|---:|
|20% continuation/R16, world gradients off|200k /24,970|21m59s|3333.333 /182.292|18/24|
|Same, world gradients on|200k /24,970|21m28s|3670.833 /182.292|19/24|
|Stronger retained5%/R32, world gradients off|200k /49,939|25m23s|4730.208 /182.292|23/24|

All actual initial controls have0/24 pyramids. The new run remains22.40% below
the stronger control's score while taking15.42% less training time; it fails
both original mean15000/90%-pyramid criteria. Final first-pyramid coverage is
20.625/21 versus primary control20.750 and own initial5.167. No selected frozen
episode reaches15000. Retain5%/R32 as the stronger measured Qbert recipe and
keep this small R16 improvement as a budget point. No extension, seed
confirmation or new allocation is selected; world gradients remain default off.

The primary comparison keeps4M/RGB48/Auto/N8/B8/T16/H15/R16/coefficient1/
actor0/full18/original rates, continuation teacher,20% episode probability,
cap1920, expert scale1, NOOP weight.05, fresh4013/evaluation4000004013 and host
fixed. Only the expert-world-gradient learning setting changes, but native
source`0e157df`, exported runner and audited Python package differ from the
old control. All346 shared initial tensors and complete initial semantic
traces match; all357 initial tensors also match the preceding bounded check.
No weights, replay, belief, games or RNG are restored into training. The seed
and evaluation cohort are reused, and no same-binary off arm is measured.

All24,970 finite reports, first248/residual0,100k recovery/12,470 updates,
357-tensor checkpoint schemas/actor-copy equality,357 unchanged tensors per
frozen arm,52 unchanged declaration pins, host guards, exact guided input/
action/outcome replay, frozen outcome replays/videos, curves, coverage,
comparison and exposure pass. CNN/reward/actor/value parameters change during
joint learning; that does not assign those changes exclusively to expert loss.
The maximum world/behavior expert-loss difference is1.7e−7. Ordinary frozen
logs lack live pixel hashes, so their replays establish outcomes rather than
direct equality to live images.

| Frozen arm | Actions | Emulator frames | Seconds | Updates |
|---|---:|---:|---:|---:|
|Final|22,496|89,958|10.605|0|
|Actual initial|8728|34,875|4.125|0|

Each arm uses the first3 natural episodes per8 lanes=24, no truncations or
guidance. The30-minute training and300-second per-arm caps remain unchanged.
The main service starts14:54:02 UTC/source`a8da75d`, invocation
`020e6239255e4d4ba97293ab8815647d`, and succeeds at15:21:35 on October28.
Its27m32.975s wall time includes construction, evaluation and CPU replays;
the full training replay alone takes about5m18s.

Original CPU postprocessing fails at15:22:04 because importing the new
checkpoint helper eagerly imports ALE, absent from the plotting environment.
Keep the original failed service, journal, controller and52 pins. A separate
helper copy defers the unused checkpoint-audit import, leaving tensor comparison
unchanged. The failed comparison then passes, the missing exposure analysis
completes, and a separate repair receipt finishes at15:24:27. Existing curves/
coverage remain unchanged; no GPU stage repeats or original failure is relabeled.

Training overrides92,176 actions (**46.088%**, not20% of actions), including
70,408 NOOP and21,768 hops.49of219 episode starts receive guidance.45 completed
guided episodes average15150 and all clear a pyramid;166 unguided episodes
average1471.988 with46 first pyramids, and the last50 unguided average2907.5.
These guided online scores are not unaided competence.24,854 supervised reports
contain1,507,728 repeated replay-label rows, total weight412,366.564 and actor/
world loss sums1659.824352/1659.824550. Repeated labels are not unique examples.
The late weighted cross entropy.1459 is descriptive of changing training
samples, not held-out accuracy or a diagnosis of the remaining policy gap.

Artifacts:`/x/Temp/kindle-expert-world-20261028/qbert-4m-48-r16-p20-world-gradient-200k-seed4013`.
The directory retains curves, comparison PNG/SVG, videos, source/config,
checkpoints and original failure evidence. Selected declaration SHA256
`fb873d471da02acd66c90ecbd633a9c9e506a93b8c6e24b218bc242a0cf33209`;
learning review`d4e8b50e1868bc05ea5f4d04a87ac804c446c0e4daf6495ebf63e80fb6b4d9aa`;
comparison`c28961e0fcafec146483347222ed97af142874feca7a37da2c86c5b8615f6798`;
CPU repair`cd06fbc1f258b34c6ad2a7980ca0a4a5640e5251dbc072848496722a32268f24`;
completed postprocessing`fe37b62889b2eed68144a4073cb0094eb348368fd006b009d266879f8a26ce32`.

## Guided12M B16 R16 passes its bounded cost check

After the small4M/R16 world-gradient gain, a separate
[capacity check](../experiments/2026-10-28-cdp-qbert-expert-capacity.md) keeps
RGB48/Auto/N8/T16/H15/R16/coefficient1/actor0/full18,20% continuation/cap1920/
expert1/NOOP.05/world gradients and original rates, while changing4M→12M and
batch/microbatch8→16 together.14,123,473 parameters; fresh4013/eval4000004013.

It completes8192 actions /32,756 frames /489 updates in **60.333s**, followed by
1024 frozen /4096 frames /.810s, zero updates and no natural episodes. All41
pins,357 finite/invariant tensors,346 shared initial tensors against retained
12M4013 initialization, actor-copy equality, finite reports, changed parameters,
full guided replay and host guards pass.4492 trusted actions include1026 hops;
all489 reports are supervised, max world/actor expert-loss difference1.4e−7.
Service succeeds15:49:49 UTC on October28,1m25.463s total. No GPU stage repeats.

A final CPU prelaunch check corrects an assumed first384/residual0 to observed
first376/residual.5. The original unselected service refuses before any GPU
stage; its failure and40 pins remain. The separate reviewed declaration is the
first actual GPU execution. All budgets/caps remain fixed. The early rate
projects24.55min/200k, with later cost and learning unknown. This is execution
and early-cost evidence; no full learner or automatic promotion is selected.
The linked protocol records exact failures, identities, counters and hashes.


## Guided12M B16 R16 lowers quality and costs more

The separately declared [capacity comparison](../experiments/2026-10-28-cdp-qbert-expert-capacity.md)
completes200,000 aggregate actions /799,693 emulator frames /12,477 updates in
**24m51s** (1490.781s). Frozen **1792.708/9of24 first pyramids** improves its own
initial177.083/0, but is **51.16% below** the same-native4M/B8 control3670.833/19
while taking **15.74% more training time**. Both original mean15000/90%-pyramid
criteria fail. Retain4M/B8; no extension, seed confirmation or automatic successor
follows this negative result. The five-game goal remains unfinished.

| Recipe, fresh4013 | Actions / updates | Training wall | Frozen final / own initial | First pyramids |
|---|---:|---:|---:|---:|
|4M/B8/R16/20%, expert world gradients|200k /24,970|21m28s|3670.833 /182.292|19/24|
|12M/B16/R16/20%, expert world gradients|200k /12,477|24m51s|1792.708 /177.083|9/24|
|Stronger retained4M/B8/R32/5%, world gradients off|200k /49,939|25m23s|4730.208 /182.292|23/24|

All actual initial controls have0/24 pyramids. Candidate mean first-pyramid
coverage is19.833/21, versus4M/R16's20.625 and own initial4.917. No selected frozen
episode reaches15000; the maximum is4150. The stronger5%/R32 recipe remains the
best retained4013 Qbert budget point here, despite also missing the full-game gate.

The primary comparison keeps RGB48/Auto/N8/T16/H15/R16/coefficient1/actor0/
full18/original rates,20% episode continuation/cap1920/expert1/NOOPweight.05 and
expert world gradients fixed. Native`0e157df`, audited runtime, runner, teacher,
wrapper and host match exactly. Capacity4M→12M and batch/microbatch8→16 change
together; the optimizer cadence halves at the same replay positions per action.
This does not isolate capacity or establish that larger models generally fail.
Seed4013/evaluation4000004013 are reused development evidence. No weights,
replay, belief, games or RNG are restored into training.

All12,477 finite reports, first376/residual.5,100k/6227 recovery,357 finite tensor
schemas/actor-copy equality,357 unchanged tensors per frozen arm,63 unchanged
pins, changed CNN/reward/actor/value parameters, exact guided input/action/outcome
replay, frozen outcome replays/videos, guards, curves, coverage, exposure and
comparison pass. All357 initial tensors match the bounded12M check;346 shared
tensors match retained12M initialization only. Different-size actual initial
policies and semantic traces are assessed separately, not required to match.
Ordinary frozen logs lack live pixel hashes; replay verifies recorded outcomes.

| Frozen arm | Actions | Emulator frames | Seconds | Updates |
|---|---:|---:|---:|---:|
|Final|17,640|70,508|10.472|0|
|Actual initial|8728|34,881|5.362|0|

Each arm selects the first3 natural episodes per8 lanes=24, with no guidance or
truncations; excess episodes and partial tails remain recorded. Training and
per-arm caps remain1800s/300s. The service starts16:09:28 UTC/source`eebf838`,
PID171410/invocation`0a687e9f00f747189c3546f29534f036`, and succeeds16:40:13
on October28,30m44.506s including construction, evaluations and CPU replay.
CPU postprocessing succeeds16:40:29. No GPU stage repeats, cap changes or repairs
are needed. The earlier bounded-check prelaunch failure remains preserved.

Training overrides92,822 actions (**46.411%**, not20% of actions), including
70,933 NOOP and21,889 hops.51of231 episode starts receive guidance.46 completed
guided episodes average15089.674 and all clear a pyramid;177 unguided episodes
average809.887 with24 pyramids, and the last50 unguided average1851.5. Guided
online returns are not unaided competence.12,476 supervised reports contain
1,525,417 repeated label rows, total weight416,970.851 and actor/world expert-loss
sums898.986512/898.986619, max difference1.4e−7. Late weighted cross entropy.1479
is descriptive of changing replay samples, not held-out accuracy or a diagnosis.

Artifacts:`/x/Temp/kindle-qbert-expert-capacity-20261028/full/qbert-12m-48-b16-r16-p20-world-200k-seed4013`.
Selected declaration SHA256
`616019129a983dfe59af77ebfecd85bd6df695d38f296bf7ae8e1f5c5a2b5e97`;
learning review`c6ce278f90f151b3667ff2ae570d9b3b2e7624d2fd8f9558dafc671ce9d7ed23`;
comparison`278cd6b3738d39adc7fe44475c6b323c87f762e25330a28fc99c8dc169a0e813`;
postprocessing`09dbbb8ecc68fc677adddf9290a85e24e87e7c456b17bfa15864ead8a677dd3d`.
No learner or GPU allocation is active after this result; check live state before
further GPU work.


## Action feedback improves the CPU teacher under fixed disturbances

The current expert API forces the teacher action and stores a boolean expert
tag on that executed transition. It cannot label a different learner action;
unguided states receive no supervised targets. The continuation rule also
remembers its proposed destination. A CPU-only wrapper clears movement intent
when the submitted command differs, while preserving board/pixel observations.
This source review identifies missing capability, not the cause of weak learning.

Accepted-command parity reproduces all24,733 retained first8 actions /98,913
frames, output pixels, rewards and boundaries in73.593s. A predeclared5% uniform18
disturbance comparison then scores **13087.500 versus8584.375** for feedback-aware
versus unchanged continuation, both8/8 first pyramids. Only2/8 feedback episodes
reach15000. These are8 reused CPU-reference episodes, not native learning,
held-out confirmation or evidence of accurate labels on learner-visited states.

Unchanged collection/replay take31.937/31.646s for10,315 actions /41,243 frames;
feedback-aware take42.827/44.614s for13,915 /55,644. All proposals, noise choices,
actions, input/output pixels, rewards, frames and boundaries reproduce exactly.
Actual changed commands526/688 differ because trajectories and episode lengths
differ. All10 declared pins and63 preceding-learner pins stay unchanged. No
native module loads or agent is constructed; zero learning updates/GPU work.
CPU service starts16:49:35 UTC/source`c12de77` and succeeds16:53:20,3m45.104s total.

Select an optional independent-label implementation: preserve actual actions for
world transitions, store teacher recommendations separately for masked imitation,
and provide actual-action feedback to the optional Qbert teacher. Legacy forcing
and defaults must remain unchanged. Native implementation, focused validation,
separate packaging and a bounded GPU check remain outstanding; no full learner,
training budget or automatic successor is selected. See the
[protocol and required invariants](../experiments/2026-10-28-cdp-qbert-corrective-labels.md).
Artifacts:`/x/Temp/kindle-qbert-correction-20261028`; review SHA256
`0eab9a3a3f710b0bfac144ad876e3c288d627f35abb2ca3bf82e315a575cea37`.


## Corrective labels pass a bounded native Qbert check

The optional independent-label API keeps actual actions in world transitions
and teacher recommendations in masked expert targets. A5.87s guarded native
check preserves sampling/beliefs/transition replay with different labels and
passes one finite aligned expert update.130 Rust CPU tests,491 packaged Python
tests with zero native constructions, and strict workspace/binding Clippy pass.
Native`447fe50`/SHA35ad0db is separately packaged; legacy forcing/defaults remain.

Fresh4M48/Auto/N8/B8/T16/H15/R16/20%episode continuation/cap1920/expert1/
NOOP.05/world gradients with corrective labels completes8192 actions/32,753
frames/994 updates in61.976s. Frozen1024/4096/.511s has zero updates and no
natural episodes. All36 pins,357 matching initial/invariant frozen tensors,
finite reports,CNN/reward/actor/value changes,full label/input-pixel/actual-action
replay and guards pass. All8192 labels include1511 hops,3607 passive states and
3195 actual/teacher disagreements; actual overrides4585 (55.9692%).994 reports
are supervised; max world/actor expert-loss difference3e−7. Main service exits
17:24:16 UTC on October28; no GPU stage repeats. This is execution and early-cost
evidence, not competence.2.6GiB build cache released, artifacts retained under
`/x/Temp/kindle-corrective-labels-20261028`.

The early25m13s/200k projection motivates a separately selected fresh200k
comparison with the retained4M control, a30-minute training cap and fixed
final/own-initial24-episode frozen evaluation. Labels and feedback change
together on reused4013; no isolated causal or reliability claim. See the
[protocol and receipts](../experiments/2026-10-28-cdp-qbert-corrective-labels.md).


## Corrective labels cost more without improving frozen quality

Fresh corrective labels complete200k aggregate actions /799,682 emulator frames
/24,970 updates in **25m57s**. Frozen **3633.333/17of24 first pyramids** versus own
initial182.292/0 is below the retained4M world-gradient control3670.833/19 in
21m28s. Training costs **20.86% more**; first-pyramid coverage20.292/21 versus
20.625. Both original15k/90% gate criteria fail. Retain the control and stronger
5%/R32/world-off4730.208/23; no extension or confirmation of this completed life.

The recipe keeps4M48/Auto/N8/B8/T16/H15/R16/coefficient1/actor0/full18/original
rates,20%episode continuation/cap1920/expert1/NOOP.05/world gradients. Independent
recommendations cover every learner-visited state, and the teacher receives
actual-action feedback. Those two changes move together on reused4013/cohort;
native/runner/validator identities also differ, so no isolated causal effect or
independent-seed claim follows.

All52 pins,100k recovery,357 matching initial/unchanged frozen tensors, paired
initial semantic traces,24,970 finite reports,parameter changes,complete training
label/input-pixel/actual-action replay,frozen outcome replays/videos,guards,curves,
coverage,exposure and comparison pass. Main service exits18:06:30 UTC; CPU
postprocessing18:07:02 on October28. No GPU stage repeats. A separate CPU figure
caption correction clarifies20% episode probability; original artifacts and
pins remain, and comparison data are byte-identical.

Training uses200k labels,103,712 passive labels and54,396 teacher/action
disagreements.96,288 overrides are48.144% of actions.47 guided completed episodes
average15176.064 with47 pyramids;177 unguided episodes average2129.096 with74.
Guided returns are not unaided competence. Final50k passive wait-label agreement
is54.505%: teacher NOOP frequency79.447%, learner46.077%. This changing-state
command agreement is not held-out label accuracy or proof of a failure cause.
It motivates a separate bounded wait-weight check, not extending this life.

Artifacts:`/x/Temp/kindle-qbert-corrective-full-20261028` and
`/x/Temp/kindle-qbert-passive-review-20261028`. See the
[completed protocol, timing and receipts](../experiments/2026-10-28-cdp-qbert-corrective-labels.md#completed-fresh200k-comparison).


The separate equal-wait-weight/R32 check passes4096 actions/16,381 frames/963
updates in43.238s, then1024 frozen/.495s with zero updates/no natural episodes.
All36 pins,357 matching initial/unchanged frozen tensors,finite aligned expert
losses,equal labeled-row/weight sums,parameter changes,full replay and guards
pass. Service exits18:15:27 UTC on October28; native unchanged, no GPU repeat.
This is execution/early-cost evidence only. A fresh100k/R32/all-weight1 comparison
is separately selected, with roughly the same update count as200k/R16, half the
experience and a30-minute cap. No full allocation is active at this milestone.
See the [wait-labels protocol](../experiments/2026-10-28-cdp-qbert-wait-labels.md).

## Equal wait weights and100k actions save time but lower frozen quality

The fresh R32/all-weight1 recipe completes **100,000 aggregate actions /399,864
emulator frames /24,939 updates in18m58s** (1138.028s). Frozen mean
**2260.417/12of24 first pyramids**, versus own actual initial182.292/0, is below
the corrective200k/R16/NOOP.05 control's3633.333/17. Training time falls26.90%,
but frozen mean falls37.79%. First-pyramid coverage falls20.292→20.125 of21.
Both original mean15000/90%-pyramid criteria fail. Retain the controls; this
completed life has no extension, confirmation or automatic successor.

Wait weight.05→1, ratio16→32 and experience budget200k→100k change together;
the nearly equal24,970/24,939 update counts do not isolate any one change.
Same native447fe50/runner/teacher,4M48/Auto/N8/B8/T16/H15/coefficient1/actor0/
full18/original rates,20%episode continuation/cap1920/expert1/world gradients/
all-state corrective labels. Fresh4013/eval4000004013 reuses development seeds
and cohort, not weights, replay, live belief, games or RNG. No held-out seed claim.
The forced-only4M world control3670.833/19 in21m28s and stronger5%/R32 world-off
4730.208/23 in25m23s remain contextual controls.

All55 declared inputs remain unchanged. All24,939 reports are finite, first
update248/debt0, and CNN/reward/actor/value tensors change. The50k recovery
passes at12,439 updates with357 finite tensors. All357 initial tensors match
both the primary control and bounded check; paired actual-initial evaluations
have identical1140 semantic events. Both frozen arms preserve all357 tensors,
finish24 natural episodes without truncation or updates, and pass full recorded
outcome replay/video. Final17,800 actions/71,165 frames take8.320s; initial8728
/34,875 frames take4.132s. Ordinary frozen logs have no live pixel hashes.
Complete training teacher-label/input-pixel/actual-action replay, host guards,
curves, coverage, exposure and comparison pass. The comparison figure is checked.

There are100k teacher labels:74,530 NOOP/25,470 movement, including52,928 passive
states and16,508 actual/teacher disagreements. Actual overrides47,072 (47.072%)
are distinct from20% episode selection. All24,939 reports are supervised;
2,989,442 labeled replay rows have exactly2,989,442 weight sum, and maximum
world/actor expert-loss difference is8e−7. The22 completed guided episodes
average15110.227/all22 first pyramids;84 unguided episodes average1353.869/20
pyramids (last50:2042.5/20). Guided returns are not unaided competence.

On the last50k training actions, passive teacher/learner NOOP frequencies are
72.193%/76.339%. Wait-label agreement is88.121%, movement agreement33.651%,
overall exact agreement72.974%; weighted replay cross entropy.224 and mean
policy entropy.241. The prior control's last50k had54.505% wait/59.735% movement
agreement. These describe different visited states, experience budgets and replay
distributions; they neither establish label accuracy nor explain the frozen
regression causally. Higher overall agreement is not the task objective.

Main service starts18:29:15 UTC/source`00eed19` and succeeds18:53:59 on October28;
whole service24m44.154s. CPU postprocessing succeeds18:54:18. Original optional
progress observer rejects unsupported arguments; its failed service/journal is
retained and a separately named CPU observer succeeds. Learner identity stays
PID189477/invocation`00d3435a0afe4e2688a0d7c2b43aa1a6`; no GPU stage repeats,
runtime pin changes or cap increases. The protocol records two short affined
CPU Breakout studies as timing caveats, with no wall-time correction.

Artifacts:`/x/Temp/kindle-qbert-wait-full-20261028/qbert-4m-48-r32-wait1-100k-seed4013`;
passive analysis`/x/Temp/kindle-qbert-wait-analysis-20261028`. Learning review
`e191ac5e1a04352869d568476f0724ac98e442208d9944557304c26fa82253ae`;
comparison`32f496a5aced6b5a4cd2ab2589ac106082ea2027a261ce590e2f34d39d47cc9e`;
postprocessing`1e790da08baefa5943ff1672c617c9b29883f8e71e6168e6a049b287d4f360e5`;
completion`c3cd7f5b46c64be183626ab2c720541c3a4ba7f0b18d8c512824d9ce8bb4c10c`;
passive review`23be3fd9d4b80f1053b726d4548cfad9fe09adf47ce2ba4fce9fcbdb42414c67`.

## Greedy actions improve the retained equal-weight policy

With zero new training, the retained100k/R32/all-weight1 policy scores
**4965.625/21of24 first pyramids** under greedy actions, versus its sampled
2260.417/12. Own actual initial greedy scores263.542/0, sampled182.292/0.
The original mean15000/90%-pyramid gate still fails both criteria. Retain greedy
as a useful development option for this18m58s learner; no global default change,
new learner confirmation, extension or automatic successor is selected.

| Frozen action choice | Final mean | First pyramids | Own initial mean |
|---|---:|---:|---:|
|Sampled, original result|2260.417|12/24|182.292|
|Greedy, same retained weights|4965.625|21/24|263.542|

The mean is4.98% above the stronger retained5%/R32/world-off sampled recipe's
4730.208, but pyramid successes are21 versus23. Original training18m58s versus
25m23s is about25% cheaper; this additional frozen assessment costs46.404s as
a whole service. The two training recipes and evaluation choices differ, so
this is a quality/cost comparison, not isolated training attribution. That older
4013 learner's greedy result is3770.833/19. All use reused development cohorts;
this single retained learner is not independent confirmation or broad reliability.

Both arms use unchanged native447fe50/runner/runtime/checkpoint/config and
seed4000004013, with no guidance or labels. Final31,544 actions/126,141 frames
take14.710s; initial10,432/41,697 take4.752s. Total41,976 aggregate actions/
167,838 emulator frames/19.462s evaluation, zero learner updates. Each selects
first3 natural episodes per8 lanes=24; final collects27 and initial25, with
excess episodes and partial tails retained. No truncations. All47 declared pins,
all55 source-training pins,357 finite/invariant tensors per arm, source-file
invariance, full outcome replays/videos and host guards pass. Ordinary frozen
logs lack live pixel hashes; no direct live-image equality claim is made.

Main starts19:02:41 UTC/source`1222ce9`, PID192628/invocation
`da526475b8194663b55c86b9c8a23abd`, and succeeds19:03:28 on October28.
CPU postprocessing succeeds19:03:31. No failed attempt, repeated GPU stage,
runtime change or cap increase. The earlier label-path inspection establishes
no implementation defect; greedy improvement does not diagnose the training
failure or establish teacher-label accuracy.

Artifacts:`/x/Temp/kindle-qbert-wait-greedy-20261028/qbert-4m-48-wait1-100k-greedy-frozen`.
Declaration`ae93f4ca26aee04f35d4f357f48c08e02520faa2220808881a8ab8f4bf5dc341`;
review`fa142ce8a695278f1e2e8136cff3e59a325650fbc11c2164fbd38d7900e5e74e`;
postprocessing`38cbbfc1f33d693c00cd60cd8507a1d066905c8d016977783bd89a1a51bc3a74`;
completion`df9f451190b2d1b041a032cd170685d130ae18e8a5983442948daa2f8221ff8c`.
See the [frozen protocol](../experiments/2026-10-28-cdp-qbert-wait-greedy.md).

## Retained greedy actions disagree more with the teacher after the first pyramid

A CPU replay of that unchanged4965.625/21-of24 policy recomputes corrective
teacher suggestions from native max-pooled RGB and feedback of the recorded
native action. Teacher intent resets on disagreement exactly as in training;
it receives no rewards or RAM. The existing task observer labels decisions
before/after first-pyramid completion retrospectively. Only the original first3
natural episodes per lane enter these counts:

| Decision phase | Actions | Exact agreement | Agreement when teacher waits | Agreement when teacher suggests a hop |
|---|---:|---:|---:|---:|
|Before first pyramid|10,442|74.717%|94.396% (6620/7013)|34.471% (1182/3429)|
|After first pyramid|13,844|62.092%|90.056% (6946/7713)|26.912% (1650/6131)|

Native NOOP counts are8557/10,442 before and10,628/13,844 after, versus teacher
7013 and7713 on the same observed states. Exact command agreement is not teacher
accuracy, action equivalence or a causal diagnosis. Stage/state distributions,
teacher errors and sticky-action timing remain relevant; these observations do
not establish that additional waiting causes the task-performance gap. They
select no policy change or additional GPU allocation.

All31,544 recorded actions/126,141 frames replay, including excess episodes and
tails. Rewards, boundaries, resets, all27 task outcomes and partial outcomes
exactly match the retained replay; the selected24 contain24,286 decisions.
The original frozen log has no live pixel hashes. The new label trace records
reconstructed input hashes, not proof of equality to unavailable live images.
All14 declared pins and84 active expert-scale10 inputs remain unchanged; no
native module loads, agent is constructed or learner update occurs.

The affined low-priority service starts19:29:28 and succeeds19:31:09 UTC on
October28, whole1m41.407s; measured analysis101.232s. The active learner retains
PID193426/invocation`b9d840aed2e34da79c483f9e348d5210`. This is a co-host timing
caveat, not a wall-time correction. Saved labels independently reproduce the
phase totals. No retry or cap change occurs.

Artifacts:`/x/Temp/kindle-qbert-greedy-phases-20261028`.
Declaration`905548a183270780d64d700cd7dd8659d814ddcd1347e5a3b44b66ac3f738c77`;
review`b7d61f78c02572204fc1e253b5b9647b4c2c839dc8f22c5039ea912e7c8940fc`;
completion`45bef23a486a497c5bb2c209235f6d7658b3c618c1da3f09f352a6589347a4ac`.

## Stronger expert supervision improves frozen Qbert at the same100k budget

Increasing expert scale1→10 raises primary greedy frozen score **4965.625→7700**
and first pyramids **21/24→24/24**, a55.07% mean gain. Sampled score also improves
**2260.417→5572.917**, with12/24→24/24 pyramids. The fresh fixed100k actions/
399,860 emulator frames/24,939 updates take **18m55s**, versus control18m58s.
That.25% timing difference is effectively unchanged cost. Both action modes
pass the first-pyramid fraction but still fail the original **mean15000** gate.
Retain scale10/greedy as a useful compute-constrained development recipe on this
one reused learner seed/cohort. No extension, second learner or successor is
selected by this completed comparison; the five-game goal remains unfinished.

| Expert scale | Training | Sampled frozen mean / pyramids | Greedy frozen mean / pyramids |
|---|---:|---:|---:|
|1, retained control|18m58s|2260.417 /12of24|4965.625 /21of24|
|10, fresh candidate|18m55s|5572.917 /24of24|7700.000 /24of24|
|Own actual initial, both scales|Zero updates|182.292 /0of24|263.542 /0of24|

Keep4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor-unimix0/full18/original
rates,20% episode continuation/cap1920, all18 action weights1, expert-world
gradients and corrective labels on all states. Only expert scale changes;
actor and world supervision change together, so gradient-route attribution
is not isolated. Native447fe50/runtime, teacher, seeds and initial tensors are
the same. No weights, replay, belief, games or RNG are reused. Seed4013 and
evaluation4000004013 remain reused development data, not independent confirmation.

Each of four frozen arms selects first3 natural episodes per8 lanes=24, with
no guidance/labels, updates or truncations. Final/initial sampled collect25/24
episodes; greedy collect26/25; excess episodes and partial tails are retained.
The four arms total84,920 actions/339,538 emulator frames/40.315s. All357 tensors
remain unchanged per arm. Every final selected episode reaches21/21 initial
cubes. Ordinary frozen logs lack live pixel hashes; full action/reward/frame/
boundary/reset/outcome replays and videos pass without a direct live-image claim.

All84 declared inputs remain unchanged. The50k recovery passes at12,439 updates;
all357 initial tensors match control and bounded check. Both actual-initial
semantic traces match their controls:1140 sampled events and1354 greedy events.
All24,939 numerical reports are finite, CNN/reward/actor/value tensors change,
and full training teacher-label/input-pixel/actual-action replay, host guards,
curves, coverage, exposure and comparison pass. The comparison PNG is visually
checked. No GPU repeat, cap change, runtime edit or failed attempt occurs.

Training labels comprise74,107 NOOP and25,893 hop recommendations, including
57,853 passive states and15,224 actual/teacher disagreements. The20% episode
rule selects23/101 starts and actually overrides42,147 actions (42.147%), versus
control47.072%; action exposure changes with episode trajectories. All24,939
supervised reports have2,989,603 labeled replay rows and identical weight sum;
maximum raw world/actor expert-loss difference is8e−7. Reported raw losses exclude
the scale10 coefficient. Completed guided episodes average15,551.250/20pyramids;
73 unguided episodes average2538.699/32pyramids (last50:3465.500/31). These changing
online returns are not frozen competence. There are5726 released native actions.

In the final50k actions, passive teacher agreement is78.267% versus control
72.974%; agreement on suggested hops47.203% versus33.652%, and suggested waits
90.672% versus88.121%. Weighted replay cross entropy is.137775 versus.223893;
mean policy entropy.152069 versus.240761. These changing states/replay samples
are descriptive, not held-out teacher accuracy or a causal mechanism.

The main service starts19:16:56 UTC/source`c0abf6a` on October28 and succeeds
19:42:39, whole25m43.379s; CPU postprocessing succeeds19:42:59. Recovery/progress
observers also exit successfully. The protocol retains67.624s Breakout CPU work
and101.232s retained-policy CPU replay as co-host timing caveats. The control has
its own recorded co-host work; no wall-time adjustment is made. No allocation
remains active at completion; check live state before further GPU work.

Artifacts:`/x/Temp/kindle-qbert-expert-scale10-20261028/full/qbert-4m-48-r32-wait1-expert10-100k-seed4013`;
log-only analysis:`/x/Temp/kindle-qbert-expert-scale10-analysis-20261028`.
Learning review`32bc3d5f60c0c84a24fa3f16c4237f492cd7c94c4a79006d992b9a4454cce33f`;
postprocessing`42c9372a2de69f2cbd38c88fd1d390503608627a4773bb686e5789c5b0168c8a`;
comparison`3ba4d8f904c4f956d660c6431c01ffe587d83f1a47923b27a745a508922633bc`;
completion`bf316ed8851a650fde2934d856abe476645289ed61e6f2f530926b43d4c06c94`;
log-only analysis`eb64f866780b0ae5f4557584081a2f6976c55cdc1867d3dc480e7f9057a66096`.
See the [fixed-budget protocol](../experiments/2026-10-28-cdp-qbert-expert-scale10.md).

## Expert10 reaches the first-pyramid fraction on two greedy learners, with large score variation

The fixed-recipe seed1009 check completes **100k actions/399,823 emulator frames/
24,939 updates in19m11s**. Primary greedy frozen score **3832.292/23of24 first
pyramids** improves its own85.417/0 initial, but is much weaker than4013's
**7700/24of24 in18m55s**. Both greedy learners meet the90% first-pyramid fraction;
neither meets the original **mean15000** gate. Retain both development results,
not just the stronger seed. No extension, third learner or automatic successor
is selected; the five-game goal remains unfinished.

| Learner | Training | Sampled final / own initial | Sampled pyramids | Greedy final / own initial | Greedy pyramids |
|---|---:|---:|---:|---:|---:|
|4013|18m55s|5572.917 /182.292|24/24 /0|7700.000 /263.542|24/24 /0|
|1009|19m11s|2667.708 /180.208|15/24 /0|3832.292 /85.417|23/24 /0|

Only learner/environment/evaluation seeds change from the completed expert10
recipe:4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor-unimix0/full18/original
rates,20% episode continuation/cap1920,all18 weights1,world gradients and
corrective labels. Native447fe50,runner and teacher bytes are identical. No
weights,replay,belief,games or RNG are reused. Both seeds/cohorts have earlier
development use. There is no matched100k scale1 control on1009: this supports
usefulness of the recipe on two learners, not a paired expert-scale effect
across seeds or broad reliability. Different seeds need not match trajectories.

All115 declared inputs remain unchanged. The50k recovery passes at12,439 updates;
357 initial tensors are finite and346 shared initial tensors match the retained
legacy1009 control. Its1203-event sampled actual-initial semantic trace matches
exactly. All24,939 reports are finite; CNN/reward/actor/value tensors change.
All357 frozen tensors remain unchanged per arm. Complete training label/input-
pixel/actual-action replay,frozen outcome replays/videos,host guards,curves,
coverage,exposure and two-learner comparison pass. The comparison PNG is visually
checked. No failed attempt,GPU repeat,cap change or runtime edit occurs.

Each of four frozen arms selects first3 natural episodes per8 lanes=24 with no
guidance/labels,updates or truncations. Sampled final collects25 episodes; other
arms collect24 each. Excess episodes and partial tails remain. Evaluations total
61,392 actions/245,415 frames/28.689s. Mean first-pyramid coverage is20.167/21
sampled and20.792/21 greedy, versus own5.833/3.375. Ordinary frozen logs have no
live pixel hashes; outcome replay does not establish direct live-image equality.

The20% episode rule selects22/121 starts and overrides40,745 actions (40.745%).
100k labels include75,051 NOOP/24,949 hops,59,255 passive states and16,871 actual/
teacher disagreements. All24,939 supervised reports have2,988,750 labeled replay
rows and identical weight sum; maximum raw actor/world loss difference8e−7.
20 completed guided episodes average15,295/20pyramids;93 unguided average1717.473/
34pyramids (last50:2945.500/34). There are4471 released native actions. These
online returns include changing policies and teacher exposure, not frozen
competence. The original4013 exposure and fixed-budget scale1 comparison remain.

Main starts19:51:22 UTC/source`447d3c5` on October28 and succeeds20:16:56,
whole25m34.027s; postprocessing succeeds20:17:25. Recovery/progress observers
also succeed. The protocol preserves the affined Breakout CPU co-host work;
no timing correction is made. No allocation remains active at this milestone.

Artifacts:`/x/Temp/kindle-qbert-expert10-seed1009-20261028/qbert-4m-48-r32-wait1-expert10-100k-seed1009`.
Learning review`22ec3a12ccefa0101865f726df696e5e50cb57c17781ddea27eb2997cf1a7b00`;
postprocessing`346be2c6742ce86dc7c238730bf31c6036da735d1dad92ede9cf1e34ed4974c0`;
comparison`fa5b7c0dbc2b800e964cd1f67762554164913f272fd709d09f48d008e35c03ae`;
completion`9a5423a24cf2b2510c43f0dfd821c844cfa91d6ba84f0ecccee65fb830141a3c`.
See the [second-learner protocol](../experiments/2026-10-28-cdp-qbert-expert10-seed1009.md).

Retrospective saved-log analysis finds final50k passive teacher agreement of
77.637% on1009 versus78.267% on4013; hop agreement45.634% versus47.203%,wait
agreement89.573% versus90.672%. Weighted replay CE is.157522 versus.137775.
These aggregate differences describe distinct visited states and replay samples;
they do not establish the cause of the much larger frozen-score difference or
measure held-out teacher accuracy. The affined CPU analysis takes.687s during
the separately selected expert100 life, preserving all111 inputs and its process
identity, with no native/environment construction. Evidence:
`/x/Temp/kindle-qbert-expert10-seed-analysis-20261028`, review
`0ec88bec80f3b9126e220976ddb4b9d47620e8acb58e682a963dfd7a4290403e`.

## Expert100 gives a smaller further gain at the same100k budget

Increasing expert scale10→100 completes **100k actions / 399,886 emulator frames /
24,939 updates in18m59s**. Greedy frozen mean rises **7700→8365.625 (+8.64%)**,
with24/24 first pyramids for both. Sampled mean rises5572.917→7339.583 (+31.70%),
but first-pyramid successes fall24/24→22/24. Both action modes still miss the
original **mean15000** gate, while meeting90% first pyramids. This is a modest
further gain on reused4013, not a confirmed recipe improvement across learners.
Retain both budget points and the separate expert10/1009 result. No extension,
second expert100 learner or automatic successor is selected.

| Expert scale | Training | Greedy final / own initial | Greedy pyramids | Sampled final / own initial | Sampled pyramids |
|---|---:|---:|---:|---:|---:|
|10|18m55s|7700.000 /263.542|24/24 /0|5572.917 /182.292|24/24 /0|
|100|18m59s|8365.625 /263.542|24/24 /0|7339.583 /182.292|22/24 /0|

Only the expert coefficient changes, affecting actor and world supervision
together. Keep4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor-unimix0/full18/
original rates,20% episode continuation/cap1920/all18 action weights1/world
gradients/all-state corrective labels. Native447fe50, runner and teacher bytes
match the control. No weights, replay, belief, games or RNG are reused. Seed4013
and evaluation4000004013 are development data. The3.523s (+.31%) training-time
difference is not a meaningful speed change; prior development costs remain.

All111 declared inputs stay unchanged. The50k recovery passes at12,439 updates;
357 initial tensors match the control and bounded check. Actual-initial semantic
traces match in both modes:1140 sampled events and1354 greedy events. All24,939
reports are finite, CNN/reward/actor/value parameters change, and357 frozen
tensors remain invariant per arm. Full training teacher-label/input-pixel/action
replay, all frozen outcome replays/videos, guards, curves, coverage, exposure and
paired comparison pass. The comparison PNG is visually checked.

Each frozen arm selects first3 natural episodes per8 lanes=24, with guidance and
labels off, zero updates and zero truncations. Greedy initial collects25 natural
episodes; the other arms collect24 each. Excess episodes and partial tails remain
recorded. Four arms total86,104 actions /344,276 frames /39.722s. Mean first-pyramid
coverage is21/21 greedy and20.750/21 sampled, versus own7.667/5.167. Ordinary frozen
logs contain no live pixel hashes; replay does not prove equality to unavailable
live images. No selected episode in either final arm reaches15000.

The20% episode rule selects22/94 starts and actually overrides38,397 actions
(38.397%), versus scale10's42.147%. Labels comprise74,506 NOOP and25,494 hop
recommendations,61,603 passive states and12,771 actual/teacher disagreements.
All24,939 supervised reports have2,989,863 labeled replay rows and identical
weight sum; maximum raw world/actor expert-loss difference is8e−7. Raw reported
losses exclude the coefficient. Seventeen completed guided episodes average
15,297.059/17 pyramids;69 unguided average3774.638/40 pyramids (last50:5094.500/40).
There are5152 released native actions. These online scores are not frozen
competence.

During the final50k actions, passive teacher agreement is84.217% versus scale10's
78.267%; hop agreement56.922% versus47.203%, wait agreement94.953% versus90.672%.
Weighted replay cross entropy is.078477 versus.137775, mean policy entropy.086033
versus.152069. These different visited states and replay samples are descriptive,
not held-out teacher accuracy or a causal explanation of the score gain.

The bounded4096-action check passes in43.170s before full selection. Its initial
CPU preparation fails on a copied receipt-schema field, then is corrected before
any GPU work; original failure/script are retained. No GPU stage repeats, cap
change or runtime edit occurs. Main starts20:29:26 UTC/source`a6ee6da` on October28
and succeeds20:55:18, whole25m51.514s. Postprocessing succeeds20:55:29; both
observers also succeed. The protocol retains.687s affined CPU log analysis during
training as a co-host caveat, without correcting wall time. All services are
inactive at completion. The five-game goal remains unfinished.

Artifacts:`/x/Temp/kindle-qbert-expert-scale100-20261028/full/qbert-4m-48-r32-wait1-expert100-100k-seed4013`;
log analysis:`/x/Temp/kindle-qbert-expert-scale100-analysis-20261028`.
Learning review`24cfd46ad02b66af67f260a142df26418ac0acb2a68afd0169c7f7e4d8de76fb`;
postprocessing`2bb5d52c5031bc6dafa23f3f174cd0ffa8c4e206dc5193d6df7c30afdd4b8b51`;
comparison`67d27a221d61e10c96aad0dbb39baf059977247e7cce8e75a0a8015bde398466`;
completion`60680c92fe500bad048c9a3de2ec9748e45c7b94dccba2e8a1758c86f3ce17a5`;
log analysis`e8249c4abec2d2048f3444b54428d3f3a63854e4985a2cc8d5f20d8e0877bd02`.
See the [expert-scale100 protocol](../experiments/2026-10-28-cdp-qbert-expert-scale100.md).

## More experience at nearly equal updates improves Qbert with higher cost

Fresh **200k actions /799,703 emulator frames /24,970 updates in26m07s** improves
greedy frozen mean **8365.625→9654.167 (+15.40%)**, but costs **37.63% more
training time** than100k/R32. Greedy first pyramids fall24/24→23/24. Sampled mean
rises7339.583→7569.792 (+3.14%) and first pyramids rise22/24→24/24. Both
candidate modes still miss the original mean15000 gate while meeting90% first
pyramids. Retain both measured budget points: higher score with more experience,
and the cheaper100k option. No extension,seed confirmation or automatic successor.

| Experience / replay ratio | Updates | Training | Greedy final / own initial | Greedy pyramids | Sampled final / own initial | Sampled pyramids |
|---|---:|---:|---:|---:|---:|---:|
|100k /32|24,939|18m59s|8365.625 /263.542|24/24 /0|7339.583 /182.292|22/24 /0|
|200k /16|24,970|26m07s|9654.167 /263.542|23/24 /0|7569.792 /182.292|24/24 /0|

Budget and replay ratio change together at nearly equal update counts, not equal
wall time; no isolated budget or ratio effect follows. Keep4M48/Auto/N8/B8/T16/H15/
coefficient1/actor-unimix0/full18/original rates,20% episode continuation/cap1920/
expert100/all18 action weights1/world gradients/all-state corrective labels.
Native447fe50,runner and teacher bytes match the control. No weights,replay,belief,
games or RNG are reused. Seed4013 and evaluation4000004013 are reused development
data, not independent confirmation. Earlier development costs remain explicit.

All138 declared inputs remain unchanged. The100k recovery passes at12,470 updates;
357 initial tensors match the control and bounded check. Sampled and greedy
actual-initial semantic traces match their controls. All24,970 reports are finite,
CNN/reward/actor/value change, and357 frozen tensors remain invariant per arm.
Full training teacher-label/input-pixel/action replay,all four frozen outcome
replays/videos,host guards,curves,coverage,exposure and paired comparison pass.
The comparison PNG is visually inspected before publication.

Each frozen arm selects first3 natural episodes per8 lanes=24,with guidance and
labels off,zero updates and zero truncations. Greedy initial collects25 episodes;
the other arms collect24 each. Excess episodes and partial tails remain recorded.
Four arms total91,168 actions /364,528 frames /42.866s.
Mean first-pyramid coverage is20.750/21 greedy
and21.000/21 sampled,versus own7.667/5.167.
One selected sampled episode reaches15000; no selected greedy episode does
(maximum14975). Ordinary frozen logs lack live pixel hashes; replay does not
prove equality to unavailable live images.

The20% episode rule selects44/195 starts and actually overrides
82,765 actions (41.383%). Labels comprise150,350 NOOP and
49,650 hop recommendations,117,235 passive states and
23,447 actual/teacher disagreements. All24,970 supervised reports
have2,993,352 labeled replay rows and identical weight sum; maximum raw
world/actor expert-loss difference is8e-07. Raw losses exclude the coefficient.
Completed guided episodes:43,mean15321.512,43 first pyramids;
unguided:144,mean3052.431,74 first pyramids.
There are12,290 released native actions. These online scores are not frozen competence.

The8192-action bounded check passes in60.947s before full selection. Full service
starts21:02:45 UTC/source`0d73300` and succeeds21:41:05,postprocessing21:41:19
on October28. All observers succeed and all services are inactive. No failed
attempt,GPU repetition,cap increase or runtime edit occurs. During training,
affined CPU-only Breakout tests/reference replay/feedback replay take1.734s,
30.982s and108.206s; all138 active inputs and the same learner process remain
unchanged. Retain these co-host timing caveats without correcting wall time.
The five-game goal remains unfinished.

Artifacts:`/x/Temp/kindle-qbert-expert100-r16-20261028/full/qbert-4m-48-r16-wait1-expert100-200k-seed4013`.
Learning review`53ed3c806a096a7c00444a4352ed4f3329007d480f3c8a66c85a0955e74eb926`;
postprocessing`1b2738dda73547df352a0ceb5754329c240ca2949086f8d18b44a17431805991`;
comparison`b188cecda9ddc2b1db672dce55079f8d52a2ec806fe32e9b7b34b2fca015eaf5`;
completion`451bb3b1f140dca1766f07688e92e6fe5bc25bf819a98cd739c263a538ea8f39`.
See the [expert100/R16 protocol](../experiments/2026-10-28-cdp-qbert-expert100-r16.md).

## Verified training labels retain a larger disagreement after the first pyramid

A CPU-only recount of the completed expert100 runs uses their already verified
training labels and first-pyramid frame markers. It reconstructs episode-local
frame positions, assigning each decision to the phase of its input state before
the submitted action. No environment, native module, inference or learning is
run. Agreement denominators include only passive actions; forced actions remain
counted separately. These are the last50k aggregate actions of each life:

| Recipe | First-pyramid phase | Passive actions | Exact agreement | Agreement on teacher waits | Agreement on teacher hops |
|---|---|---:|---:|---:|---:|
|100k/R32|Before|8,112|95.994%|98.675%|86.919% (1608/1850)|
|100k/R32|After|25,127|80.415%|93.629%|49.555% (3733/7533)|
|200k/R16|Before|9,133|98.095%|99.393%|93.610% (1919/2050)|
|200k/R16|After|25,375|83.732%|94.372%|56.399% (4010/7110)|

The longer run agrees more in both phases on its own visited states, while
post-pyramid hop disagreement remains much larger. This helps locate remaining
policy/teacher disagreement for development; it does not identify a capacity,
resolution, memory, optimization or teacher error as the cause. Training stages
and state distributions differ. These counts are neither frozen-policy agreement,
teacher accuracy, action equivalence nor independent seed evidence. They select
no additional GPU allocation by themselves.

All300k recorded actions/1,199,589 frames are accounted for. Every completed and
partial episode frame boundary matches the verified replay; full teacher-label,
passive-action, disagreement and override totals match the existing learning
reviews. All9 analysis inputs and64 active Breakout inputs remain unchanged,
with the same learner PID207148/invocation`f08ef9cc0f4c41f2b54af5a619dd7e5b`.
The affined low-priority analysis takes1.131s during Breakout training; retain
that co-host duration without adjusting training wall time. No failed attempt
or GPU stage occurs in this analysis.

Artifacts:`/x/Temp/kindle-qbert-supervision-phases-20261028`.
Review`e5569a190729d8a5a06ca7e24c5f9b2cc98676d4a9dd5d7d3031a4d1213efbf1`.

## Larger strongly supervised Qbert passes the bounded check

A fresh12M48/Auto/N8/B16/T16/H15/R16/expert100 check completes8192 actions/
32,754 frames/489 updates in69.886s.1024 frozen actions/4096 frames take.788s,
zero updates/no natural episodes. All44 inputs,357 matching retained12M initial/
invariant frozen tensors,finite reports,changed CNN/reward/actor/value,full
teacher-label/input-pixel/action replay and guards pass. Main succeeds22:54:14
UTC on October28,whole1m45.686s;no GPU repeat or failed attempt occurs.

Native447fe50,runner and Qbert teacher match the strong4M control. Capacity4M→12M
and batch/microbatch8→16 change together,halving optimizer cadence atR16;no isolated
capacity effect or task competence follows. All489 reports are supervised;
117,261 labeled rows equal weight sums.4341actual overrides differ from20%
episode probability;3851 passive labels/3425 action disagreements replay exactly.
The early200k projection28m26s supports a separately selected fresh200k comparison
under a35-minute cap,with fixed final/own-initial sampled and greedy assessments.
The original mean15000/90%-pyramid gate remains. No bounded-check experience is
reused. [Protocol and receipts](../experiments/2026-10-28-cdp-qbert-expert100-capacity.md).

## Larger strongly supervised Qbert improves greedy score with higher cost

Fresh12M48/B16/R16/expert100 completes **200k actions /799,742 emulator frames /
12,477 updates in30m27s**. Greedy frozen mean **11,984.375 /24of24 first pyramids**
improves4M/B8's9654.167/23 by24.14%,for16.57% more training time. Sampled mean
falls10.71%,7569.792→6759.375,with24/24 pyramids on both. Own actual initials are
80.208 greedy and177.083 sampled,zero pyramids. Both trained modes still miss
the original mean15000 gate. Retain the larger greedy recipe as a promising
development budget point; the4M recipe remains cheaper and stronger under
sampled inference. No extension,seed confirmation or automatic successor is
selected at completion.

| Recipe | Actions / updates | Training | Greedy final / initial | Sampled final / initial |
|---|---:|---:|---:|---:|
|4M48/B8/R16/expert100|200k /24,970|26m07s|9654.167 /263.542|7569.792 /182.292|
|12M48/B16/R16/expert100|200k /12,477|30m27s|11984.375 /80.208|6759.375 /177.083|

Capacity4M→12M and batch/microbatch8→16 change together;optimizer cadence halves
and warmup spans more actions. This does not isolate capacity or match update/
wall budgets. KeepRGB48/Auto/N8/T16/H15/coefficient1/actor-unimix0/full18/original
rates,20%episode continuation/cap1920/expert100/all18 weights1/world gradients/
all-state corrective labels. Native447fe50,runner and Qbert teacher match the
control. Newer Python audit/Breakout modules and current replay are explicitly
pinned;historical control paths resolve through recorded archives. No weights,
replay,belief,games,RNG or bounded-check experience are reused. Seed4013 and
evaluation4000004013 are reused development inputs,not independent confirmation.

All190 inputs remain unchanged. The100k recovery passes at6227 updates. All357
initial tensors match the retained12M reference and bounded check;initial traces
are recorded separately across architectures,without claiming4M/12M equality.
All12,477 reports are finite,CNN/reward/actor/value change,and357 tensors remain
invariant in each frozen arm. Complete training label/input-pixel/action replay,
four frozen outcome replays/videos,host guards,curves,coverage,exposure and the
comparison pass. The comparison plot is visually inspected. Each frozen arm
selects/collects24 natural episodes,first3 per8 lanes,with partial tails retained,
zero updates/truncations and no guidance. Four arms total98,664 actions /394,520
frames /59.634s. Ordinary frozen logs lack live pixel hashes.

Episode guidance selects45/198 starts but actually overrides85,916 actions
(42.958%). Labels include151,616 NOOP and48,384 hops,114,084 passive states and
24,319 actual/teacher disagreements. All12,477 reports are supervised;2,991,282
labeled replay rows equal weight sums,max raw world/actor expert-loss difference
8e-7. The44 completed guided episodes average15,184.091 with44 first pyramids;
146 unguided online episodes average2509.932 with64. There are10,642 native
actions after a selected episode reaches its cap. Guided/released outcomes are
not unaided frozen competence. Replay cross entropy and these changing online
samples do not establish a mechanism or held-out action accuracy.

Main starts22:57:47 UTC/sourcecb4ab40 and succeeds23:40:33 on October28,
whole42m46.186s. CPU postprocessing succeeds23:40:50,whole43m03.375s;all four
services are terminal success. No failed attempt,GPU repeat,cap change or runtime
edit occurs. CPU-only Breakout preparation/preflight takes.632s/.672s during
training with the same learner process and190 pins;retain the co-host caveat
without correcting wall time. Training time excludes initialization,evaluation
and CPU audits. The device is idle at completion;the five-game goal is unfinished.

Root`/x/Temp/kindle-qbert-expert100-capacity-20261028/full`,trial
`qbert-12m-48-b16-r16-expert100-200k-seed4013`. Completion
`0050b01138057006f4f35f02c13ea9c0c644ff076400f03f4b4cf48c2e5e8a70`,
learning`e84006962525e7616052aa75e970176b28fe63e65704dcedc49f79369e1584e1`,
comparison`5d816b773b39daae47b31005364e7ffb836f409559b0d4ca0dfd60d1e5c3b3d6`.
See the [fixed protocol and receipts](../experiments/2026-10-28-cdp-qbert-expert100-capacity.md).


## More experience at R8 costs more and lowers primary greedy quality

Fresh12M48/B16/R8/expert100 completes **400k actions /1,599,459 emulator frames /
12,489 updates in40m35s**, within its45-minute cap. Primary greedy frozen mean
**10,042.708 /23of24 first pyramids** is16.20% below the retained200k/R16
control's11,984.375/24,for33.28% more training time. Sampled mean improves10.31%,
6759.375→7456.250,but first pyramids fall24→22. Both modes miss the unchanged
mean15000 gate. Retain the shorter200k/R16 recipe;no extension,seed confirmation,
nearby ratio retry or automatic successor is selected.

| Fresh recipe | Actions / updates | Training | Greedy final / initial | Sampled final / initial |
|---|---:|---:|---:|---:|
|12M48/B16/R16/expert100|200k /12,477|30m27s|11984.375 /80.208|6759.375 /177.083|
|12M48/B16/R8/expert100|400k /12,489|40m35s|10042.708 /80.208|7456.250 /177.083|

Experience doubles and replay ratio halves together, preserving similar update
counts while changing optimizer cadence,warmup in actions and visited experience.
This does not isolate ratio or experience or match wall cost. Keep Auto/N8/T16/
H15/coefficient1/actor-unimix0/full18/original rates,20% episode continuation/
cap1920/expert100/all18 weights1/world gradients/all-state corrective labels.
Native447fe50,runner,teacher and other saved config remain unchanged. No previous
weights,replay,belief,games,RNG or bounded-check experience are reused. Seed4013
and evaluation4000004013 are reused development inputs,not independent confirmation.

All228 inputs remain unchanged. Scheduled recoveries100k/3114,200k/6239 and
300k/9364 pass. All357 initial tensors match control and bounded check;the1139
sampled/1101 greedy initial semantic events match their corresponding control
exactly. All12,489 reports are finite,CNN/reward/actor/value change,and357 tensors
remain invariant in each frozen arm. Full training label/input-pixel/action
replay,four frozen outcome replays/videos,host guards,curves,coverage,exposure
and comparison pass. Both plots are visually inspected. The higher final online
curve includes teacher actions and does not overturn the lower frozen result.

Each frozen arm selects first3 natural episodes per8 lanes=24,with guidance off,
zero updates and zero truncations. Each final arm collects25 natural episodes;
each initial collects24,with excess episodes and partial tails retained. Four
arms total104,512 actions /417,903 frames /63.134s. Mean first-pyramid coverage
is20.958/21 greedy and20.917/21 sampled,versus own2.500/4.917. One selected greedy
episode reaches15000;none sampled does. Ordinary frozen logs lack live pixel
hashes;their action/reward/frame/boundary replays cannot prove equality to
unavailable live images.

The20% episode rule selects85/388 starts and actually overrides161,639 actions
(40.40975%). Labels comprise301,367 NOOP and98,633 hops,238,361 passive states
and48,759 actual/teacher disagreements. All12,489 reports are supervised;
2,994,362 labeled replay rows equal weight sums,max raw world/actor loss
difference7e-7. The82 completed guided episodes average15,268.598 with82 first
pyramids;298 unguided online episodes average3022.567 with147. There are23,487
native actions after a selected episode reaches its cap. These changing online
samples are not unaided frozen competence,teacher accuracy or a causal diagnosis.

Main starts01:12:47 UTC/source90a9475 and succeeds02:16:04 on October29,
whole1h03m17.041s;CPU postprocessing succeeds02:16:22,whole1h03m34.982s.
Training time excludes construction,evaluation and CPU audits;the full training
replay alone takes1222.256s. Recovery and replacement progress observers succeed.
The original progress observer exits2 at launch on unsupported`--seconds`;its
script,journal and failure are preserved. The private replacement and repaired
collector pass without changing any228 pins or repeating a GPU stage. No cap
changes. CPU observer repair.285s and retained Breakout analysis1.953s co-host
with training;retain those timing caveats without correcting measured wall time.
The device is idle at completion;the five-game goal remains unfinished.

Root`/x/Temp/kindle-qbert-expert100-r8-20261029/full`,trial
`qbert-12m-48-b16-r8-expert100-400k-seed4013`. Completion
`1f3eba07f3be98b31e1557d2e22573d3176cecc1dc6d8d9f291c2e05f43bf1c6`,
learning`baff95ce8d30350eeb15626b8a6664dc4e419c5bd50fe09384301e6c3ea2ae02`,
comparison`dc4ad5388433147cc5c12fcc041f890b9d2cd3dc064adb5c69f5d21b8b1f4b42`,
training replay`7175a47255f52be777c6840529ec6d9894d046e9587875af1030679e05f43054`.
See the [fixed protocol and preserved observer failure](../experiments/2026-10-29-cdp-qbert-expert100-r8.md).


## Stronger imitation lowers both frozen scores at the same short budget

Fresh 12M48/B16/R16 with expert scale1000 completes **200,000 actions / 799,697
emulator frames / 12,477 updates in30m31s**, within the35-minute cap. Primary
greedy mean falls **11.65%**, and sampled mean falls **9.02%**, relative to the
retained scale100 control. Training costs0.21% more. Both modes miss the unchanged
mean15000 gate; sampled also misses90% first-pyramid success. Retain scale100.
No extension, coefficient retry, checkpoint selection or automatic successor
is selected.

| Expert scale | Training | Greedy mean / first pyramids | Sampled mean / first pyramids |
| --- | ---: | ---: | ---: |
|100, retained|30m27s|11984.375 /24of24|6759.375 /24of24|
|1000|30m31s|10588.542 /24of24|6150.000 /21of24|
|Own initial, both recipes|Zero updates|80.208 /0of24|177.083 /0of24|

Only the saved expert scale changes100→1000. Keep centered CDP,12M48/Auto/N8/
B16/T16/H15/R16/coefficient1/actor-unimix0/full18/original rates,20% episode
continuation guidance/cap1920/all18 weights1/world gradients/all-state labels.
Native447fe50, runner and teacher stay fixed. The coefficient affects actor and
world supervision together. No check experience, retained weights, replay,
belief, games or RNG are reused. Seed4013/evaluation4000004013 are reused
development inputs; this comparison does not establish seed reliability or the
mechanism of the regression.

All229 inputs remain unchanged. Recovery100k/6227 passes;357 initial tensors
match the control and bounded check. All1139 sampled and1101 greedy initial
semantic events match their respective controls. All12,477 reports are finite,
CNN/reward/actor/value parameters change, and357 tensors remain invariant in
each frozen arm. Complete training label/input-pixel/action replay, four frozen
outcome replays/videos, host guards, curves, coverage, exposure, comparison and
the original terminal collector pass. Both plots are visually inspected.

Each frozen arm selects the first3 natural episodes per8 lanes=24, with guidance
off, zero updates and no truncations. Final greedy/sample collect27/25 episodes;
each initial collects24. Excess episodes and partial tails remain recorded.
The four arms total88,432 actions /353,575 frames /53.646s. Mean first-pyramid
coverage is21/21 greedy and20.833/21 sampled, versus own2.500 and4.917. Ordinary
frozen logs lack live pixel hashes, so outcome replay does not establish equality
to unavailable live images.

The20% episode rule selects47/207 starts and actually overrides87,605 actions
(43.8025%). Labels comprise152,140 NOOP and47,860 hops, including112,395 passive
states and23,761 actual/teacher disagreements. All12,477 reports are supervised;
2,991,252 labeled replay rows equal weight sums, with maximum raw world/actor
loss difference8e-7. The45 completed guided episodes average15,210 with45 first
pyramids;154 unguided online episodes average2314.286 with64. There are10,022
native actions after guidance caps. These online outcomes are not unaided frozen
competence. Final25k replay cross entropy is.098881, versus control.096632;
changing replay samples do not measure held-out action accuracy.

Main starts02:31:35 UTC/sourcecff4772 and succeeds **03:14:19 UTC on October29**,
whole42m44.356s. CPU postprocessing succeeds **03:14:39**, whole43m03.805s;
all four services finish successfully. Training time excludes construction,
evaluation and audits; full training replay takes611.664s. No observer or GPU
failure, repeated GPU stage, cap change or runtime edit occurs. The.668s
read-only numerical monitor co-hosted with training; retain that timing caveat
without correcting measured wall time. The GPU is idle at completion and the
five-game goal remains unfinished.

Root `/x/Temp/kindle-qbert-expert1000-20261029/full`, trial
`qbert-12m-48-b16-r16-expert1000-200k-seed4013`. Completion
`e5436127858cfd1ffaa8d1f27826f0e8c82e2ece6057ffc9a223788c9869e552`,
learning `2fcb0d85b24044ea0af8a61478d3ee290b94463b35a01f8ffba775e82b3fb069`,
comparison `2243636ff74833b2813b0ad4e3f0ebdb0c17ef3bbf1e03c5bf88e753929e0f47`,
training replay `824aa035c01304fa3479124b7b474db0f5c5cc9ca4a281ba286a57acab26538a`.
See the [fixed protocol](../experiments/2026-10-29-cdp-qbert-expert1000.md).


## Retained Qbert episodes end soon after guidance release

A CPU-only recount of the two completed12M200k learners separates the1920-action
guided prefix from subsequent native actions. All44 completed guided episodes
at expert100, and all45 at expert1000, reach the cap. The retained expert100
episodes all terminate within512 further actions; their mean tail is241.864
actions and182.386 reward, after a guided-prefix mean15001.705. Expert1000 tails
average222.711 actions and195 reward;44of45 terminate within512 actions, all45
within1024. Prefix mean there is15015.000.

| Retained training log | Expert100 | Expert1000 |
| --- | ---: | ---: |
|Completed guided episodes reaching cap|44/44|45/45|
|Median actions after release|264.5|174|
|All released native actions, including tails|10,642 (5.321%)|10,022 (5.011%)|
|Exact agreement on teacher hop labels after release|24.021%|21.091%|

These observations motivate considering later-game exposure, but do not show
that the cap causes the frozen-score gap. Release occurs at a particular game
phase, and the policy changes throughout training. Teacher disagreement does
not establish action incorrectness, teacher correctness or held-out accuracy.
The guided-prefix scores are not unaided competence. No new learner, cap change,
extension or automatic successor is selected by this analysis.

Both complete200k logs reconcile with their verified replay, episode returns,
lengths, labels and prior exposure totals. No environment, native agent or GPU
work is constructed. The.894s affined CPU analysis co-hosts with the active
Breakout expert100 learner; all69 pins and PID235051/invocation
e8a6e8c36aa241f294f9b845645e770b remain unchanged. Retain that timing caveat
without correcting Breakout's measured training time. Evidence:
`/x/Temp/kindle-qbert-cap-release-20261029/{analyze.py,review.json}`; review
`cc353f89b15aa95847784280710df080fe006fac2e29673a2fe42f90297f1171`.
