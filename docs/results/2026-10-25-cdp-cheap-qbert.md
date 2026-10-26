# Small-image Qbert gains score and coverage but completes no pyramid

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
[prepared training-aid candidate](../experiments/2026-10-26-cdp-qbert-persistent-aid.md)
is not selected; active Pong remains the only learner.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-persistent-random-reference`;
review SHA256`b916cc639c11d38bd5a7f0eafa9ef4ad0569d215a21e2fbeb91891fc1c4f92e1`;
declaration`4f01f70b87669577fc19833e175988aa59c607bb1b4facb17782b48562c0fc2f`.
Script:`qbert_persistent_reference.py`; traces and complete episode records stay
outside Git. No reference success or five-game reliability is claimed.
