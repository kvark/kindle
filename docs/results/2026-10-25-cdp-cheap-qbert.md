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
evaluation. **No imitation implementation or allocation is selected.** The
prepared reward-forecast check remains unselected while Pong trains.

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
