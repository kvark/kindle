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
