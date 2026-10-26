# Small-image Qbert gains score and coverage but completes no pyramid

The fixed200k screen learns in19m45s: frozen809.375 versus actual initial180.2083.
It reaches more of the first pyramid, but neither policy completes one. The
existing15k/90%-first-pyramid gate fails; this is useful development progress,
not successful Qbert play or seed reliability.

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
with the model, geometry, native runtime, seed and200k budget fixed. It has not
started. The analysis constructs no agent and takes0.513s of CPU wall time.
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
reported separately; the stronger-bonus candidate is a fresh controlled test.
