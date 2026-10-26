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
Retain this result without extending it. The next bounded allocation tests
Breakout4M/RGB64/T16, preserving its small-ball detail.
