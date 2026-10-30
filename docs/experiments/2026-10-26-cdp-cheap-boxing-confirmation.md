# Confirm the cheap Boxing recipe on one new seed

The selected seed2017 recipe reaches76.8333 with24/24 frozen wins in28m22s;
[all audits pass](../results/2026-10-25-cdp-cheap-boxing.md). Test one additional
fresh learner, seed4013, with the recipe fixed before launch. This is a focused
confirmation, not a broad seed campaign or a warm-start extension. Start only
after reviewing the active Breakout screen.

Keep Size1M/RGB32/Auto (834,673 parameters), centered CDP/action-effects1,
N8/B8/T16/H15/R32, full BPTT/microbatch8 and replay100k. Full18/sticky.25/repeat4,
native max-pooled frames, one GPU resize, no action aid. All learner settings,
native runtime and experience/update budgets match2017; only learner and frozen
evaluation seeds change. Audit the full configuration against the successful run.

Fixed500k aggregate actions, expected124939 updates, first update248, zero debt.
Training cap1800 seconds including construction. Save actual initial weights
and250k recovery weights, replacing them with fixed500k final weights. A cutoff
retains the partial checkpoint and is reported as such, without a cap increase,
resume, favorable-checkpoint selection or automatic extension.

Frozen seed4000004013, first three natural sampled matches per lane,24 per arm,
zero updates,600k-action/300-second caps per arm. Keep the existing mean≥50,
at least90% natural wins, at least20 natural matches and zero-truncation gate.
Compare final to its own actual initial control; report both learner seeds
separately. Episodes within a learner do not become independent training seeds.

Audit every numeric report, complete config/source/counter identity and finite
checkpoints; verify all346 frozen tensor equalities per arm and full CPU action
replays with stream0 videos. Audit cap1200 seconds; service cap3660 seconds.
The previous28m22s training time is a measured reference, not guaranteed completion
for this seed. No new backend or precision work is part of this allocation.

Artifacts: `/x/Temp/kindle-cheap-20261025/boxing-1m-32-500k-r32-seed4013`.
Controller: `run_boxing_500k_r32_4013.py`; intended service owner:
`kindle-cheap-boxing-r32-500k-4013-20261026`. Exact commands, hashes, caps and the
seed2017 review hash are recorded before launch. The seed is selected for this
confirmation; no held-out benchmark claim. Even two successful learners would
leave broader seed reliability and the other four games' gaps open.
