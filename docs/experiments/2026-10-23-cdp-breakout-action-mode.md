# Breakout: bounded frozen action-mode check

More warm-start training did not improve the source98.21 score:4M-total80.17,
8M-total85.17. Existing forecast evidence is useful but incomplete. Check whether
policy sampling contributes before another training or representation change.

Restore exactly source2M, warm-final8M and actual-initial models. Use the existing
`--greedy` flag; change no model/config, preprocessing, action space, evaluation
seed4000001009, eight streams or qualified native c8aeba28 implementation.
Compare each against its retained sampled evaluation. Greedy changes action
selection, not ALE stickiness or latent sampling; it is not deterministic play.

Three natural rounds per stream,600k-action/1800s cap per model. Preserve all
excess episodes/cutoffs/tails. If the cap prevents a complete natural cohort,
report it incomplete with no comparable mean; never extend or select cutoffs.
Keep sampled results as the benchmark, never promote a best mode/seed.
Report reward scores and two-wall864-point successes;381.81 remains the reference.

Require zero learner updates, exact346 tensors/model, same runtime/config/source
identities apart from action mode, full CPU action/reward/frame replay and whole
videos. One lineage is not seed replication. A gain would motivate a targeted
policy-learning question; no gain motivates examining control state/credit.
Neither result automatically allocates an entropy sweep or backend project.

GPU workers12GiB/no swap/no CPU quota; CPU audits one CPU/2GiB/no swap.
Host-only guard, current boot/driver fixed, allocation warnings recorded,
genuine faults/deadlines stop. No NVML polling/recovery or native rebuild.
Service cap3h includes all three hard caps and CPU replays; expected a few minutes.

Service: `kindle-cdp-breakout-greedy-20261023`.
Artifacts: `runs/cdp-breakout-greedy-20261023.C20j01e3`.
[Controller](../../runs/cdp-breakout-greedy-20261023.C20j01e3/run.py).
