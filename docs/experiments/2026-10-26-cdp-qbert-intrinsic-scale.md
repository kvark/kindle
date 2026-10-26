# Test a stronger existing Qbert action-effects bonus

The4M/RGB48/R32 control scores809.375 versus initial180.2083 after200k actions,
but completes0/24 first pyramids. CPU replay finds incomplete per-episode cube
coverage. Its [recorded intrinsic bonus](../results/2026-10-25-cdp-cheap-qbert.md#scale-of-the-existing-exploration-bonus)
is only0.146% of predicted extrinsic reward late in training. Test coefficient
1→100 in one fresh life; this is an unproved exploration hypothesis, not an
unchanged extension or a diagnosis of the failure's cause.

Holding recorded predictions fixed,100 would make that late ratio14.58%, but
also8.56 times extrinsic reward over the first1000 updates. New training will
change those quantities. Preserve the existing constant coefficient throughout
the test; no schedule, new exploration algorithm or backend work is introduced.

Keep seed1009/evaluation4000001009, centered CDP, Size4M/RGB48/Auto,
3,578,225 parameters, N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k,
actor_unimix0, full18/sticky.25/repeat4, native max-pooled frames and one GPU
resize. No external action aid, detector, RAM input or task reward shaping.
The existing native ensemble supplies actor/critic bonuses during learning.
Audit the complete native configuration against the control, allowing only
intrinsic_reward_scale to differ. All346 actual initial tensors must match;
retain this life's own actual-initial frozen control.

Fixed200k aggregate actions, approximately800k emulator frames, expected49,939
updates, first update248, zero debt. Training cap1800 seconds including
construction. The completed control took1184.523s (19m45s); that is a cost
reference, not guaranteed runtime or learning. Save100k recovery weights then
fixed200k final; the earlier control saved only200k. Preserve cutoffs and the
latest checkpoint, without cap increases, automatic resume or best selection.

Frozen final/actual-initial evaluation uses the first three natural sampled
episodes per lane,24 per arm, zero updates and no action aid. Each arm has
600k-action/300-second caps. Retain excess episodes and tails, and require zero
truncations for a completed test. Keep the existing task gate: at least20 natural
episodes, mean≥15000 and≥90% first-pyramid completion. Also report score, pyramid
count and coverage as development outcomes; improvement does not lower the gate.

Audit every numeric report, full source/config/counters, finite checkpoints,
all346 unchanged frozen tensors, full CPU action replays/videos and host guards.
Use the existing post-hoc task observer without feeding its state into learning.
Audit cap1200 seconds; service cap3660 seconds. Stop on real faults or numerical
failures. No automatic extension follows a positive score or finite losses.

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-intrinsic100-200k`.
Controller: `run_qbert_intrinsic100_200k.py`; intended service:
`kindle-cheap-qbert-intrinsic100-200k-20261026`.
Native source`bfb8c31` is unchanged. Current CLI`bcef410` versus control`a99416e`
adds the actor-unimix option; its default remains0. Verify runner identity and
the entire native recipe. Exact commands, caps and source/control/analysis hashes
are prepared; the existing CPU recipe check passes with coefficient100.

This is a prepared candidate, not an allocated learner. Review Pong and the
focused Freeway confirmation before selecting another allocation. Seed1009 is
reused development data. A useful outcome requires focused seed confirmation;
no broad queue, held-out benchmark result or five-game reliability claim follows.
