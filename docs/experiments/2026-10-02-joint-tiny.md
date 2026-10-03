# Joint causal Tiny: task-adaptive latent prediction

The user explicitly authorizes this experiment on October 2. It precedes
exploration/reward, without reopening the completed Phase 2 allocation.

**Current boundary:** isolated native Tiny gradients, regularizer and causal
cache refresh pass after the [attention backward fix](../results/2026-10-03-joint-tiny-backward.md).
Full update/restore/cost qualification passes. The six-run early learning screen
started October3 at07:28 UTC; results remain pending. Both implementation PRs
pass CI (Kindle260, Meganeura856). The active status dashboard remains PR31.
The [initial qualification stop](../results/2026-10-02-joint-tiny-qualification.md)
is retained. October 3's explicitly authorized
[initialization diagnostics](../results/2026-10-03-allocation-initialization.md)
supersede that blanket stop, not the fault/review rules. No recovery implied.

## Question

Does letting world-model/task losses update the pretrained Tiny encoder improve
learning over the same encoder frozen? Both historical Tiny arms were frozen.
Their negative result does not test this question.

Use the same pretrained 5.49M Tiny (`7fe9b252`), native-detail preprocessing,
JL64/pool2 output and action-conditioned latent prediction. Train all used
encoder weights, not an adapter marketed as encoder fine-tuning. Keep reward,
continuation and replay-value supervision; the actor remains separately trained.
No RGB reconstruction, privileged policy inputs, action aids or intrinsic reward.
Control and treatment share the new replay/interaction recipe. Historical RGB
and frozen-Tiny curves remain context, not matched new-backend controls.

This is task-adaptive representation learning, **not direct actor-to-encoder
autodiff**. The pinned upstream Dreamer also defaults to `ac_grads: False`.
Its optional `ac_grads` path allows actor/imagined-value gradients through the
initial posterior state only; future imagined states stay detached. That is a
separate, clean follow-up if direct policy supervision is required, not a change
to the running comparison. Do not equate either nonzero task gradients or this
early screen with proof that the latent state is sufficient for control.

## Qualification before gameplay

1. Adopt merged Meganeura `6268ea5` / Blade `e349cddf`, including main's optimizer
   fix, plus the now-qualified dV alias fix `13b19d33` (PR223).
   Check optimizer/reference tests before learning. Preserve adjacent dirty
   backend worktrees; do not pull their unmerged experimental changes.
2. Review the retained allocation warning and strengthen the host-only guard.
   No recovery, external-memory retry or NVML polling is implied by this task.
   A new warning/native failure stops work for review.
3. Check dense causal Tiny against the independent streaming reference. Verify
   nonzero task gradients and actual parameter updates, stopped prediction
   targets, independent streams, causal context and reset/chunk boundaries.
4. Replay must retain native-detail pixels and re-encode with current weights.
   Refresh acting KV history after encoder updates. Do not reuse stale causal
   caches or reset the RSSM at an encoder chunk boundary.
5. Check representation collapse explicitly and retain a declared stabilizing
   objective. Measure a complete update, including visual backward, replay,
   cache refresh, memory headroom and optimizer work, before setting run length.

The first implementation uses complete phase-zero 16-frame encoder chunks for
replay. Chunks never cross an encoder/environment reset; their RSSM context is
retained, not reset at chunk boundaries. Both arms use this sampling scheme.
Pixels are retained as GPU-preprocessed F32 patches without a second resize or
lossy quantization. Capacity must therefore be set from the full memory cost.

The online stabilizer is **VICReg-style variance/covariance**, not SIGReg or a
claim to reproduce LeVJEPA pretraining. On the 64 spatially averaged feature
channels, minimize mean ReLU(1 - sqrt(sample variance + 1e-4)) plus .04 times
off-diagonal covariance-square sum /64; total weight .02. Compute the same
metric in both arms; frozen features stop its gradients. The merged backend
does not expose the sine/cosine operations used in historical pretraining;
this small native regularizer avoids adding an unrelated backend feature.
Task supervision and causal prediction remain the learning objectives.
[VICReg](https://arxiv.org/abs/2105.04906) motivates the stabilizer, but this is
a new combined objective, with no theoretical guarantee of control sufficiency.
Statistics use one microbatch's frames; fix the same microbatch size in both arms.
An independently generated F64 oracle checks this objective separately from Tiny.

### Historical host review before ordinary-compute qualification

The same boot (`3e89d55c-a9e5-472f-a18a-06508c5bafa7`, driver580.178.04) has
three retained `_memdescAllocInternal` allocation warnings. No new Xid, hung
worker or recovery is recorded. Their cause remains unresolved; this is not a
hardware-health claim. The user's new experiment authorization permits one
guarded ordinary-compute qualification, not a retry of external-memory import.
Declarations identify the exact three reviewed historical warning cursors and
messages. Any new occurrence (including the same message at a new cursor) stops
the child; Xids and other hard faults cannot be excepted. The new guard passes
109 CPU tests. Historical evidence and its false-negative guard result remain
unchanged. External-capture v4 producer qualification remains separate/unrun.

October3 update: bounded numerical diagnostics may retain at most two new
occurrences of the one reviewed allocation message for at most120s; other faults,
validation errors and deadlines stop. The known workgroup-layout VUID alone is
non-blocking by user direction. This is not a blanket long-training exception.
The first full probe completed three synthetic updates and saved a checkpoint,
then hit its deadline during a second model construction for restore. Split
training and restore into separately declared tests; reuse that checkpoint.
Keep live encoder prefixes in cost measurements, not only empty chunk boundaries.

## Allocation and decision

At most **six complete learning runs**: frozen versus joint, Seaquest,
seeds1009/2017/3019. No automatic retries or successors after failures.

**Declared October3, after optimized synthetic timing and before gameplay:**
8,192 actual actions per run; Size1M/N8/B8/T16/H15/R32, world microbatch1,
replay capacity8,192, learning rate4e-5 with1,000-update warmup, AGC.3.
Keep the upstream default optimizer/return settings. Full18 actions, sticky.25,
repeat4, published Atari protocol, native-detail Tiny input, no reward/action
aid. Pretrained Tiny and all objectives above are identical between arms; only
encoder updates differ. Order: frozen1009, joint1009, joint2017, frozen2017,
frozen3019, joint3019. Deadlines are2h per joint run and1h per frozen run.
The existing scheduler retains R32 after replay eligibility; report actual
updates and initial eligibility, not an assumed count or added prefill credit.
Report every1,024 actions; retain checkpoints at4,096 and8,192 actions.

Optimized synthetic full updates measure2.211s joint versus.487s frozen,
about4.54x more time. Three paired8k screens should cost roughly4–5GPU hours,
not the multi-day200k-action repetition. This is an **early learning/collapse
screen**, not competence or a sufficiently powered claim of superiority.
The new frozen control deliberately shares pixel replay/current encoding; it
is not the faster historical cached-feature path. Compare against it honestly,
and keep old results as context only. New allocation warnings remain fatal
under ordinary long-run guards; do not extend the120s diagnostic exception.

Report all episodes/tails, score versus actual actions and wall time, paired
seed differences and bootstrap uncertainty. Record encoder movement, latent
spread, task losses, prior forecast diagnostics and complete-agent cost. A
finite checkpoint or nonzero gradient is not learning evidence. Preserve failed
qualifications and extra compute. Do not claim frozen competence or a universal
JEPA result from this one held-out title. Status remains in PR31, not STATUS.md.

Use `summarize_representation_learning.py --joint-tiny` with completed logs.
It audits causal-chunk eligibility, reset arrivals, eviction and training credit;
partial groups get no three-seed aggregate. Update-window latent/task/timing
means remain beside the action/time curves. The existing dynamics probe now
accepts native pixels and sticky actions, samples Vulkan budget headroom and
reports held-out temporal latent spread alongside persistence/unrelated-action
and reward/continuation controls. Never compare raw feature MSE across encoders
without their scale and collapse diagnostics. Run those frozen probes serially
after reviewing the learning guards, not alongside timed learning.
