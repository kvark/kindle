# Joint causal Tiny: task-adaptive latent prediction

The user explicitly authorizes this experiment on October 2. It precedes
exploration/reward, without reopening the completed Phase 2 allocation.

**Current boundary:** implementation and CPU checks only; no learning runs.
The backend canary stopped on a new NVIDIA allocation warning and also logged
shader validation errors. See the [qualification report](../results/2026-10-02-joint-tiny-qualification.md).
No follow-up GPU launch, automatic retry or recovery is authorized by this report.

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

## Qualification before gameplay

1. Adopt merged Meganeura `6268ea5` / Blade `e349cddf`, including main's optimizer
   fix. Check optimizer/reference tests before learning. Preserve adjacent dirty
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

### Host review before ordinary-compute qualification

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

## Allocation and decision

One bounded numerical/throughput qualification, then at most **six complete
learning runs**: frozen versus joint, Seaquest, seeds1009/2017/3019. No automatic
retries or successors after failures. Set the common interaction/update budget,
replay capacity and deadline here after timing and before any learning results;
the selected recipe must fit the fast-screening intent. Alternate arm order.

Report all episodes/tails, score versus actual actions and wall time, paired
seed differences and bootstrap uncertainty. Record encoder movement, latent
spread, task losses, prior forecast diagnostics and complete-agent cost. A
finite checkpoint or nonzero gradient is not learning evidence. Preserve failed
qualifications and extra compute. Do not claim frozen competence or a universal
JEPA result from this one held-out title. Status remains in PR31, not STATUS.md.
