# Bounded actor-only width cost check

The [actor-width implementation](2026-11-06-cdp-actor-width.md) passes CPU
qualification and the original guarded two-update Tiny test. Native execution
and exact restore/synchronization pass in2.820s with243 unchanged pins and zero
Atari actions. Select one fresh8,192-action Breakout check with
`actor_hidden_units=512`, versus the retained12M actor's256. This adds1,054,720
trainable actor parameters, for17,389,073 total. No full learner is selected.

Keep12M64/Auto/N8/B16/T16/H15/R16, seed1009, replay100k, centered CDP,
latent/actor mixing.01, encoder/base4e-5, dynamics4e-4, behavior8e-5,
imagined/replay critic1/1 and expert100 with world gradients. All states receive
native-RGB corrective labels;20% episode selection enables overrides up to1,920
actions. Preserve full18 actions, sticky.25, repeat4, zero reset no-ops, native
pooled RGB and one GPU resize. Coordinates and actor dynamics remain off.
Actor width is the sole configuration change; native/runner versions also
differ from the historical control. Critic and world architecture stay fixed,
but their learned weights and collected experience can change.

Use native source80085208, SHA256
`eeaf699a84e341b99a4c9d2341fb9113c0edfe221de0569be7964bde67c9b00a`,
with its matching packaged runtime and copied current runner. Pin sources,
teacher, helpers and environment. Require all313 non-actor initial checkpoint
tensors to match the retained initial values. Exclude44 actor tensors/copies/
moments;40 shapes widen, while four output biases keep18 elements. Preserve
the new initial actor without requiring equal initial actions or scores.

Require first update at action376, total489 updates and final debt.5. Then run
1,024 greedy frozen actions at seed4000001009, zero updates and357 invariant
tensors. Recount both weighted losses, require finite reports, synchronized
actor copies and changed encoder/reward/continuation/actor/critic parameters.
Replay all training actions, pixels and corrective labels on CPU. Separately
replay frozen actions, outcomes, frames and episode boundaries; frozen logs do
not contain live image hashes, so that replay does not prove unrecorded image
equality. The frozen action budget checks restore only, not natural-episode
competence. Qualify both actual CLI command paths without native/emulator
construction, including rejection of an omitted width argument.

Use the existing host guard on an idle healthy device, one learner and eight
environments. Stop on numerical failures or real GPU faults. No reset/recovery,
automatic stage repeat or cap increase. Caps:180s training process,60s frozen
process,300s queue,180s audit; training CPU replay120s and frozen replay30s.
Keep original failures. A separately declared CPU-only repair can complete
missing evidence without repeating GPU stages. Parent audit uses this runtime.

Record construction, training/component times, exact actions/frames and native
memory snapshots. Padded200k projection is training wall time times200000/8192
times1.1, capped at1,800s. Subtract the entire5,939,200,000-byte full replay
footprint from minimum observed headroom; require2GiB remaining. This counts
already allocated replay pages twice. Report cost/memory gate failures as
measured results, without extending the check or selecting a full learner.
Passing establishes execution and projected cost only; it does not establish
better learning, an actor-capacity bottleneck or broad runtime equivalence.

Root `/x/Temp/kindle-actor-width-20261106/bounded`. Keep artifacts outside Git,
release the disposable build cache after packaging verification, and do not
reuse bounded weights/experience as fresh learning. Retain143.667/70.750 and
the original864/90%-two-wall gate. The five-game goal remains unfinished.
