# Bounded enumerated one-step actor cost and learning check

The [native qualification](../results/2026-11-06-cdp-actor-expected-value.md)
passes independent objective/derivative arithmetic, all Tiny action branches,
shared draws, pre-update snapshots, two learner updates and optimizer restore.
This establishes implementation correctness within those fixtures, not gameplay.
Select one fresh8,192-action Breakout learner with `actor_expected_value=true`,
then1,024 frozen greedy actions. No full learner is selected.

This replaces the sampled-action policy term with the expectation over all18
actions at replay posterior starts. Q includes predicted reward, the existing
bonus and predicted discounted continuation times critic value. Q is detached;
one common latent draw is reused across actions. Entropy, imitation, critic and
ordinary imagination remain. This changes horizon and start distribution along
with the gradient estimator; it does not isolate categorical gradient bias.

Keep12M64/Auto/N8/B16/T16/H15/R16, seed1009, replay100k, centered CDP,32×16,
latent/actor mixing.01, encoder/base4e-5, dynamics4e-4, behavior8e-5,
imagined/replay critic1/1 and constant expert100 with world gradients. Corrective
labels use native RGB;20% episode selection permits overrides up to1,920 actions.
Full18/sticky.25/repeat4/noops0, native pooled RGB and one GPU resize. Actor256;
coordinates, residual, dynamics-gradient and completed-return actor options off.
No new learned parameter, optimizer slot, image transform or replay field.
Only the actor option changes the effective recipe; serialization may make the
existing imagined replay-value bootstrap default explicit.

Use package source `f2b80fe596ae553bbea3cc7a3c638ea55529dc22`, native SHA256
`961f79583c1e2384f5dd2daba56fc2357c1a8dc280abd6bb8aef4d35fbf8b069`,
with its matching runner/teacher/runtime. Pin build/native receipts, sources,
protocol, queue and audits. No overlay or rebuild. Require all357 actual-initial
tensors/moments and47 pre-update transition events to match retained exactly.
Require first update at376 actions,489 updates total and final debt.5. Frozen
seed4000001009 has zero updates,357 invariant tensors and the preserved option.

Recount finite reports, expected-value loss and positive snapshot timing, both
weighted losses and changes in encoder, reward, continuation, actor and critic.
The expected-value metric is already inside policy_loss; never add it again.
World actor copies must stay synchronized. Replay every training action, native
pixel, label, reward, frame and boundary on CPU. Independently replay frozen
actions/outcomes/frames/boundaries; absent frozen pixel hashes do not establish
unrecorded image equality. CPU preflight exercises both queued commands without
native/emulator construction, missing-option rejection, initial tensor/trace
equality, and negative tensor/trace/loss fixtures.

Use idle healthy AMD and existing host guards. Caps180s training process,60s
frozen process,300s queue and180s audit; training replay120s and frozen30s. Stop
on real faults/nonfinite results, without reset, cap increase or automatic retry.
Preserve original failures; CPU-only repair may complete missing evidence without
repeating GPU stages. Record terminal children from sealed guard results and
verify controller/children/services/render holders are gone after completion.

Record actions and emulator frames separately, construction/training/component
times and memory snapshots. Compare bounded time with retained47.689684s.
Require padded200k projection `training_seconds*200000/8192*1.1 <= 1800`.
Subtract the entire5,939,200,000-byte full replay footprint from minimum observed
headroom and require2GiB remaining, conservatively double-counting allocated
pages. Failed cost gates are measured results. Passing only permits considering
a separately declared fresh comparison; bounded experience must not be reused.

Root `/x/Temp/kindle-actor-expected-value-20261106/bounded`. Keep checkpoints,
logs and packaged artifacts; release disposable build cache. Retain143.667 greedy/
70.750 sampled and original864/90%-two-wall/five-game gates. No full learner,
extension, coefficient/seed sweep or automatic successor follows from this check.
