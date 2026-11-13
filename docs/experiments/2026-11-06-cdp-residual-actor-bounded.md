# Bounded residual actor cost and learning check

The [separate qualification](../results/2026-11-06-cdp-residual-actor.md) now
completes the required evidence: original native affine/derivative/routing fixture
passes; the original learning fixture fails before its first transition on an
RGB observation constructor mismatch, and a separately compiled test-only repair
passes two synthetic updates, sharing and restore. Preserve that original failure.
No completed native fixture or old Atari stage is repeated.

Select one fresh8,192-action Breakout check with `actor_residual=true` and then
1,024 greedy frozen actions. It adds46,098 trainable parameters, total16,380,451.
The base receives imitation only and the zero-initialized linear correction
receives imagined policy/entropy gradients. Acting and imagination use their sum.
The CNN/RSSM/critic, world expert supervision and loss weights remain unchanged;
experience and subsequent learned weights can differ. This tests execution and
cost, not a causal bottleneck or natural-episode competence.

Keep12M64/Auto/N8/B16/T16/H15/R16, seed1009, replay100k, centered CDP,
latent/actor mixing.01, encoder/base4e-5, dynamics4e-4, behavior8e-5,
imagined/replay critic1/1 and constant expert100 with world gradients. All states
receive native-RGB corrective labels;20% episode selection enables overrides
up to1,920 actions. Full18/sticky.25/repeat4/noops0, native pooled RGB and one GPU
resize. Base actor width256; coordinates and dynamics gradients stay off.
Only the residual option changes the effective recipe; serialization makes the
existing replay-value bootstrap default explicit. No full learner is selected.

Use original native7e0bb827/SHA3003151f and its matching package/runner. The
fixture-repair source94f0a0d3 changes only a test factory; production/package
bytes remain identical. Pin the original build and combined-qualification
receipts, repaired source receipt, teacher, runner, queues and audit sources.
Require all357 retained initial tensors to match exactly and all six added
residual parameter/moment tensors to be zero. Compare all47 pre-update semantic
transitions. Require first update at376 actions,489 total updates and final
debt.5. Frozen seed4000001009 must perform zero updates and preserve all363
tensors (world280/behavior72/slow11).

Recount both weighted losses and finite reports; require changed encoder,
reward, continuation, base actor, residual weight/bias and critic parameters.
Verify synchronized base actor copies. Replay every training action, native
pixel, label, reward, frame and boundary on CPU. Replay frozen actions/outcomes/
frames/boundaries separately; absent live frozen pixel hashes cannot establish
unrecorded image equality. CPU preflight exercises both exact queued CLI commands
without native/emulator construction, missing-flag rejection, initial tensor
comparison and changed/extra/missing tensor rejection.

Use existing host guards on an idle healthy AMD GPU. Caps180s training process,
60s frozen process,300s queue and180s audit; CPU training replay120s, frozen30s.
Stop on real faults or numerical failures; no reset, cap increase or automatic
repeat. Preserve failed stages; a separate CPU repair may complete missing
evidence without repeating GPU work.

Record exact aggregate actions/emulator frames, construction/training/component
wall times and native memory snapshots. Require padded200k training projection
`seconds * 200000 / 8192 * 1.1 <= 1800`. Subtract the entire5,939,200,000-byte full
replay footprint from minimum observed headroom and require2GiB remaining;
this conservatively counts already allocated pages twice. Report failed gates
as results. Passing permits consideration of a separately declared fresh full
comparison, never reuse of bounded experience or an automatic successor.

Root `/x/Temp/kindle-residual-actor-20261106/bounded`. Keep useful artifacts and
release disposable build cache after package verification. Retain143.667 greedy/
70.750 sampled and the original864/90%-two-wall gate. Five-game goal unfinished.
