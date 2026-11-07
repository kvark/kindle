# Fresh 200k Breakout completed replay-value comparison

Select one fresh learner after the [bounded check](../results/2026-11-06-cdp-completed-value.md#completed-bounded-activity-and-cost-check)
passes native execution, target activity, replay, cost and memory gates.
Its 8,192 actions / 489 updates take36.403s, with56,061 completed-target row
exposures, padded200k projection977.620s and11.920GiB conservative replay
headroom. These establish affordability and execution, not gameplay benefit.

The hypothesis is that completed observed discounted returns can ground replay
value training more usefully than short targets bootstrapped from the same
critic. Use them only where available after a true game terminal; retain the
existing lambda target elsewhere. They describe the collecting policy, including
guidance, and are not unbiased current-policy values. Timeouts and live episodes
remain unlabeled. This changes an existing target, adding no graph, parameters,
loss coefficient or actor objective. It is not a SuperMimic reproduction.

Reuse exact qualified native d0e7bb79, SHA256
`3146e0dca5a008f9a3335c823572a862833b93983f6f3bf7fce5664bb9dbfc37`,
runtime and bounded runner; no build or overlay. Keep CDP500/12M64/Auto/N8,
B16/T16/full BPTT/micro16/context1/H15/R16/replay100k, categorical32x16,
base/encoder4e-5, dynamics4e-4, behavior8e-5, critic1/1, representation KL.1,
dynamics KL1/free nats1, actor/latent mixing.01, continuation1, expert100/unit
weights/world gradients/all-state corrective labels, 20% guided episodes/cap1920,
sampled collection/posteriors. Disable disagreement; enable completed replay
values. Full18/sticky.25/repeat4/noops0, native max-pooled RGB/one GPU64 resize.

Start fresh at reused development seed1009/evaluation cohort4000001009. No
bounded weights, replay or experience transfer. Train200,000 aggregate actions /
12,477 updates within1800s. Retain a read-only100k/6,227 recovery with261 finite
tensors. Require all261 initial arrays and47 pre-update events to match bounded
and the historical bonus-off control. Check exact configs, cadence, both weighted
losses, actor sharing, component timings and completed-target activity by50k
windows. Count repeated row exposures, first terminal and first active update;
model bonus must remain zero and the completed-return actor absent.

The closest historical control is `/x/Temp/kindle-episodic-actor-20261103/full/control`:
65.208 greedy/46.458 sampled in1025.486s, with only the new boolean differing in
effective learning config. Runtime/native/runner and guidance-wrapper versions
differ; core teacher class ASTs are qualified unchanged. Preserve these limits
and require identical initial frozen semantic traces. Reuse the control; do not
rerun it. Also report the older completed-return actor point95.375/56.542 in
1032.959s as context, with its distinct objective disclosed.

The primary adoption comparator remains bonus-on143.667/70.750 in1287.910s,
own-initial greedy gain/min6.590132. Favor the candidate only if raw greedy
return and own-initial gain/min improve and the paired eight-lane95% gain/min
interval has positive lower bound (10,000 resamples, seed0). Report the same
comparisons against the bonus-off control separately; beating that weaker
control alone does not displace primary. Report sampled results and raw-return
intervals. Original task gate remains mean864 and90% two-wall. One reused seed
and cohort do not establish reliability; a favorable result needs confirmation.

Evaluate final and actual-initial checkpoints in sampled and greedy modes:
first three natural episodes per eight lanes=24 per arm, guidance/labels/learning
off, all261 tensors invariant. Caps per arm:600k actions,100k frames/episode,
300s wall; no substitution or cap increase. Audit complete action/reward/frame/
boundary replays and save videos. Frozen logs omit live pixel hashes, limiting
image-equality claims. Retain full training pixel/label/override replay, curves,
exposure/score-range accounting, independent recovery and terminal collection.
CPU observer time remains included in training wall time.

Root `/x/Temp/kindle-completed-value-20261106/full`; service
`kindle-breakout-completed-value-200k-20261106`. Qualify actual five command
paths with native/emulator construction mocked before launch. Pin scripts,
runtime, protocol, controls and ALE/ROM identities. Use the existing AMD host
guard and check current live device/process identity. One GPU allocation at a
time; stop on actual faults/nonfinite values. No driver recovery, successful-stage
repeat, extension, second learner, coefficient sweep or automatic successor.
Main/collector caps7200/7500s; training/frozen replay caps900/500s. Inspect the
original service after observer timeout. Preserve prior packaged fixture failures
and separate CPU repair; any new protocol failure remains visible. The five-game
goal remains unfinished and the retained default stays unchanged pending evidence.
