# Bounded64×8 categorical grouping cost and learning check

The [separate native qualification](../results/2026-11-06-cdp-latent-shape.md)
completes with a test-only indexed-read repair. Original qualification remains
failed; repaired F64 categorical values/derivatives/samples and two Tiny updates/
sharing/restore pass. No completed fixture or old Atari stage is repeated.

Select one fresh8,192-action Breakout learner with `latent_shape={variables:64,
classes:8}`, followed by1,024 frozen greedy actions. The retained preset is32×16.
Both have512 flattened entries,2048 deterministic state and2560 actor features;
all parameter shapes/counts remain, total16,334,353. Sampling, active one-hot
count and KL behavior change together. This tests execution and cost, not a
representation cause, natural-episode competence or SuperMimic efficiency.

Keep12M64/Auto/N8/B16/T16/H15/R16, seed1009, replay100k, centered CDP,
latent/actor mixing.01, encoder/base4e-5, dynamics4e-4, behavior8e-5,
imagined/replay critic1/1 and constant expert100 with world gradients. All states
receive native-RGB corrective labels;20% episode selection allows overrides
up to1,920 actions. Full18/sticky.25/repeat4/noops0, native pooled RGB and one GPU
resize. Actor width256; coordinates, residual and dynamics gradients off. Only
categorical grouping changes the effective recipe; serialization may make the
existing imagined replay-value bootstrap default explicit. No full learner.

Use package source536012d5/native SHA256
`2c9c6e6f40bbcaf33b0c982d09e3a130057894acbe2115f54acd77d0d0a2e8bd`
and the matching runner/teacher/runtime. Test repairacc24213 changes only a test
file; production bytes stay unchanged. Pin both build receipts, original failed
qualification receipt, completed combined qualification, source files, protocol,
queue, native, teacher, runner and audits. No overlay or rebuild.

Require all357 actual-initial tensors/moments to match retained exactly, with the
same schemas. Initial sampled action traces may differ under changed grouping;
do not require retained trace equality. Require first update at376 actions,
489 updates total and final debt.5. Frozen seed4000001009 has zero updates and
357 invariant tensors (world280/behavior66/slow11), with64×8 preserved in metadata.

Recount finite reports and both weighted losses; require changed encoder, reward,
continuation, actor and critic parameters plus synchronized world actor copies.
Replay every training action, native pixel, label, reward, frame and boundary on
CPU. Independently replay frozen actions/outcomes/frames/boundaries; absent frozen
pixel hashes do not establish unrecorded image equality. CPU preflight exercises
both actual queued commands without native/emulator construction, missing-shape
rejection, initial tensor equality and changed/extra/missing tensor rejection.

Run on idle healthy AMD with existing host guards. Caps180s training process,
60s frozen process,300s queue,180s audit; CPU training replay120s and frozen30s.
Stop on real faults or numerical failures; no driver reset, cap increase or
automatic repeat. Preserve failed stages. A separate CPU repair may complete
missing audits without repeating GPU work.

Record aggregate actions and emulator frames separately, construction/training/
component times and native memory snapshots. Require padded200k projection
`training_seconds * 200000 / 8192 * 1.1 <= 1800`. Subtract the entire
5,939,200,000-byte full replay footprint from minimum observed headroom and require
2GiB remaining; this conservatively counts already allocated pages twice.
Failed cost gates are measured results. Passing permits consideration of a
separately declared fresh full comparison, without reusing bounded experience.

Root `/x/Temp/kindle-latent-shape-20261106/bounded`. Keep useful artifacts; the
package build cache is already released. Retain143.667 greedy/70.750 sampled and
the original864/90%-two-wall/five-game gates. No extension, sweep or automatic
successor is selected by this check.
