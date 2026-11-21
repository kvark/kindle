# Bounded RGB96 learning-cost check

The [native qualification](../results/2026-11-06-cdp-rgb96.md) now passes corrected
resize, encoder values/derivatives and two Tiny updates with exact restore.
Original inactive-stream/scalar-layout fixture failures remain failed. Select one
fresh8,192-action Breakout learner and1,024 frozen greedy actions to measure the
full recipe's feasibility. No full learner is selected.

Keep retained12M/Auto/N8/B16/T16/H15/R16, seed1009, replay100k, centered CDP,
32×16, actor/latent mixing.01, base/encoder4e-5, dynamics4e-4, behavior8e-5,
imagined/replay critic1/1, expert100/world gradients and corrective labels.
Guidance selects20% of episodes with a1,920-action cap; evaluate with aid off.
Full18/sticky.25/repeat4/noops0, native pooled RGB andone GPU resize remain.
All optional actor paths, coordinates, frame pairs and cropping remain off.
Only `observation_kind=rgb96` changes the effective recipe. Explicit serialization
of the existing imagined replay-value-bootstrap default is permitted.

This increases CNN spatial cells16→36 and embedding1024→2304. CNN weights, RSSM
and actor/critic widths remain, but posterior input, both CDP predictor layers
and four disagreement output heads grow. Derived trainable count is24,862,993
versus16,334,353, an8,528,640 increase. Independently patch and verify the42
affected checkpoint tensors, including moments, while retaining357 tensor names.
Full replay pixels/cached states require12,083,200,000 bytes. This is a joint
capacity/geometry/compute test, not isolated resolution or perception evidence.

Use native source1279a415/SHA256
`7704d25c9df5d37c48d5223f2d12c1d5d3c0a812c1f68cf393c87a637d010ab0`, matching
runner/teacher/package and the reference-repair receipt. Its only source changes
are the two preserved test-fixture repairs; no production overlay or rebuild.
Pin all receipts, source resolutions, helpers, commands and protocol. CPU
preflight exercises both real commands with native/emulator construction blocked,
missing96-geometry rejection, exact schema, cadence, frozen tensor invariance
and negative loss/tensor fixtures. Do not require initial weights/actions to equal
the64px baseline: geometry changes shapes and observations. Keep the candidate's
actual initial checkpoint for any later independent fresh comparison.

Require first update at376 actions,489 updates, final debt.5, finite reports,
correct weighted losses and changed encoder/reward/continuation/actor/critic
weights. Verify actual parameter count and all checkpoint schemas, synchronized
frozen actor copies and full357-tensor frozen invariance. Frozen seed4000001009
haszero updates and preserves96px geometry. Replay every training action, native
pixel hash, label, reward, frame and boundary on CPU. Independently replay frozen
actions/outcomes/frames/boundaries; do not claim absent frozen pixel hashes.

Use idle healthy AMD and existing host guards. Caps180s train,60s frozen,300s
queue,180s audit; training replay120s and frozen30s. Stop on faults/nonfinite
results, without reset/reload/reboot, automatic repeat or cap changes. Verify
original terminal child/controller identities and idle closure. Retain failures;
separate CPU repairs may complete evidence without repeating GPU stages.

Compare training time against retained47.689684s at the same action/update budget.
Require `training_seconds*200000/8192*1.1 <= 2100` (the predeclared35-minute cap).
Subtract the entire12,083,200,000-byte full replay footprint from minimum observed
bounded headroom and require2GiB remaining, conservatively counting allocated
pages twice. Failed gates are measured results, not permission to raise caps.
Passing permits consideration of a separately justified fresh comparison; do
not transfer bounded experience or start an automatic successor.

Root `/x/Temp/kindle-rgb96-20261106/bounded`. Keep checkpoints, logs and useful
packaged artifacts; release disposable build caches after hash verification.
Retain RGB64 and143.667 greedy/70.750 sampled. Original864/90%-two-wall and
five-game gates remain unfinished.
