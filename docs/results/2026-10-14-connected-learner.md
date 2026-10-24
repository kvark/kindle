# Connected regions: real block gain, small learner gain

October14,2026. [Declaration](../experiments/2026-10-14-connected-learner.md)
· [Data](2026-10-14-connected-learner.json)
· [Earlier block prototype](2026-10-14-connected-row-region.md).

**Complete RGB updates improve28.280→27.999ms, only.99%.** This misses the
declared3% gate: no CDP follow-up, production promotion or further row-fusion
sweep. The isolated35.5% block gain did not translate into a large learner
gain. The remaining experimental gap is still about2.2× retained JAX12.836ms;
production stays on its earlier qualified~53ms runtime. The whole goal is open.

## What changed

The general reduction supports per-column inputs and additional row/element
outputs. Automatic matching saves forward normalization/sigmoid values, emits
the activation directly, and combines activation/normalization backward with
scale-gradient contributions. It retains existing hierarchical scale-gradient
reductions, explicit BF16 stores, F32 arithmetic and public outputs. No
Dreamer-specific or separate BF16 shader is introduced.

The actual learner selects118 world regions and nine behavior regions. Count
the staged world once, not once again under its posterior export: complete
dispatches fall5,286→4,908. Imagination and slow-value plans do not change.
All heavy-operation descriptors match, including original input provenance and
channel-packing recipes; only physical packing-buffer IDs move.

| Complete-update stage | Control ms | Connected ms |
| --- | ---: | ---: |
| Posterior |4.324|4.226|
| Imagination/targets |7.409|7.428|
| World training/sync |14.981|14.771|
| Behavior training/sync |1.567|1.575|
| Total |28.280|27.999|

One ordinary timing pair,16 warmup+256 measured updates/arm, same checkpoint,
RGB12M/B8/T16/H15/BF16 and four world chunks, same binary disabled/enabled.
No concurrent compilation or dispatch instrumentation. A short smoke's3.15%
difference is not substituted for this full comparison.

## Numerical evidence and limits

Seven shapes, F32/BF16, control/candidate and alias/no-alias poisoning cover
56 cases and10,251,192 values. Independent analytic F64 activation/dX/dW gates
pass, maximum relative L2=1.133e-6. BF16 outputs exactly match explicit RNE of
the separately qualified F32 control. Two staged pairs retain forward values
while dy arrives later; whole/staged outputs and derivatives match exactly.
Fourteen CPU cases, three unsafe-match rejections, public-output preservation,
27 shader tests and release Clippy pass.

The disabled learner reproduces retained272 reports, first outputs and all292
initial/first/final tensors exactly. The candidate preserves all initial292
tensors and first outputs, but only2/272 reports are exact. First saved state
has188/292 exact tensors, aggregate relative L2=.07862; final2/292, L2=.08319.
Both include optimizer tensors. Finite state is not learning equivalence, and
an optimizer-only explanation has not been demonstrated. The inherited
independent row-fusion rounding-boundary failure is not erased by this check.

The earlier backward-only integration remains separately recorded:60
optimizer-free block replays and48 smoke updates, no timing pair. Completing
forward fusion before timing adds60 replays and592 updates. Total:640 synthetic
updates,120 optimizer-free replays, zero game actions; nine guards pass, five
allocation warnings retained. No recovery or GPU worker remains.

## Reproducibility and decision

Artifacts: `runs/connected-learner-20261014.wLj595Bi`; final candidate lives
under `full/`. Complete source patches, locks, binaries, inputs, plans, states,
failed builds and both failed CPU auditors remain. The schema audit originally
assumed a nonexistent fixture field; the inventory audit originally compared
physical buffer IDs. Corrected checks prove four-chunk source configuration and
identical packing recipes/provenance, not merely equal convolution dimensions.

The source retains the optimized056562b-based experimental backend and
Blade96c1dc7/Naga a97559b. Upstream7574ad1/e7c3f7b is rechecked; this is not a
combined latest-main migration. Neighboring user work and installed pins stay
unchanged. Keep this measured capability and its limits, but move investigation
back to heavy-kernel execution and device dependencies. Neither fewer
dispatches nor one fast block explains the full gap.
