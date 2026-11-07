# Native qualification of saved Breakout optimizer estimates

The CPU estimate finds strong encoder clipping and a large inherited-momentum
contribution on both qualified gradient batches. Before choosing a learner
change, validate that estimate against native Meganeura423d373 AGC/LaProp.
This is an optimizer diagnostic on disposable copies, not resumed learning or
an assessment of gameplay benefit.

Reuse both raw total-gradient captures from
`/x/Temp/kindle-gradient-attribution-20261029/auto-corrected` and the completed
CPU analysis at `/x/Temp/kindle-optimizer-attribution-20261029`. Verify all191
prior pins and resolve any subsequently changed source from an exact retained
copy/Git before execution. Preserve all original evidence.

Prepare a diagnostic checkpoint containing the86 trainable world parameters and
their172 saved moments, with logical shapes/types and Adam step12477 preserved.
Require all258 tensor values to match the complete source exactly. This subset
exists only to satisfy the backend's strict graph-layout checkpoint format;
it is not a replacement learner checkpoint. The complete357 source tensors
and original checkpoint files must remain unchanged.

Construct a test-only native graph whose loss is the sum of parameter/gradient
dot products, with the original parameter names/shapes. Its derivative is the
captured raw gradient. For each batch independently, restore the same weights
and moments, then run an optimizer-free backward pass and require every raw
derivative exactly equal to the captured input. Configure Kindle's production
optimizer with its saved learning rates/warmup, clipping and LaProp settings,
then execute one synthetic backward/optimizer pass. Save before/after state and
post-clipping gradients. Both batches must start at12477 and finish12478.
Exactly two disposable optimizer steps/four synthetic backward passes occur;
no world-model backward, learner update, environment action or policy evaluation.

Before GPU work, pin the test executable, sources, derived checkpoint and CPU
validator; run CPU tests and strict Clippy. Use the RX7900XT under the established
host guard with a120-second cap and at least2GiB sampled memory headroom. Stop on
device/numerical failure; no host recovery or automatic retry. Reuse completed
GPU outputs for any CPU-only repair without repeating GPU stages.

Compare each parameter's clipped gradient and moments against the existing F64
formula. Require relative L2 error<=3e-5. Maximum error must be<=3e-6 times
`1 + max(abs(reference))` for gradients/momentum, and<=3e-5 times
`max(abs(reference)) + 1e-30` for variance. Check finite values and nonnegative
variance everywhere. Compare stored parameters with the F64 estimate using an
elementwise allowance of two F32 ULPs of both old/expected values plus3e-5 times
the estimated movement magnitude. Additionally require group-level observed
movement to match the F32-rounded estimate with relative L2<=1e-2. Preserve
all errors and every declared group, without selecting favorable layers.

These checks qualify the transformation of two captured gradients through the
saved optimizer state. They do not validate arbitrary new coefficients,
earlier training or gameplay improvement. No new learner or optimizer setting
is selected. Record results, source hashes and terminal evidence outside Git.
Evidence root: `/x/Temp/kindle-optimizer-native-20261030`.

## Preserved graph-shape failure and corrected execution

The original invocation starts00:05:42 and fails00:05:44 UTC October30,
source875718d. The diagnostic's row reduction leaves a[1,1] root loss, while
autodiff seeds the scalar[1] loss gradient. CPU graph compilation fails after
GPU context creation, before session restore, either batch or any backward/
optimizer step. All219 original pins are verified at closure; the empty native
output directory, exact source, journal and failed guard remain preserved.
No completed GPU stage is repeated and no host recovery occurs.

Select a separate corrected execution under `scalar-corrected/`. Use the full
scalar `sum_all` reduction, preserving the intended dot-product loss. A new CPU
compile test checks scalar output, parameter shapes and one gradient/input per
parameter, including the complete actual86-parameter manifest. All142 Rust CPU
tests and strict Clippy pass; the four prior CPU comparison fixtures remain
unchanged. Reuse the exact derived checkpoint and gradients, with the same
120-second/memory/numerical limits. No backend or production learner change.

## Terminal outcome

The [result](../results/2026-10-30-cdp-breakout-native-optimizer.md) is
`closed_protocol_failed`: both native steps and the host guard complete, but
six scalar parameter values exceed the original F64 rounding allowance. All
clipped-gradient/moment/group checks pass. CPU closure preserves every bound
and both original service failures, with238/11 pins and no repeated GPU work.
No passing full qualification, learner update or new optimizer setting exists.
Terminal collection00:17:01 UTC October30; corpus provenance is corrected in
the linked reports to unaided greedy evaluation with offline teacher labels.
