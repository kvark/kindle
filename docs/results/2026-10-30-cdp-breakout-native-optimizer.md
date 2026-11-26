# Native optimizer observations agree in aggregate; strict parameter bounds fail

Native execution directly measures encoder updates of 0.00211656% and
0.00233922% of its weight norm on the two captured batches, versus recurrent
0.14900506% and0.14489360%. This reproduces the small encoder movement from the
[CPU estimate](2026-10-29-cdp-breakout-optimizer-attribution.md). Every raw gradient
matches exactly, and all clipped-gradient, momentum, variance and group-movement
checks pass. Nevertheless, the declared complete comparison **fails**: six scalar
parameter values exceed its F64 rounding allowance. No passing full qualification
is claimed and no tolerance is changed afterward.

Both disposable native optimizer steps complete with the host guard passing.
The controller fails during CPU comparison, then a separate CPU closure checks
both already-saved batches under the original bounds. Status is
`closed_protocol_failed`. There is no repeated GPU stage, learner update,
gameplay action or optimizer-setting change. The accepted learner is unchanged;
the five-game goal remains unfinished.

## Native observation

The [protocol](../experiments/2026-10-30-cdp-breakout-native-optimizer.md) uses the
same86 trainable world parameters and172 moments from the accepted12M64 control,
with source counter12477. A diagnostic subset preserves all258 tensors exactly.
Its purpose is to satisfy the backend's strict checkpoint-layout match for a
linear dot-product graph; it is not a replacement learner checkpoint.

For each batch independently, the native test restores this subset and proves
the linear graph produces the captured raw gradient exactly with updates off.
It then applies Kindle's production-configured AGC/LaProp, including the existing
per-name rates and warmup, and saves parameters, moments and clipped gradients.
Both before-states match the source exactly, and each counter changes12477→12478.
Exactly two disposable optimizer steps and four synthetic backward passes occur.
There is no world-model backward or resumed learner.

|Comparison across both batches|Passing tensors|Maximum relative L2 error|Maximum absolute error|
|---|---:|---:|---:|
|Clipped gradient|172/172|1.823e-7|4.022e-7|
|Momentum|172/172|1.480e-7|4.899e-7|
|Variance|172/172|4.916e-8|7.051e-8|

The measured parameter movement, compared with the F32-rounded CPU estimate, is:

|Batch|Group|Observed update L2 norm|Observed update / weight norm|Movement relative L2 error|
|---|---|---:|---:|---:|
|0|encoder|0.000431601145|0.002116557%|9.009e-06|
|0|posterior|0.0852170559|0.144744693%|1.133e-06|
|0|dynamics|0.232056393|0.149005056%|1.131e-06|
|0|other|0.27738872|0.179395725%|9.935e-07|
|1|encoder|0.000477004989|0.002339215%|3.683e-06|
|1|posterior|0.0877868714|0.149109631%|1.206e-06|
|1|dynamics|0.22565332|0.144893598%|1.091e-06|
|1|other|0.327063698|0.211522044%|9.167e-07|

All eight group checks pass the declared1e-2 bound; the largest observed error
is9.009e-6. The raw derivatives match for all14,688,256 elements in each batch.
The complete source checkpoint remains byte-identical, protecting all357 tensors.
These direct measurements support investigating the encoder's slow movement,
but they do not identify a beneficial rate/clipping change or explain gameplay.
A smaller model's previous higher encoder-rate comparison was worse; do not
infer that simply raising the rate will solve Breakout.

## Six cancellation-sensitive values fail the original bound

The CPU reference computes the current contribution and inherited momentum in
F64, while the native optimizer accumulates them in F32. At these six values,
large opposite contributions nearly cancel, leaving a small final movement.
The declared allowance scales partly with that small final movement, so it does
not cover the observed difference from rounding the intermediate momentum.
This explains the narrow failure; it does not turn the original check into a pass.

|Batch|Parameter|Flat element index|Absolute parameter error|Original allowance|
|---|---|---:|---:|---:|
|0|`world.dynamics.core.dynhid0.weight`|853614|8.852e-12|3.742e-12|
|0|`world.dynamics.core.dynhid0.weight`|1282930|3.196e-12|3.108e-12|
|0|`world.exploration.head0.layer0.weight`|560075|9.043e-12|2.916e-12|
|0|`world.future_predictor.out.weight`|154385|8.077e-12|7.476e-12|
|1|`world.dynamics.core.dyngru.weight`|548574|4.545e-12|4.405e-12|
|1|`world.future_predictor.out.weight`|867379|5.987e-12|4.007e-12|

For example, one recurrent weight has history−8.83713e-5 and current+8.83679e-5,
leaving a predicted movement−3.46542e-9. Its saved native momentum is8.68537e-6
versus F64-predicted8.66354e-6, resulting in parameter error8.85213e-12 beyond
allowance3.74194e-12. The per-tensor momentum error still passes its bound.

A separate conditional CPU check computes the parameter write using the *observed
native momentum*. After F32 rounding, it exactly reproduces every parameter:
14,688,256/14,688,256 per batch, including all six violations. That verifies the
write given its saved momentum; it is not an independent validation of the F64
moment prediction. All original thresholds, failed comparisons and service
results remain unchanged. No replacement passing original review is created.

## Execution, closure and provenance correction

The first source875718d invocation starts00:05:42 and fails00:05:44 UTC October30.
The diagnostic's row-reduced loss has shape[1,1], whereas autodiff expects a
scalar[1] root. Compilation fails after GPU-context creation, before restore,
either batch or any backward/optimizer step. All219 original pins and the exact
failed source/guard remain preserved.

The corrected scalar reduction is sourced56d59fd. It adds a CPU compile test
covering all86 actual parameter shapes, and passes142 Rust CPU tests/strict
Clippy plus the unchanged four comparison fixtures. Corrected service
`kindle-breakout-native-optimizer-scalar-20261030` starts00:10:03 UTC, invocation
`6113e6733a494fa4b6861f382fb95204`. Both native batches complete; the measured loop
takes.808s and the complete native test1.57s. Minimum sampled headroom is19.44GiB.
The host-only guard passes; it does not claim continuous telemetry or hardware
qualification. The main controller fails its CPU parameter check at00:10:08.

CPU closure preserves the original batch0 review exactly and evaluates batch1
under every original threshold. It verifies238 original pins and11 closure pins,
with zero additional GPU work or native imports. Closure review completes00:13:13
UTC; terminal collection00:17:01. Both invocations remain terminal failures,
reflecting the distinct startup and comparison failures. No allocation is active
at collection; check live state before later GPU work.

The corpus provenance was also rechecked. Earlier gradient/optimizer report prose
incorrectly called the frames teacher-guided. They are from the retained control's
**unaided final greedy evaluation** (`evaluate_greedy`), with corrective teacher
labels added offline. All20,960 recorded actions match the corpus; the source
log hash is`a35b99a6a9b097d59834ac4b7dddaec64b79ec93dafc47e3323a07b483b5ae66`.
The two earlier reports are corrected. No input, gradient or numerical result
changes. This already covers selected states from that unaided rollout; it does
not establish general autonomous-state coverage or reproduce training replay.

Evidence root:`/x/Temp/kindle-optimizer-native-20261030/scalar-corrected`.

- `native/` retains both complete before/after subsets and clipped gradients.
- `batch-0-review.json` preserves the original failed comparison.
- `closure/batch-0-review.json` and `closure/batch-1-review.json` retain all
  original numerical checks and the conditional native-momentum reconstruction.
- `closure/review.json` SHA256
  `652751b2ad1f53795cb130c5475c4328acec7e7bbbd777c352ea5f921c86f3a8`.
- `completion.json` records the failed original service, completed native work,
  unchanged238/11 pins, six bound violations and zero repeated GPU work.
- `corpus-provenance.json` records the corrected trajectory provenance.
