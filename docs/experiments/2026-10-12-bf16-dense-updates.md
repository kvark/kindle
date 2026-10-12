# Native BF16 dense learner integration

Declared October 12, 2026, before GPU execution. This continues the
[backend plan](../backend_training_performance_plan.md); no game actions,
evaluation episodes or learning campaign are authorized by this check.

## Change and controls

Explicit `bf16_dense` converts ordinary dense products in all five learner
sessions: posterior, imagination, world gradients, behavior gradients and slow
value. Master weights, optimizer state, accumulators, losses and reductions
remain F32. Supported pointwise boundaries use BF16 storage and F32 arithmetic.
Convolution and grouped RSSM products remain F32; live acting is unchanged.
This is not whole-network BF16 or a change to the default precision.

First qualify graph selection with independently evaluated F64 derivatives,
both fused and unfused. Require its positive native-device completion marker.
Then use one binary for each control/candidate pair, the existing fixed-batch
benchmark, identical checkpoints and fixture RNG:

| Recipe | Control | Candidate | Work |
| --- | --- | --- | --- |
| CDP 12M, B32/T16/H15 | F16 matrix permission | BF16 dense | 16 warmup + 256 measured updates |
| RGB 12M, B8/T16/H15 | F16 matrix permission | BF16 dense | 16 warmup + 256 measured updates |

The candidate uses native F32 for products not selected for BF16. Historical
Auto inference is retained in the control. Those are explicit precision-policy
differences, not an exact F32-trajectory comparison. No convolution PR248 or
row-layout PR249 is stacked into these measurements.

## Acceptance and limits

- Same exact initial master tensors (346 CDP / 292 RGB), config identical
  except precision. Record effective source revisions, patches and binary hash;
  a temporary Cargo path override does not change the embedded backend label.
- Require native RTX 5080, at least 2 GiB sampled Vulkan headroom, passing host
  guards and no unreviewed validation/API failure. Retain allocation warnings.
- Each BF16 learner session must contain native BF16 matrix dispatches. Record
  typed pointwise coverage and standalone conversions, not just an option flag.
- All 272 reports, first targets, and initial/first/final tensors must be finite;
  confirm updates actually change master tensors. Report loss distributions and
  changes, without demanding identical stochastic trajectories or calling
  synthetic stability Atari quality.
- Time ordinary complete updates including parameter synchronization, without
  dispatch instrumentation or compilation. Report stages and scratch separately.
  An isolated kernel gain does not pass whole-update speed acceptance.
- One 120-second kernel qualification; four complete-update processes bounded
  to 600 seconds each, serialized under the existing GPU guard. Stop and review
  any failure before a separately recorded correction. No automatic sweep.

Artifacts: `runs/bf16-whole-update-20261012.38E86FlS`. Two CPU preparation failures
(a Clippy pattern and a test enum path) precede the corrected build; neither
ran GPU work. Preserve their systemd journals with the experiment.

## Reviewed tiled-kernel follow-up

The first five guards pass with zero allocation warnings. All initial tensors
match, all targets/reports/checkpoints are finite, and all candidate tensors
change. CDP161.54->156.40 ms (3.18%); RGB71.35->66.90 ms (6.24%). World training
regresses in both; this does not close the backend performance objective.

The original BF16 kernel duplicates its 16x16 output across subgroups. Test one
32x32 workgroup tile with independent subgroup tiles and 32-wide K staging.
Keep precision selection and model code unchanged. Qualify all BF16 fixtures
again, expanding the reduced-gradient cases to35x37x41 to cover multiple and
partial tiles. If they pass, repeat both complete-update pairs under the same
bounds and audit, in new output directories. No tolerance relaxation or sweep.

## Complete-submission tracing

After ordinary timings finish, use the same synthetic benchmark with calibrated
GPU timestamps and stage markers, without per-dispatch instrumentation. Record
all Meganeura submission intervals in Perfetto, not only the last submission's
timings. Initially capture CDP F16 and BF16, 272 updates each, 600-second bounds.
Compare traced wall time with the ordinary measurements to expose perturbation.
Pass intervals are neither SM utilization nor a claim that game capture and
every Kindle-side copy have been measured. Retain traces and parsing checks.

Coverage review finds `record_groups` submits intermediate chunks without
harvesting their timestamps before Blade resets its query pool. These two
captures therefore remain **incomplete**; uncovered intervals are not CPU
time or GPU idle time. Keep the failed interpretation and raw traces. Do not
use their sums as a complete-update GPU measurement.

## Git-pinned qualification and fresh JAX control

The tiled ordinary comparison passes: CDP161.49->149.66 ms (7.33% lower),
RGB71.43->66.55 ms (6.84% lower), five guards and two retained allocation
warnings. The same code is now pinned without local paths: Meganeura679ea87,
Bladeb52965b, Naga forkc9fd6a85a; Kindle integration0010772 keeps Auto default.
Workspace/Python Clippy,12 config tests, the backend-lock test and187 Python
runner tests pass. A failed targeted Cargo update from the temporary path
dependency is retained; normal manifest resolution fixes the lock refresh.

Repeat the four native ordinary processes using that pinned binary, then the
existing pinned upstream JAX0.6.2 RGB12M/B8/T16/H15 comparison, FP32/highest and
BF16. Each JAX process is bounded to900 seconds,16 warmup +256 measured updates,
explicit completion/policy synchronization, no profile updates or game actions.
Keep the original framework/weights/RNG/decomposition differences: not a
same-model numerical comparison. No other GPU work or compilation during timing.

## Reviewed trace repair

Harvest intermediate chunk timestamps before query reuse only when GPU timing
is enabled. This adds diagnostic waits between chunks; ordinary timing-disabled
execution is untouched. Qualify all four labels and unchanged chain outputs in
one120-second native check, then repeat the two CDP traces under600-second bounds.
Require all four world-chunk labels and exact272 reports/first targets/final346
tensors against their retained ordinary arms. Report added trace overhead;
neither uncovered intervals nor recorded pass intervals establish SM utilization.
