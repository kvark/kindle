# Connected spatial lowering

One compiler intervention follows the [actual image attribution](../results/2026-10-14-jax-image-profile.md): preserve a max-pool winner tape, absorb an eligible typed pointwise producer, and emit the backward dense gradient and its spatial sum together. This tests whether unnecessary materialization and recomputation explain a meaningful part of the remaining gap. It is not another tile, dtype, or batch-size sweep.

The candidate is isolated under `runs/spatial-tape-20261014.IrngqAGq`. Its base is the qualified cached backend and unchanged B32 fixture, not the user's neighboring checkout. Rechecked upstream Meganeura `99e3de27` and Blade `e7c3f7b` contain no identified fix for this path. No production pin, precision policy, learning equation, Atari allocation, or model size changes.

## Contract

- One type-parameterized implementation; F32 accumulation and the original explicit BF16 gradient rounding boundary remain. No BF16-specific execution engine.
- Match first-maximum ties, padding, overlapping windows, and the existing spatial sum's 256-lane accumulation/tree order. Save decisions during forward execution; never reconstruct them using later parameter values.
- Eliminate a producer or gradient materialization only when every consumer is accounted for. Preserve graph outputs, parameter gradients, preparation dependencies and saved-tape lifetimes. Unmatched graphs retain their original path.
- CPU qualification covers shader validation, matching/rejection cases and storage types. Bounded GPU qualification checks independent references, overlap/padding/ties, and staged execution with poisoned late inputs before timing.

## Fixed budget and decision

At most one guarded generic numerical process (120 seconds), one complete first-CNN qualification/timing process (300 seconds), then one ordinary B32 control/candidate pair (300 seconds each, 16 warmup + 256 measured synthetic updates per arm). The complete block reuses all 5,104 independent F64 values and the retained actual-size outputs; timing uses the existing 152-call protocol. Generic cases and their exact invocation count are sealed before launch. Full-update tensors/reports and changed dispatch inventory are audited, not assumed equivalent from a microbenchmark.

Keep the existing numerical tolerances; stop and review failures without automatic GPU retries. A 3% ordinary whole-update reduction is the minimum useful speed result, not proof of learning quality or closure of the JAX gap. Keep all failures and costs. If this region is too small or a fused kernel loses the savings, report that directly rather than launch a tuning sweep.

The generic qualification is fixed at 24 cases: three rectangular/overlapping/padded geometries, F32/BF16, ties/non-ties, protected/unprotected source and gradient outputs. Each runs an unfused and fused ordinary/staged pair: 96 block invocations, zero measured timing calls. F16 is shader-validation coverage only, not a new GPU precision claim.

All GPU jobs use persistent guarded services, recorded allocation warnings, the expected 5080 and at least 2 GiB sampled Vulkan estimated budget headroom. No separate NVML polling or unsafe Vulkan timestamp capture. CPU preparation uses one CPU, 2 GiB and no swap. Zero game actions; no learning campaign resumes.
