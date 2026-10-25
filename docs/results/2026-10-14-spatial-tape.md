# Connected image lowering helps, but does not explain the whole gap

October 14, 2026. [Declaration](../experiments/2026-10-14-spatial-tape.md) ·
[Data](2026-10-14-spatial-tape.json) · [Performance plan](../backend_training_performance_plan.md).

**Ordinary complete RGB12M/B32 updates improve 68.491 → 65.736 ms, 4.02%.**
The candidate still takes **2.42× retained JAX's 27.187 ms**. This closes only
6.67% of the fresh control-to-JAX gap: a demonstrated cause, not a sufficient
explanation or a performance-goal completion. Keep the isolated improvement;
production and Atari remain unchanged.

## What changed

The [preceding attribution](2026-10-14-jax-image-profile.md) located costly
image materialization and recomputation around convolution. A single compiler
pass now joins eligible pooling forward/backward dataflow:

- Fold the typed elementwise producer, here channel bias, into the pooling
  loads. Compute F32 values and save a compact winner index for each window.
- Backward reads those saved decisions instead of searching each window again
  for every input pixel. Decisions belong to the original forward execution.
- Write the explicitly BF16-rounded dense gradient and its F32 spatial sum
  together. Keep the original per-lane accumulation and 256-lane reduction tree.

This is one storage-parameterized implementation, not a BF16-only engine.
F32/BF16 are GPU-tested; F16 has shader-validation coverage only. External
consumers/protected outputs prevent invalid elimination. Unmatched regions
retain the previous path. No model, optimizer, loss, precision-policy, tensor
product, tile/K partition or input-image change.

All four encoder regions qualify. The staged world plan changes 4,101 → 4,089
dispatches: twenty old operations become eight, with 4,081 remaining descriptors
unchanged. The posterior and world-gradient plan files are two labels for this
same staged session, not two executions. The other three plans/pipeline lists
are exact. The first complete CNN block changes 23 → 20 dispatches.

Across four regions, eliminated dense F32 bias/gradient writes minus the new
winner tapes remove **686 MiB of logical scheduled output extents per update**.
This is not measured DRAM traffic, peak memory or physical free VRAM. The first
forward tape is 64 MiB instead of retaining its 256 MiB biased image.

## End-to-end result

| Timed work | Control ms | Candidate ms |
| --- | ---: | ---: |
| Complete update, fresh ordinary pair |68.491|65.736|
| Posterior preparation, within that update |9.250|8.476|
| Remaining world training/sync |36.866|34.904|
| Imagination/targets |17.814|17.752|
| Behavior training/sync |4.562|4.604|
| First complete CNN block, retained control |6.218|4.336|

The block improves 30.27%, but it is not a newly paired block control or a JAX
block comparison. Whole-update timing is the decision: 16 warmups plus 256
measured updates per arm, no profiler, compilation or competing GPU work.
The 3% keep gate passes. Small unrelated-stage fluctuations remain in the table;
do not sum isolated timings into a promised saving.

Both whole-update arms reproduce the retained control's **272 reports, first
outputs and all 292 saved tensors at initial/first/final boundaries exactly**.
Command-cache recording/reuse counts match. Fresh control plans/pipelines are
exact. This is unchanged synthetic learning math, not a game-learning result
or a claim of universal bitwise equivalence.

## Qualification and provenance

CPU checks validate fifteen shader variants plus fusion/consumer contracts.
Twenty-four GPU cases cover rectangular windows, overlap, padding, ties,
F32/BF16 and protected source/gradient outputs: 25,632 reference values and
exact unfused/staged comparisons. The complete CNN passes all 5,104 independent
F64 values and its retained full-size outputs exactly. Late inputs are poisoned
during preparation; saved-tape lifetimes and restored outputs pass.

Four guards pass with zero allocation warnings, numerical/API failures, NVML
polling or recovery. Total: 544 synthetic updates, 248 block invocations
(128 measured), zero game actions. No worker remains. Minimum sampled full-model
Vulkan estimated headroom is 12,665,225,216 bytes, not physical free/peak VRAM.

Seven CPU preparation/report failures remain: incomplete inherited source copy,
two incorrect CPU-test assumptions, nested-workspace discovery, an unused field,
the staged-posterior audit assumption, a repeated-key reader assumption, and a
patch-export header omission. These are seven failed invocations; the CPU-test
invocation contained two assertions. Corrections change no executed GPU cohort
or tolerances. Failed sources/logs and the first exported patch remain. No GPU
retry or timing sweep.

Artifacts: `runs/spatial-tape-20261014.IrngqAGq`; qualified block `f2b98efc`,
whole learner `3551bc53`. `intervention.patch` contains only the eight changed
backend files and applies cleanly to the retained cached backend. Full inherited
patches, source inventories, binaries, locks and reviews are sealed separately.
Meganeura `99e3de27` / Blade `e7c3f7b` upstream recheck adds no identified fix for
this path; the isolated base remains `056562b` plus the qualified inherited
changes, Blade `96c1dc7`, Naga `a97559b`. User checkouts and production pins stay
untouched. CI402 passes on the preceding `f01502e`.

## What remains fundamental

This demonstrates why operation-by-operation lowering is costly for training:
a backward consumer can force a large forward image to stay materialized, even
when the derivative needs only a small discrete decision. Single-consumer
pointwise fusion cannot remove that contract by itself. Saved values and
multiple outputs need to be planned together.

But this particular pattern saves only 2.75 ms; **38.55 ms still separates the
candidate from retained JAX**. Do not extend pooling, row-only or tile sweeps.
The next structural target is the existing measured convolution staging/reuse
gap, alongside the remaining image layout/storage program. The current checked
convolution still stages scalar values through 32×32 workgroup tiles with two
barriers per 32-term stage; the earlier actual-library counters showed a very
different data-movement implementation. Those old counters are not measurements
of this candidate or proof of the next kernel's speed.

Use the dominant decoder forward/weight-gradient contracts to qualify one
shared typed staging/reuse intervention, then require a complete-update gain.
Keep the separately measured 5.38 ms host decoding/target path as a concrete
GPU-residency opportunity. Neither alone is an additive budget or proof that
parity is attainable. The root-cause goal stays active; no production repin,
learning campaign or new broad capture follows automatically.
