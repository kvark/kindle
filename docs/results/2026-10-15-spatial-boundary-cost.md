# The connected image path is the large world-model cost

October15,2026. [Declaration](../experiments/2026-10-15-spatial-boundary-cost.md)
· [Data](2026-10-15-spatial-boundary-cost.json).

**Replacing the complete image/convolution region with its exact boundary values
reduces fixed-world GPU time34.195→6.564ms,80.8%.** Replacing only non-convolution
image work saves4.173ms. Intermediate materialization matters, but arithmetic-free
non-convolution fusion alone does not explain the whole learner gap. Convolution
algorithms, their data feeding and connected buffer boundaries remain the larger
joint target. This is an oracle intervention, not a production speedup.

| Fixed-world condition | Keep all intermediate copies | Copy only escaping values |
| --- | ---: | ---: |
| Original |34.207ms|34.195ms|
|23 convolutions substituted |14.275ms|14.289ms|
|286 non-heavy image operations substituted |33.677ms|30.022ms|
|Both substituted |13.979ms|6.564ms|

The left column is the preceding [all-output study](2026-10-15-spatial-value-cost.md),
not a fifth arm rerun in this allocation. Unchanged normal controls differ by
−0.035%; the convolution-only cut is identical. All samples remain. With real
convolutions present, the closed non-heavy intervention saves4.173ms; with them
already replaced it saves7.725ms. Interaction is−3.552ms: removing both regions
also removes their connecting values. Do not sum conditional differences or
subtract JAX kernel sums into a full-update savings budget.

The all-output copier still writes344 buffers/3,154,096,284 logical bytes. The
combined closed cut retains46 buffers/7,014,164 bytes;263 selected dispatch slots
need no copy at all. The spatial-only cut retains86 outputs/705,121,428 bytes
and drops235 internal outputs. Reducing copies saves7.416ms in the combined
condition, including changed memory accesses and dispatch work. This is not
measured DRAM traffic, a pure bandwidth cost, or evidence that real convolution
training can execute without its necessary intermediates.

## Validity, not zero-work masking

Same RGB12M/B32/T16/BF16 graph, checkpoint, actual first update, original
allocations/groups/barriers and typed kernels. The Python cut proof and an
independent runtime simulation require the original logical version at every
still-executed read. Matrix prologue/epilogue/norm operands, multiple writers,
primary/extra outputs and892 protected graph/parameter/gradient/input/constant/
derived/LSE buffers are covered. CPU rejection tests exercise missing escaping
values, wrong producer versions and missing final outputs.

For each nonempty condition, discarded snapshot words are replaced by0x7fc17fc1.
The all-output copier writes those poisoned values at their original slots;
live checks verify the poison and every retained/unselected output. Complete
graph outputs remain finite and exact. Original execution then recaptures every
snapshot exactly before closed substitution. Thus stale valid intermediates do
not explain the passing result. All four closed conditions pass live/full-output
checks, every measured block passes complete outputs, and final normal restoration
passes. This does not weaken original synchronization.

All292 initial/first/before-after tensors, first targets/report and full
plans/pipelines/groups match the retained current control. The8 shader/selector/
version tests,3 Python dataflow checks, formatting and Clippy pass. The copy shader
is unchanged from its3906-value GPU qualification. Normal resident timing passes
the5% gate: +0.071% versus pre-snapshot34.170ms,−0.035% versus retained34.207ms.
One passing guard, zero allocation warnings; minimum sampled Vulkan estimated
headroom9,575,661,568B. One initializer update,733 fixed-world replay equivalents,
576 measured samples, zero game actions. No current SM-utilization measurement.

GPU service finishes10:35:20UTC in75.565s. CPU review finishes10:38:05 in3.556s,
291.8MiB peak; no worker remains. Diagnostic native50a0df64, artifacts
`runs/spatial-boundary-cost-20261015.dyPFMiZW`. Production and Atari are unchanged.

## Retained failures and limits

Two explicit-reference Clippy attempts remain with source/log snapshots. The
first completion audit catches an orchestration-only seal mismatch: an in-flight
edit added a wait for CPU publication and was then reverted. The exact sealed
intermediate source is reconstructed/hash-checked; its only AST difference is
that wait. All GPU/review commands and all4123 source receipts reconcile,
including that explicitly mapped snapshot.
The first repair incorrectly queried already garbage-collected transient-unit
timestamps; retained journal records and the publication's final flushed output
resolve chronology instead. Both failed audits remain. No GPU rerun, changed
model result, tolerance relaxation or repinning of original evidence.
The publication check's raw-versus-compact JSON key error and corrected audit
also remain; measured data are unchanged.

This result includes RGB reconstruction, absent from CDP. It is not a CDP cost
share, a frozen learning result or a whole-update optimization. Complete RGB
remains54.113ms versus inclusive JAX27.223ms,1.99×. The result identifies a major
connected region; it does not prove that an available native implementation
can match cuDNN's pass-specific reuse/asynchronous pipeline.

Next inspect the existing JAX executable's corresponding image layouts and
materialized boundaries against this exact native cut, without another GPU
allocation. Use that contract to choose one complete connected-image lowering
covering forward and both derivatives, accounting for conversions and scratch.
The rejected standalone family, full im2col, halo and invalid-mask routes remain
rejected. No geometry/precision/flag sweep, new Naga project or automatic learner
restart. Kernel/data-feeding and non-image execution costs remain separate open
parts of the parity goal.
