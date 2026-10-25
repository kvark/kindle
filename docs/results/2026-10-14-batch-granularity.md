# Larger batches expose a convolution fast-path limit

October14,2026. [Declaration](../experiments/2026-10-14-batch-granularity.md) ·
[Data](2026-10-14-batch-granularity.json) · [Backend plan](../backend_training_performance_plan.md).

Follow-up: the [qualified packing-grid correction](2026-10-14-packing-grid.md)
saves only1.49% of full B32 time. It does not explain the dominant gap; the
original measurements and untested-at-the-time hypothesis below remain.

**The remaining gap is not primarily a small-batch problem.** Increasing the
RGB12M replay batch8→32 improves native throughput1.33×, but JAX improves1.87×.
The native/JAX latency ratio grows1.79→2.52×. Both execute four times the
nominal heavy arithmetic; sequence length16 and imagination horizon15 stay
fixed. Native uses one actual B32 world microbatch, not four B8 microbatches.

| Replay batch | Native complete update | JAX complete update | Native/JAX | Native / JAX ms per replay position |
| --- | ---: | ---: | ---: | ---: |
|8|22.780ms|12.737ms|1.79×|.17797 / .09951|
|32|68.456ms|27.187ms|2.52×|.13370 / .05310|

These are ordinary16-warmup/256-measured comparisons, with cached native
commands and all four world submission chunks. JAX includes seed handling,
metric materialization and the final pending drain. Its B32 core-only26.513ms
is **not** the complete comparison boundary. Model/optimizer initials are exact
within each framework; cross-framework weights/RNG remain different. Larger
batches change gradient statistics, scheduling and feedback latency. This is
throughput evidence, not unchanged-learning speed or time to score.

## A concrete scaling defect

Four convolutions that select the packed path at B8 fall back to the
checked path at B32: decoder32×32 forward/weight-gradient and encoder32×32
input/weight-gradient. Together they account for80.531G nominal MACs per update.
This is not an extra-work count: it is work selecting a different implementation.

The cause is exact and visible in the selected plans and admission code:

- A padded activation contains512×1,024×64 =33,554,432 BF16 elements,64MiB.
  It satisfies the existing32Mi-element packing bound.
- The packing shader handles two elements per invocation,256 invocations per
  workgroup. It therefore needs65,536 workgroups.
- Its current launch uses X only, and admission rejects anything above65,535.
  The fast convolution is consequently rejected too, despite substantial
  sampled memory headroom.
- Meganeura already has a general `linear_grid` helper that represents this as
  **32,768×2×1**. The packing shader currently reads only `global.x`; it must
  flatten the actual grid before using that helper.

No larger memory allowance or looser numerical bound is needed for this grid
correction. The effect on whole-update time is **not yet measured**; do not
attribute the entire B32 gap to these four fallbacks. B8 still has a large gap
without them. The separately reviewed initialization cap below is a different
limit, not the explanation for these runtime selections.

Native dispatches actually fall5,286→5,163 at B32. Source-level cooperative
product slots per nominal MAC fall1.80→1.42, while the relative latency worsens.
Neither fewer dispatches nor less padding alone predicts performance. These
are source/grid counts, not hardware instructions or JAX issued-work ratios.

Native posterior/world stages grow3.572/11.466→9.224/36.847ms; imagination/
targets6.428→17.854ms and behavior/sync1.314→4.531ms. Stage walls include their
host work and dependencies; they are not isolated shader costs or utilization.

## Connected projection and JAX work checks

The preceding CPU audit finds128 native product→partial-sum→BF16-conversion→
pointwise chains. Typed/cooperative products do not use the legacy epilogue
fuser. But actual reachable JAX also uses separate reductions, tape writes and
backward kernels; it is not one monolithic projection. Saved/shared values
constrain both graphs. Earlier whole-learner row fusion saved only.99%, so these
counts did not justify another local fusion implementation.

JAX's first B32 preflight stops before updates: six metadata totals change
because XLA stops packing some heads together. The exact total remains4×.
Producer tracing proves B8 combines four forward heads across128 posterior+
2,048 imagined rows and two behavior gradients. B32 retains four-head packing
for512 posterior rows, but uses separate heads for8,192 imagined rows and
separate behavior gradients. All changed shapes/counts reconcile exactly;
there is no percentage allowance. The corrected validator changes no algorithm.
Only the two JAX arms run in the new queue; native timings are not repeated.

## Qualification and retained failures

The first native B32 initialization exits101 before updates because the isolated
typed convolution compiler charges75,366,400 logical partial-buffer bytes
against a64MiB cap. B8 needs18,841,600. An isolated Kindle build raises only
that configured cap to128MiB, using the identical backend. The4096-position
accumulation bound, kernels and precision stay intact. This is logical scratch
before aliasing, not peak physical memory. Its24-update B32 smoke passes.

Both fresh B8 controls reproduce their retained numerical results exactly:
native272 reports/first outputs/five plans and pipelines/all292 tensors at
initial/first/final; JAX272 reports and initial/final288 tensor fingerprints.
B32 retains exact per-framework initials and finite outputs/reports/states;
the native smoke's first24 reports and first state reproduce exactly. B32's
first target/state sizes and all heavy work scale4×. No cross-batch trajectory
or learning-equivalence claim is made.

Seven guards: five pass, two reviewed pre-update failures remain. Five allocation
warnings, no observed GPU hard fault/recovery, zero separate NVML queries or
unfinished children.1,112 synthetic updates, zero game actions. Minimum sampled
estimated Vulkan headroom is12,665,225,216 bytes for native B32 and7,930,773,504
for JAX B32, not physical free/peak VRAM. Three batch-audit classifier/schema
failures and two earlier projection-reader failures retain their sources;
repaired checks pass without numerical relaxation.
Formatting/release Clippy and source/accounting audits pass. Nativee5d890dd;
production and learning stay unchanged. CI398 passes on preceding712b5cd.

## Decision

Do not launch a batch sweep or concurrent seeds. Correct the specific packed-
layout grid limit using the existing general grid helper and type-generic
shader, then qualify address coverage/padding/transposes and complete updates.
Keep the memory cap, accumulation partitions and fallback rules otherwise
unchanged. Measure the full B32 learner before claiming a performance benefit.
This follows a demonstrated implementation defect, not a tile-size search.

The broader root-cause goal remains open. Library-kernel efficiency and device
dependency execution remain established structural differences; neither larger
batches nor a single grid correction establishes parity. No Atari restart.
All raw plans, HLOs, proofs, exact checks, failed guards and source snapshots:
`runs/connected-projections-20261014.lmjwrepW`. Upstream7574ad1/e7c3f7b rechecked
unchanged; main does not yet contain this isolated packed derivative lowering.
