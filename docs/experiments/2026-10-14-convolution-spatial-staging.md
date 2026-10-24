# Follow physical order when staging convolution operands

The [connected spatial trial](../results/2026-10-14-spatial-tape.md) leaves
65.736 ms native versus retained JAX 27.187 ms at B32. This bounded intervention
targets the remaining measured decoder forward/weight-gradient cost, not another
pooling, tile-size, split-count or precision sweep.

The checked path stages scalar values in logical matrix order. For weight
gradients this scatters image reads across channels/filter offsets even though
consecutive reduction positions are spatial neighbors. Test a shared typed
four-value loader that stages this operand in physical order and loads the
cooperative matrix with the corresponding transpose. Forward can use the same
contiguous spatial load. Preserve 32×32 tiles, K32 stages, native16 products,
partitions, compensated F32 accumulation and exact rounding boundaries.

Admission requires stride 1 and four-element-compatible row extents; retain
checked scalar handling for padded window edges and the existing implementation
outside the supported contract. No global layout conversion, increased packing
budget, changed image detail or BF16-only execution engine. Existing competitive
encoder-forward/input-gradient paths remain controls where admission does not
apply. Upstream99e3de27/e7c3f7b was rechecked unchanged before work.

Artifact root: `runs/conv-spatial-staging-20261014.OA6I86kN`; base is the qualified
spatial-tape candidate, not production or the neighboring user checkout.

CPU shader/address/admission tests precede one guarded numerical process
(180 seconds; 200 cases, 600 calls, 100,960 F64 values), then a fresh pair
of the retained five-operation boundary fixture (300 seconds each, 16 warmups
+128 measured calls and two input-change checks per operation). Require exact
retained outputs and the unchanged independent F64 gates before interpreting
timing. Qualify all affected derivatives, not only forward execution.

If the numerical checks pass and the two targeted decoder operations improve
at least 10% in summed matched GPU-pass time, run one ordinary B32 control/
candidate pair (16 warmups+256 measured synthetic updates, 300 seconds each).
Complete-update improvement of at least 3% is the keep gate. Independent source,
plan/pipeline, finite tensor/report and initial/first/final state audits remain
required. No relaxed tolerances, automatic retries or tuning sweep after failure.

Use persistent serialized guarded services, the expected5080, recorded allocation
warnings and at least2GiB sampled Vulkan estimated headroom. No CPU quota on GPU
workers; preparation/analysis uses one CPU,2GiB, no swap. No separate NVML polling,
unsafe Nsight Vulkan capture, production repin, Atari allocation or goal closure.

The numerical process completed all 200 cases/600 calls, then failed my final
bookkeeping assertion: these shapes contain93,280, not100,960, checked outputs.
Keep that failed guard/binary/source and declaration; do not rerun it. Before
timing, independently reconstruct every saved output against F64 on CPU with
the same pointwise/L2 gates, check exact controls/input changes and review logs.
Per-case headroom assertions ran, but their minimum was not saved after the
counter failure. Only a passing independent review permits the already-declared
boundary pair; no shader, shape, tolerance or budget change.
