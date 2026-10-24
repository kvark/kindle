# Keep useful training intermediates visible to the compiler

October14,2026. CPU design check after the
[non-heavy attribution](../results/2026-10-14-materialization-cost.md).
No GPU allocation, production change or learning restart.

The retained native plans contain132 paired RMSNorm input/weight derivatives;
all135 input-gradient regions consume a separate SiLU derivative. Their
dedicated kernels recompute normalization statistics and sigmoid respectively.
The matching retained JAX backward fusions combine slicing, activation gradients,
casts and reductions, with saved forward tensors and no rsqrt/exp in those
fusions. Trace saved row-statistic provenance through actual call/loop bindings;
do not infer the whole program's recomputation count from one fusion.

Meganeura already provides primitive decompositions. Its training build
recognizes composites before autodiff, which selects these opaque derivatives.
Before designing a new kernel or IR, compare composed versus primitive
autodiff for one connected RMSNorm -> SiLU -> weighted quadratic loss block
at actual128x256 extent, with x and normalization scale trainable. Use current
upstream7574ad1 in an isolated checkout, existing decomposition/optimization/
compile paths, and independent F64 derivatives. Inspect surviving shared
statistics, dispatches, dependency groups, fusion boundaries and logical buffers.
Preserve any extra work created by recomposition; counts are not timing claims.

One CPU,2GiB,zero swap; no unfiltered backend tests. No GPU follow-up is implied
by a passing CPU fixture. A whole-agent candidate needs a separate declaration,
numerical qualification, finite/exact state review and complete-update timing.

Upstream rechecked: Meganeura7574ad1 merges explicit BF16, shared matrix groups,
direct generated WGSL and shared backward widening. Bladee7c3f7b/Nagacbbac959
adds Metal BF16. Review/adopt relevant fixes before touching the same area;
this latest-main CPU fixture is not a qualified replacement for historical
isolated binaries. Preserve neighboring branches and all prior evidence.

Artifacts: `runs/row-region-audit-20261014.oDslX6Vm`.

The first check passes132,100 independent F64 values, but primitive expansion
raises optimized dispatches14->28 and groups7->16. Forward recognition also
creates a SwiGLU from the weighted quadratic; its derivatives remain opaque in
that first selected decomposition. One CPU-only clarification includes SwiGLU
in the same decomposition, keeps the first outputs/source/binary, and checks
the same derivatives and plans. This is not a GPU timing or tuning allocation.
The initial workspace/build failures and corrected HLO tape-label assertion
are retained; none ran GPU work.
