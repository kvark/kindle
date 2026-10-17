# Direct image views help, but do not explain the remaining multiplier

October15,2026. [Implementation](../experiments/2026-10-15-image-views.md)
· [Qualification/timing declaration](../experiments/2026-10-15-image-views-qualification.md)
· [Data and receipts](2026-10-15-image-views.json).

**Keep the isolated candidate: complete RGB updates improve54.205→51.945ms,
4.17%, with every report, target and saved tensor unchanged.** CDP improves
47.699→47.112ms,1.23%, satisfying the no-regression gate. This is a measured
compiler dataflow improvement, not a different model, precision or convolution
algorithm. The remaining RGB gap is **1.91× inclusive JAX27.223ms**; JAX was not
remeasured. Production and Atari remain unchanged.

## One connected producer/consumer change

Image producers now write their already-required converted/expanded packed
views directly. Identical packed versions are shared across forward and
derivative consumers. Four decoder backward regions consume the existing
four-pixel sum without materializing an intermediate. The shared typed image
generator supports these views; there is no BF16-only parallel implementation.

| Complete RGB stage | Control | Direct views |
| --- | ---: | ---: |
| Posterior |7.531ms|7.315ms|
| World training/sync |30.156ms|28.120ms|
| Imagination/targets |11.947ms|11.909ms|
| Behavior training/sync |4.572ms|4.601ms|
| **Complete update** |**54.205ms**|**51.945ms**|

RGB changes13 image kernels, removes31 dispatches and shares eight packed
versions. CDP changes5/removes11/shares4. The RGB rewrite removes876,609,536
logical output bytes; this is not measured DRAM traffic or a VRAM saving.
Buffer inventories, required F32 tapes, external NCHW values, normalization/
pooling/SiLU math, BF16 rounding and left-associated gradient sums remain.
Convolution selection/arithmetic is unchanged. The rejected2D packing repair
is not included.

## Qualification and complete-program review

Both arms use isolated native67559ce5; only `MEGANEURA_IMAGE_VIEWS=0/1` differs.
Device means and command reuse stay enabled. RGB12M/B32/T16 and trained centered
CDP12M/B32/T16 retain their checkpoints, configurations and seed701 changing-batch
fixture. Each24-update numerical pair passes before its full16-warm+256-measured
pair. No per-dispatch profiling or NVTX; full plans export outside the timer.

Every paired272 non-timing metric report, first target and initial/first/final
292 RGB/346 CDP tensors is exact, including optimizer state. Each arm reproduces
its own24-update qualification prefix; control also reproduces the retained
076f3135 trajectory exactly. This is implementation equivalence on these fixtures,
not an independent full-model F64 oracle or learning-quality claim.

The CPU audit reconstructs the rewrite independently from the original graph.
All surviving reads, protected values and aliases reconcile. Every unchanged
dispatch descriptor/pipeline stays exact; only declared image views, aliases
and dependency regrouping differ. Non-world programs and graph-visible metadata
are exact. Full RGB control plans also match the retained ordinary control;
no missing historical CDP export is invented. Qualification and timing programs
match in both arms. All2441 timing source seals pass.

All256 measured samples/arm remain. RGB first/last32 means are54.171/54.107ms
control and51.892/51.983ms candidate. These are stability summaries, not confidence
intervals or SM utilization. No retime or favorable subset was selected.

## What this changes about the diagnosis

Repeated image conversions are a real cost, now causally demonstrated. But a
substantial connected rewrite recovers2.26ms, not the remaining24.72ms over JAX.
The broader [image-region attribution](2026-10-15-spatial-boundary-cost.md) and
[actual library counters](2026-10-15-library-convolution-feeding.md) still point
to convolution data feeding/reuse and device execution as distinct large costs.
Do not add these interventions into a fictitious savings budget or claim that
this compiler pass reproduces the library's asynchronous multistage pipeline.

Retain this candidate without a layout/geometry/precision sweep. The next
structural review must distinguish what our native execution interface can
express from what its generated algorithms actually achieve. A credible route
to parity needs a complete pipeline, not another favorable isolated tile.
The root-cause goal remains open; no production promotion or learner restart.

## Receipts

CPU proof covers253,755,392 addresses,7,352 RGB surviving reads and892 protected
buffers; the CDP program audit covers8,103 reads/940 protected buffers. Five CPU
tests,144 existing/432 new shader variants, formatting and both release Clippy
checks pass. The original Clippy failure/source and explicit-reference repair
remain; no failed GPU run or tolerance relaxation occurs in this allocation.

Eight guards pass; seven standalone allocation warnings remain.96 qualification
plus1088 timing updates, zero game actions/recovery/separate NVML polling.
Timing minimum sampled Vulkan budget headroom12,378,701,824B is not physical
free/peak memory. Numerical service finishes12:18:38UTC; timing finishes12:30:39
in101.241s; CPU review finishes12:31:48. No worker remains.

Sources, binaries, complete exports, checkpoints and audits remain under
`runs/image-views-20261015.EblKUBZN`. Upstream main rechecked unchanged at
Meganeura99e3de27/Bladee7c3f7b; neighboring user work is untouched. Enable
`MEGANEURA_IMAGE_VIEWS=1` and `KINDLE_DEVICE_MEANS=1` for this isolated candidate;
default control behavior/pipeline keys remain unchanged.
