# Spatial-map shaders: offline proof complete, GPU family running

October16,2026. [Declaration](../experiments/2026-10-16-convolution-metadata-kernels.md)
· [Compact evidence](2026-10-16-convolution-metadata-kernels.json).

**The actual consumer and GPU initializer pass offline qualification.** The
bounded complete-family GPU comparison is running; it has no aggregate speed
or qualification result yet. Keep native58aff452 and **RGB46.199/JAX27.223ms,
1.70×**. Production and Atari remain unchanged; the root-cause goal stays open.

## Implemented and checked

- One type-agnostic transform covers typed and packed forward/dX/dW. It changes
  only B spatial feeding, not the rejected A-address carry rewrite. Original
  cursors, A loads, matrices, K order, splits, compensation, stages and output
  handling remain exact source prerequisites.
-320 paired F32/F16/BF16 modules plus the initializer validate offline;266 old
  control modules reproduce exactly.302 consumers change;18 dense/grouped
  controls remain byte-exact. All20 current contracts/23 occurrences are covered.
- Actual emitted expressions match26,163,200 table indices and30,428,168 source
  addresses. First/last batches are explicit; interior batches use bounded affine
  equivalence.36 small cases add177,993 checks. The initializer's actual source
  matches all272,000 map entries and7,624 masked tail invocations across actual/
  small cases. Four expression/initializer mutations reject.
- Independent source/SPIR-V review preserves non-B code, memory widths/bindings,
  cooperative operations and barriers. Only the declared u32 map reads are added;
  consumers cannot write the table. Binding4 and the48-byte initializer geometry
  layout reconcile. Old address cursors remain an unchanged qualified premise,
  not a newly independently implemented convolution oracle.
- Observer**fb6d0cc1** links unchanged backend libraries. Its CPU geometry test
  passes. The runtime requests device-local storage, GPU-initializes it, compares
  readback bits with the independent map and waits before exposing it. All
  consumer submissions track a final fence before destruction. Actual full-family
  runtime qualification is still required; no cross-Session cache is implemented.

The first CPU reviewer compared module-local vector-component type IDs11/12;
the repaired semantic comparison retains dtype/width checks and seven regressions.
The next reader failed on rustfmt's wrapped `let values = unsafe`; a separate
whitespace-only reader repair passes. Both original scripts/logs remain; neither
caused a shader rebuild or GPU repeat. Offline review ends20:24:48UTC.

## GPU execution boundary

The first numerical pair passes two guards with zero allocation warnings:
524,288 independent F64 comparisons,67,108,864 full exact output pairs and102,400
exact GPU-initialized map entries. This is one forward contract, not qualification
of the entire family or F32/F16 GPU numerics.

Its CPU reviewer then stopped because serde stores an OsString as tagged Unix
bytes. The separate path-schema repair validates the tag and bytes, compares the
sealed path and finishes the original F64/plan/output review. Original controller,
finisher and reviewer failures remain. **No GPU fault, recovery or repeated pair.**

`kindle-metadata-gpu-remaining-20261016` starts**20:32:29UTC**, retaining that pair
and running only138 unstarted guards. Its persistent finisher independently
aggregates all20 contracts after completion.2,414 source seals pass. Full counters,
ordinary samples, calibration and the5% keep gates remain pending. All one-time
initialization/readback costs are recorded separately; repeated operation timings
do not silently include a shared eight-map full-world cache.

Next review the completed aggregate or the first failure before further GPU work.
No automatic integration, whole-update trial, favorable subset, retime or learner
restart. Artifacts: `runs/convolution-metadata-kernels-20261016.c0XlqDMn`.
