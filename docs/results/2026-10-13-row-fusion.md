# Typed row fusion: useful, not gap closure

October13,2026. [Declaration](../experiments/2026-10-13-row-fusion.md) ·
[Data](2026-10-13-row-fusion.json) · [Performance plan](../backend_training_performance_plan.md).

General typed reduction prologues/epilogues improve complete RGB updates
**33.684→32.848ms (2.48%)**, and Mixed CDP **111.023→105.172ms (5.27%)**.
RGB remains **2.65×** retained JAX12.417ms. Production remains on the earlier
qualified runtime; this isolated candidate does not change pins, defaults,
learning allocations or the stopped Atari campaign. The root-cause goal remains
active. No speed claim from dispatch counts alone.

## What changed and what it establishes

One general compiler mechanism absorbs single-use pointwise operations,
broadcast bias and explicit BF16 conversion into existing row reductions.
Inputs/outputs may be F32 or BF16; accumulation remains F32. Pipeline identity
includes storage types. It preserves explicit intermediate rounding, protected
outputs, fan-out and backward saved values. No dedicated Dreamer kernel, matrix
geometry change, precision-policy change or real-work barrier removal.

| Ordinary complete update | RGB12M/B8 | CDP12M/B32 |
| --- | ---: | ---: |
| Staged control |33.684ms|111.023ms|
| Typed row fusion |32.848ms|105.172ms|
| Imagination plus targets, control→candidate |10.939→10.166ms|46.023→40.199ms|
| Total dispatches, control→candidate |6,148→5,846|7,053→6,555|

One ordinary272-update pair per workload,16 warmup/256 measured; no profiler,
parallel GPU work or concurrent compilation. Matrix/convolution descriptors,
workgroup geometry and nominal arithmetic are identical. The RGB staged
posterior is counted once, not again as a separate forward graph. This is not
a fresh JAX run or matched CDP/JAX comparison.

RGB imagination falls1,367→1,087 dispatches and1,070→854 dependency groups;
world changes only4,508→4,492 dispatches, retaining1,433 groups. Slow-value
groups15→9, behavior37 unchanged. Thus the large world-backward command stream
is mostly untouched. The remaining imagination plan contains149 concatenations,
60 split kernels,60 broadcast kernels and157 standalone BF16 conversions.
These counts identify materialization candidates, not their measured individual
costs. Do not infer that all are redundant or remove their dependencies.

The prior [zero-work control](2026-10-13-residual-gap.md) isolated5.021ms of
barrier cost with no shader work. This real-work trial confirms that reducing
some boundaries helps, but cannot turn that diagnostic number into5ms of
promised savings. The problem is broader than a missing normalization kernel:
execution still materializes many small operations, crosses host/device stages,
and retains slower matrix/convolution algorithms.

## Qualification and retained failures

Both24-update smokes and both272-update candidate/control pairs have exact
metric reports, first outputs and every initial/first/final saved tensor:
292 RGB/346 CDP. That is fixture parity, not learning or universal bitwise parity.

CPU checks pass20 generated shader/plans, four row tests,60 analytic-backward
comparisons,27 scheduling tests,13 cache tests and both library Clippy checks.
GPU coverage comprises84 cases,108 ordinary F64 comparisons,60 conditional F64
loss/gradient comparisons and1,058,092 exact CPU BF16 RNE checks. Poisoning,
expected5080 device and≥2GiB sampled Vulkan budget headroom are enforced.

Two initial numerical qualifications failed and remain retained. The wide
128×2048 case crosses a BF16 rounding tie: ideal F64 activation1.035156471
rounds up while both F32 executions produce the exact tie1.03515625 and
correctly round down to even. Fused/unfused pre-round/output arrays match
bitwise; all pre-round F64 and CPU RNE checks pass. The original ideal-F64
weight-gradient also fails identically in both paths. An independent analytic
RMSNorm/SiLU/MSE chain, conditioned only on the separately validated stored
activation, predicts.000168939660761 versus GPU.000168939659488. Both complete
gradient arrays match between fused/unfused.

The completed qualification uses original tolerance values and error scales;
BF16-output derivatives are explicitly conditional on validated forward
rounding. Four wide ideal-F64 reports still fail, including both input types
and both lowerings. They are not relabeled as unconditional F64 passes.
Runtime code never changed during these test-contract corrections.

Nine passing guards, two retained failed numerical guards, two allocation
warnings,1,136 full synthetic updates and zero game actions/optimizer updates
in the small numerical fixtures. Initial CPU Clippy failure and a2GiB-cgroup
inventory OOM remain; streaming inventory completes with22.5MiB peak. Neither
CPU failure is a GPU incident. No NVML polling, recovery or remaining worker.
Publication/source/binary audits pass. Artifacts:
`runs/row-fusion-20261013.vif6rSDw`; hashes and retained failures are in the JSON.
Upstream rechecked22:22 UTC: Meganeura2214412/Bladefe5f5b5 remain unchanged.

## Next boundary

Keep the useful general fusion, without a tuning sweep. Investigate the
remaining materialized layout/broadcast/cast and saved-value boundaries,
especially world backward; carry data through consumers instead of launching
a kernel for each transformation where contracts allow. Then finish
device-side targets/RNG/slow-critic EMA to remove forced host staging.
This is a connected-learner/compiler change, not just larger batches or a
dtype flag. Measure one complete update after each coherent change; retain
numerical controls and do not sum prototype savings. No new Atari allocation
or whole-goal completion.
