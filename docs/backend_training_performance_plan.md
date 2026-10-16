# Backend training performance plan

Updated October 16, 2026. Improve one RTX 5080 learner on Meganeura + Blade
before returning to Kindle learning campaigns. Keep CDP, model capacity,
replay ratio, image detail and exploration semantics unchanged. This is an
implementation plan under the [roadmap](kindle_single_life_dreamer_plan.md),
not a new learning allocation; PR31 remains the status dashboard.

## Current diagnosis and implementation order

The goal remains open. The latest isolated complete RGB12M/B32/T16 update is
**46.199ms versus inclusive JAX27.223ms,1.70×**; B8 retains22.780/12.737ms,1.79×.
Production is not promoted and Atari stays stopped. CDP's larger exploration
workload is different and must not be compared directly with these RGB numbers.
The current isolated B32 candidate is native`58aff452` from
[private copy composition](results/2026-10-16-copy-composition.md), with
`MEGANEURA_COPY_COMPOSITION=1` and
`MEGANEURA_CONVOLUTION_WORDS=1` plus image views/device means/cached recording.
Its preceding control is native`834241b6` from
[typed word staging](results/2026-10-16-word-staging.md).
The preceding candidate is native`67559ce5` with
`MEGANEURA_IMAGE_VIEWS=1` and `KINDLE_DEVICE_MEANS=1`, from
[connected image views](results/2026-10-15-image-views.md). Its control is
`076f3135` from [repeated-row composition](results/2026-10-15-repeated-row-composition.md).
`99f8aab0` retains [GPU categorical means](results/2026-10-15-categorical-gpu.md);
`28864210` retains the preceding [general-ordering trial](results/2026-10-14-convolution-scheduling.md).

**Separate unresolved candidate: [prior batching is incomplete](results/2026-10-16-prior-batching.md).**
The isolated boundary has144→9 matrix products and1181→168 dispatches;
all serial/batched comparisons pass. Independent F64 checks fail14 arrays
in both arms. The subsequent [forward-tape capture](results/2026-10-16-prior-tape.md)
now explains the logit discrepancy: local matrices/biases match exactly;
small permitted F32 normalization/SiLU differences cross BF16 midpoints.
The [backward diagnosis](results/2026-10-16-prior-gradient.md) now explains the
first variant's remaining failures: small F32 KL errors change50 BF16 derivative
values, propagating to115/403 earlier-layer operands. All36 local checks and
both-arm matrix rechecks pass unchanged gates; the operand changes account for
all69 failed weights. A source-F32 calculation clears second-variant gradients
in both arms. This is not an end-to-end F64 pass: all original failures remain.
The new capture uses one guard/no warnings/one raw backward/zero updates/actions;
no worker or timing result. The subsequent [actual-world comparison](results/2026-10-16-prior-world.md)
passes eight whole/staged pairs, unchanged states and all56 outputs, but fails
113 raw-gradient arrays at903,879 values. RelativeL2 remains below.007095;
that does not override pointwise failures. Four guards/no warnings/24 raw
backwards, no timing. First-prior execution changes eight F32 partials plus
reduction/BF16 conversion to one tiled/BF16 product; both retain conversion.
This is a candidate cause, not an established full-gradient explanation.
Keep the failed gates, unqualifiedaca80566 and qualified834241b6 distinct.

The [real-input intervention](results/2026-10-16-prior-real-inputs.md) now
attributes the forward divergence to that first product. Its135/163 changed
BF16 values produce8,028/3,549 changed RGB/CDP logits. Substituting only the
serial product makes all3.146M path values, including524,288 logits, exact.
All102 actual-world contracts, live copies, local partials/precision checks and
independent7.340M paired comparisons pass. One guard/no warnings, six closed
forward paths/zero raw backwards or updates. The CPU approximation underpredicts
the changed values; it is not bit-exact GPU arithmetic. Full-gradient attribution
and candidate qualification remain open; no timing or production change.

The [full-world intervention](results/2026-10-16-prior-world-intervention.md)
now reduces pointwise failures903,879→60,682, but78 arrays still fail. All56
world outputs are bit-exact and32 prior-parameter comparisons pass; residual
failures lie in posterior/recurrent/encoder gradients. Ordinary/split controls,
four checkpoint pairs and independent156.977M comparisons pass. Two guards/no
warnings,14 raw backwards/zero updates. CPU inspection proves an additional
posterior-adjoint materialization boundary, not its GPU numerical effect.
Original qualification remains failed; do not time or promote the candidate.

**Graph diagnosis: [the training prior is unnecessarily unrolled](results/2026-10-16-nonconvolution-contract.md).**
Native144 matrix products versus JAX9 do identical1.107G nominal MACs outside
the posterior recurrence. This explains135 of159 extra matrix products, not
their cost or the whole gap. All28,064 JAX kernels/eight updates reconcile;
current non-world plans match the coarse capture. Imagination's180 products
also match, with only action-padding arithmetic different. Its native pass
includes heads while the JAX2.852ms kernel sum is recurrence-only: no direct
ratio or additive saving. Basic head batching/cooperative products already exist.
The original loop-scope observer defect is retained; separate repair and
independent checks pass. CPU only, no new measured speedup or GPU allocation.

**Preceding diagnosis: [column-major packet axis is rejected](results/2026-10-16-packet-axis.md).**
Complete convolution19.089→21.217ms,11.15% slower, with all numerical checks
passing. Restoring dW uniform-pipe work does not reduce total instructions:
they rise17.06%, time1.01%, while shared conflicts stay flat. dX instructions/
time rise despite less L2 traffic. Uniformity, occupancy or traffic alone is
not the missing mechanism.120 guards/22 warnings, exact outputs and
offline counter audits pass; GPU ends06:38:36UTC, no worker. No full-update
trial, selector or promotion. Keep834241b6 and the1.77× complete gap.

**Latest qualified update: [word staging](results/2026-10-16-word-staging.md).**
RGB51.763→48.068ms,7.14%; CDP46.929→45.505ms,3.03%, with exact reports,
targets/tensors and full-program/source checks. Complete convolution operations
improve15.03%, including all23 occurrences and one regression. CNN1 dX nearly
halves latency, with46% fewer warp instructions and82% fewer shared conflicts.
dW conflicts fall83%, but latency only9.1%; ordinary ALU work rises while its
uniform-pipe work disappears. This is consistent with a lane-mapping/addressing
tradeoff, not an isolated source-instruction attribution.124 guards pass, five
warnings; GPU ends05:53:42UTC, no worker. Production/learning stay unchanged.

The preceding [actual native counters](results/2026-10-16-native-counters.md)
established the baseline.
The supplied Perf SDK2026.3 works directly on580.178.04. All20 contracts/23
occurrences retain exact outputs/plans/pipeline keys;21 guards/no warnings.
Native CNN1 dX/dW execute15.94x/9.04x the warp instructions of the retained
full-JAX kernels, with fewer tensor instructions. This is native evidence,
not the old CUDA proxy or a latency ratio. Nine scalar-staged occurrences
execute10.214G warp instructions versus1.596G for14 packed occurrences.
The two derivative captures show<6% peak DRAM throughput and122M/158M
shared-bank conflicts; conflicts are not yet isolated removable latency.
No missing-tool or GPU-health/approval blocker remains; all workers exit.

The [imagination boundary audit](results/2026-10-16-imagination-boundaries.md)
now separates all802 native dispatches and accounts for45 CUDA rollout copies
per update. JAX recurrence is2.907ms of GPU activity within3.147ms elapsed,
still not directly comparable with native8.740ms including heads. A concrete
composition miss remains:15 private RSSM input chains use105 dispatches and
747.110MB logical writes where15 JAX fusions write125.829MB. All62.915M addresses
and source/consumer checks pass; these are not traffic or speed measurements.
Rollout stacking is already balanced, but independently constructs overlapping
15/16-state outputs. Six mutation tests/three label regressions/63 seals pass;
the failed registered-NVTX-label observer remains. CPU only, no worker.

The [native valid-value attribution](results/2026-10-16-imagination-cost.md)
now passes: complete fixed imagination8.702ms, matrices substituted5.932ms,
surrounding operations6.639ms,both2.275ms. The2.770/2.063ms conditional differences
have a−1.594ms interaction; neither they nor the remaining2.275ms are pure kernel
costs or additive achievable savings. Both classes matter. Exact full/live
outputs, checkpoints, first targets/reports and programs pass; two guards/four
warnings,733 fixed replays/one initializer/zero actions, no worker. All576 timing
samples remain; calibration gates pass. Original observer compile failure stays.

The [general copy/cast implementation](results/2026-10-16-copy-composition.md)
now qualifies: complete RGB48.133→46.199ms,4.02%, CDP45.669→43.939ms,3.79%.
It removes362/458 dispatches while preserving all272 reports, targets and
292/346 state tensors exactly. Actual raw gradients, changed late targets and
poisoned preparation pass47.082M values per comparison. All319.752M CPU addresses,
615 typed variants, source-version/plan/pipeline audits and world-group coverage
pass. Non-world runtime groups were not exported. Shader ordering/scope and
reader failures remain, with CPU-only repairs; no GPU repeats. Twelve guards,
ten warnings,1,188 updates/16 raw backwards/zero actions; no worker. The gain
is mostly imagination/targets, not the remaining18.977ms RGB/JAX difference.

The [kernel-memory audit/capture](results/2026-10-16-kernel-memory.md) now rules
out local-memory traffic for all20 convolution contracts, with a validated
positive control.99.711% of remaining shared-bank conflicts occur on writes.
All nine word paths fit workgroups×(32×K-stages+96); seven held-out contracts
pass, but this is not source-site or latency attribution. Software reciprocal
division is outside their reduction-loop closures. Generic bounds checking/core
robust access are already off. All current heavy descriptors/keys/shader bytes
and retained instruction counts reconcile; no new complete-update measurement.
21 guards/13 warnings,3,050 offline metrics/258.287M exact values pass; no worker.

The [shared-store contract audit](results/2026-10-16-shared-store-contract.md)
now identifies a separate remaining path: nine tiled matrix keys load vectors
but emit32 scalar16-bit shared stores per K-stage. They carry91.04%/93.24% of
RGB/CDP nominal matrix arithmetic, not measured time. All1,519 heavy occurrences,
28 selected modules,29 emitted catalog modules and152 retained typed modules
reconcile. The conditional convolution bank model fits word-kernel slopes but
fails every packed contract; legal subgroup permutations also change predictions.
The output intercept and driver lowering remain unattributed. Original CPU
reviewer failure/repair are retained; no GPU work or backend rebuild.

The [complete matrix counter diagnostic](results/2026-10-16-matrix-feeding-counters.md)
now covers all54 tiled contracts/729 occurrences plus11 checked controls.
Zero local traffic and exact tensor counts pass in all65; weighted RGB/CDP
warp instructions per tensor instruction are11.94/12.72. Shared read/write
conflicts are substantial, not measured time shares. Checked-vector reads have
about12 conflicts/tensor versus tiled1, on different shapes; copying that layout
is not an established win.335.647M output values/6,500 offline metrics pass.
65 guards pass, one observer times out; reviewed bulk-copy repair preserves its
partials. The original exact shared counter identity fails44/65 up to.12737%;
report independent counters, no relaxed numerical gates or GPU review retry.
No new backend, full-update time or learner result; all workers finish.

The [typed storage CPU proof](results/2026-10-16-matrix-vector-contract.md) now
passes all54 selected contracts/729 occurrences and468 typed requests. It keeps
physical-major vector storage and expresses logical orientation in cooperative
loads;32 scalar stores become eight vector stores in emitted SPIR-V, not yet
measured native instructions. Producer/consumer/affine-address checks,463 control/
463 candidate modules,48 storage envelopes and seven negatives pass. All selected
cases fit; two larger transposed-B/scratch geometries newly exceed48KiB. Any
integration must share orientation-aware layout accounting with the existing
capacity guard. Original CPU envelope-count failure/repair remains; no GPU calls,
backend rebuilds or new update result. Keep58aff452 and general copy composition.

The [complete vector-staging family](results/2026-10-16-matrix-vector-family.md)
now rejects that implementation: all54 contracts improve weighted GPU time only
1.68%/1.50% for RGB/CDP; CDP queued wall worsens.16%. Both5% time gates and the
instruction-reduction gate fail. Warp work grows7.18%/8.34% despite substantially
fewer write conflicts; every20 NN contract retains its total LSU count. Nine
emitted pairs keep explicit float operations unchanged while adding vector-unit
indexing; static IR does not explain native source sites. All108 guards,665.846M
integer/reference output checks and10,800 offline metrics pass, with two warnings.
No integration, full-update trial, favorable selector or retime. Zero builds/
updates/actions; GPU ends14:20:56UTC, CPU interpretation14:39:01UTC, no worker.

The [actual recurrence interface audit](results/2026-10-16-jax-matrix-contract.md)
now reconciles12 roles/180 products,20.934M physical addresses and1,440 captured
JAX launches. Padding18→24 accounts for the tiny MAC difference. Grouped layouts
match physically, but JAX's beta=1 addend is outside the native matrix boundary.
All epilogues are DEFAULT; do not infer ReLU from a kernel name.12 graph nodes
and their enclosing command buffers are exact; individual HLO ownership remains
unresolved. Existing native counters cover150/180 recurrence products, not the
two actual512-row action contracts. Seven negatives/19 seals pass; repaired CPU
upstream-file assertion remains, no GPU work. Meganeura's six commits through
12df95c are reviewed; no repin or retrospective numerical-gate relaxation.

The [actual recurrence counters](results/2026-10-16-jax-matrix-counters.md)
now cover all180 products: native277.979M versus JAX300.272M warp instructions,
**7.42% fewer native**, nearly identical tensor counts and zero local traffic.
Convolution's instruction overhead does not generalize to these matrices.
Native shared-read conflicts are24.36x but total read wavefronts only1.0174x;
neither proves a latency cause. All272 JAX reports/288 tensors/full HLO and
561,152 new native independent outputs pass. Three guards/no warnings, zero
builds/actions. Capture ends15:15:27UTC, no worker. Retain the collected-transient-
service observer failure; unchanged CPU review passes without a GPU repeat.

The [fresh complete trace](results/2026-10-16-current-update-attribution.md)
now reconciles current execution: ordinary46.108ms/coarse47.176ms,+2.315%
instrumentation. GPU-pass union39.650ms exceeds the entire inclusive JAX update;
uncovered7.528ms cannot by itself explain parity. Cached world27.935ms and
imagination6.976ms dominate; decoder-containing15.505ms/encoder-derivative6.720ms
chunks are not pure convolution costs. Exact reports/tensors/programs pass,
two guards/no warnings/544 synthetic updates. GPU15:38:52UTC, no worker.
Retain zero-test preflight failure (correct test runs only after capture),
formatting failure and crossing-interval reader failure; independent CPU
repairs preserve all timestamps/gates, no GPU repeat. Instrumentation is not
promoted, and timed chunking still adds three waits. JAX inputs are resident;
native input preparation is measured. Neither that boundary nor host-only work
explains the larger GPU-covered interval.

The [complete convolution capture](results/2026-10-16-jax-convolution-counters.md)
now covers all23 mains/18 helpers. Native/JAX warp instructions10.296/2.420B,
4.255x; ALU7.287x, tensor.792x. Every pass has excess work. Nine typed occurrences
contribute84.50% of native instructions; fourteen packed still have3.20x overhead.
These are not time shares. JAX has local traffic despite native zero. Original
36/41 capture remains failed; separate metadata and missing-profile allocations
complete coverage with exact reports/tensors/HLO and240 seals. Two passing/one
failed guard, zero warnings/816 synthetic updates, last GPU16:25:10UTC, no worker.
Retain all CPU reader failures, including unit scaling and metric membership;
offline repairs do not repeat GPU collection.

The [load-contract audit](results/2026-10-16-convolution-load-contract.md) now
confirms scalar global reads in all nine typed paths, without hot-loop BF16
conversion or injected bounds checks. All20 contracts/23 occurrences and152
CPU emissions reconcile.196.947M address positions/independent counts prove
aligned-four composition could reduce modeled source reads28.03%, not hardware
work or time. A58.59%/B11.52% reduction is uneven; repeated predicates/coordinates
remain. Public external bindings use offset zero; general logical extents still
need explicit eligibility. Original sealed-declaration publication error and
CPU restoration remain; runtime bytes are unchanged. No GPU/build/update/action.

The [complete read-span trial](results/2026-10-16-convolution-read-spans.md)
rejects the shared implementation: all23 products cost19.025/18.809/28.187ms
native/control/candidate. GPU/queued wall regress49.86%/49.61%, despite passing
calibration. All nine changed contracts regress; unchanged packed operations
stay stable. Warp work rises18.01%, CBU13.68x, ALU12.14% and LSU15.27%, despite
10.86% less L2 traffic and unchanged tensor work. Full numerics/1.033B bit-exact
pairs,304 modules, all140 guards and1,331 seals pass;133 warnings remain.
All7,680 samples, including a native33.14% tail shift, are retained. Last guard
18:23:15UTC; no worker, backend rebuild or production change. The CPU failures
and separately reviewed repairs stay. No integration, selector or retiming.

The [unchanged-kernel control observation](results/2026-10-16-control-lowering.md)
now identifies lost active lanes: thread instructions fall23.08% but warp
instructions rise18.01%; active threads per warp instruction31.997→20.857.
Local traffic is zero in both arms with a positive control. All old instruction
anchors/outputs/plans remain exact.41 guards/40 warnings,2,050 counter passes,
zero updates/actions/timing samples; last GPU18:55:17UTC, no worker. CPU proofs
exclude executed dW fallback divisions for the current workload and show existing
dX remainder helpers have divisor1. Neither proves driver elimination/inlining.
Retain CPU classification/newline failures and repairs. Direct divergent branches
are not measured; local-ID ordering does not establish subgroup membership.

The [shared-address CPU proof](results/2026-10-16-convolution-carry.md) now passes:
six derivative contracts share common-base/carry arithmetic;17 forward/packed
occurrences stay exact.532 modules validate; actual emitted expressions match
184.591M address slots,5.244M cursor transitions,20 small cases and99,585 boundary
carries. Exact structural scope preserves masks, scalar loads, B-side code,
types/K order/compensation/layouts/barriers. No new alignment branch or helper.
Static dX divisions/calls and dW branches decrease, but select/min/max work grows;
this is not a native-instruction or speed result. CPU-only, no retry/worker.

The [complete shared-address trial](results/2026-10-16-convolution-carry-family.md)
now rejects integration:18.954→18.349ms GPU,3.19%; queued wall3.26%, both below5%.
Calibration+.81%/+.85% passes in aggregate. Four dW gains cannot justify a subset:
two dX regress. Warp work rises1.60%, ALU3.03%; active lanes stay31.997 and local
traffic zero. Small elapsed improvement does not make static sites or instruction
counts a latency model. All numerical/full-output/plan/pipeline/binding gates
pass;140 guards/46 warnings,2,000 counter passes, zero updates/actions. Keep all
7,680 samples, especially the dX control's.445→.739ms unexplained tail shift.
GPU ends19:49:53UTC; no worker, retime or full-update trial. Keep58aff452/1.70x.

The [geometry-metadata CPU audit](results/2026-10-16-convolution-metadata.md)
passes: eight maps/1.04MiB cover all23 current products,30.428M explicit address
checks plus bounded affine batch extension. Forward/dW share spatial maps;
dX uses the inverse. All272,000 entries, original K/layout/staging/splits and
38 source seals reconcile. No actual shader or speed result. Its7.314B logical
lookups/29.255GB before masks are not measured traffic and cannot be ignored.
Required one-time GPU initialization is eight dispatches/1,064 groups; existing
host-visible constants do not provide the device-local immutable lifetime.
CPU review20:04:56UTC, no worker/retry, zero GPU calls/builds/actions/updates.

The [complete spatial-metadata trial](results/2026-10-16-convolution-metadata-kernels.md)
now rejects integration:19.112→36.871ms GPU,92.92% slower; queued wall92.65%
slower,19/20 contracts regress. Aggregate calibration+.0559%/+.0388% passes5%.
ALU instructions fall9.25% but total warp work rises16.26%, LSU28.08% and L2
traffic3.422x; tensor work stays exact and local traffic zero. DRAM is unchanged
within.012%; several packed cases regress despite less L2 traffic. No single
memory/stall/source-site explanation follows. Precomputing addresses alone does
not reproduce library feeding. Close this lookup implementation, without a
table/layout/geometry sweep, selector, retime or full-update integration.

All offline source proofs,2.690M independent F64 comparisons/1.033B exact GPU
pairs,2,000 counter passes/6,728 metrics and2,415 source seals pass.140 guards/
ten warnings, zero updates/actions.60 initialization/readback pairs verify2.126M
entries; complete setup wall totals376.987ms across isolated processes, not
shared full-world startup. All7,680 timings stay. GPU ends20:52:28UTC, independent
review20:53:01; no worker. Preserve all three CPU reader failures/repairs and the
unrepeated first pair. No backend rebuild or production/learning change.

The [host-target contract](results/2026-10-16-host-target-contract.md) now passes.
Both workloads construct22.1472MB of behavior inputs on CPU,16.6464MB of it dense
two-hot targets from65,280 scalar bytes. All18 behavior/160 world-target interfaces
reconcile. JAX constructs targets inside its compiled update, but also uses dense
one-hot source arrays: no unsupported compact-loss claim. GPU means/features/EMA
already exist. Seven CPU tests/36 source-F32 lambda cases/six negatives/33 seals
pass; preserve the initial stochastic-layout reader failure and exact repair.
Independent272-update trace reconstruction retains1.437ms target/report work and
1.012ms behavior-input preparation. The8.086ms readback span includes7.913ms GPU
execution; uncovered time is not pure CPU cost or additive savings.

The [resident implementation](results/2026-10-16-resident-targets.md) now exists
in isolation, preserving that complete scope and the existing losses. Offline
binding/prefix/lifetime checks, ten Vulkan modules and Clippy pass.12 component
generations/zero learner updates/actions pass the guard and structural checks,
but two occurrences of one coefficient fail the declared F64 envelope among
19.652M comparisons. Exact rational arithmetic verifies the failed reference.
The original Rust functions reproduce7.840M source-F32 values exactly and also
fail this envelope at four other occurrences. CPU equality alone would not fix
the gate; an explicit-FMA affine rewrite fails78 occurrences on CPU and is rejected
without a GPU attempt. No full-model timing or promotion follows. Preserve all
failures and the original cohort/gates; no worker remains.

**Next:** resolve the scalar-return versus encoding precision contract on those
retained inputs before selecting a repair and separately declaring GPU
qualification/full RGB-CDP comparison. Keep pre-update ordering, masks/discount,
normalizer checkpoints, diagnostics, CDP bonuses and production replay refresh.
No CPU normalizer workaround, compiler feature, favorable subset or numerical
gate widening follows. Keep58aff452/1.70x; target work is a bounded portion,
not an explanation of the entire gap.
Keep caller/pass/kernel scopes distinct; no
additive gap budget from incompatible boundaries. No padding/tile/precision/shape-
constant sweep, favorable orientation selector, rejected convolution scheme,
prior timing, compiler feature or automatic promotion. Keep58aff452/1.70x.
Recurrent matrices and host/graph execution remain open. Keep prior
batching isolated: its remaining posterior-adjoint/materialization and prior-to-
deterministic-state gradient boundaries are specific numerical follow-ups, not
an automatic allocation or reason to relax the original gates. No candidate timing.
Keep causal observation/imagination sequential,
masks/free nats/stop-gradients and staged late targets intact. Actual raw-world
gradient/boundary qualification for prior batching still precedes its matched comparison.
The [upstream workgroup review](results/2026-10-16-workgroup-contract.md) is complete:
all seven optional features are advertised, Blade enables two, and Naga lacks
tensor-addressing IR. One properties-only guard/no warnings, no logical device
or dispatch. The rejected October15 family already used tensor addressing and
workgroup matrices; new upstream exposure does not justify reviving that route
or a compiler project. No silent repin.
Matched imagination arithmetic is the
current narrow execution region to attribute, not a new broad profile or sweep.
Incomparable instrumented/ordinary regions cannot form a savings budget.
Convolution feeding remains open: current dX/dW still execute8.63x/8.17x
full-JAX warp instructions. Do not revive the rejected packet orientation,
start a layout/geometry sweep or infer parity from source arithmetic. Keep
numerical gates and834241b6; no automatic GPU allocation or compiler project.
The [complete compact-channel/reuse policy](results/2026-10-15-reuse-policy.md)
still loses21.72% and stays rejected. Its source-work reduction was not measured
hardware improvement. Host targets and non-convolution execution remain open;
no whole-gap/parity claim or production/Atari promotion follows these captures.

**Practical CDP finding:** the [existing-policy recheck](results/2026-10-15-cdp-precision-recheck.md)
reduces full B32 CDP78.206→50.122ms,35.91%, on that same binary. Historical Mixed
selection misses later BF16 derivatives and staged posterior reuse; imagination
is unchanged. Four guards and exact initials/finite trajectories pass, but this
is cost evidence, not a CDP/JAX ratio. Its [scoped qualification](results/2026-10-15-cdp-bf16-qualification.md)
now passes5.59M independent projection/loss values and exact whole/staged actual
world gradients. This is not a full-model independent oracle or learning parity.
No timing repeat or learning restart; production remains unchanged.

**Current main-path attribution:** [exact bonus substitution](results/2026-10-15-cdp-imagination-work.md)
changes fixed imagination20.771→8.705ms. Its12.066ms conditional cost belongs to
the additional action-effects ensemble, absent from RGB/JAX; this is not an
ordinary update gain or pure kernel latency. The actual12 projections add64.42G
nominal MACs. A concrete compiler-composition miss also leaves four repeat/
transpose/normalization chains writing1.132GB of logical intermediates. Existing
row-addressing support runs before typed normalization fusion exposes the
pattern, then rejects its precision marker. The [general composition fix](results/2026-10-15-repeated-row-composition.md)
now reduces complete CDP50.151→47.728ms,4.83%, with exact272 reports and346
initial/first/final tensors. RGB54.129→54.245ms is effectively unchanged.
Eight copies disappear; groups, typed arithmetic and rounding stay. Conditioned
F64/exact-pair checks pass; the original signed near-zero scalar failure remains
in both compilers with identical values and unchanged tolerances. No exploration
change, production promotion, row/geometry sweep or learning evidence.

**Explicit matrix lowering is now rejected by measurement:** [feasibility screen](results/2026-10-15-convolution-matrix-feasibility.md).
All23 products' input-patch construction alone costs34.348ms, above the declared
15ms gate, even with free NHWC preparation and no GEMM/output/reduction cost.
All6.647M exact small-case values and47,288 actual samples pass; two passing
guards, no new warnings or learner updates. No complete lowering or chunk sweep.
Logical18.119GB writes are not DRAM traffic or a universal latency lower bound.

**Fixed-geometry spatial caching also loses:** [complete-family result](results/2026-10-15-convolution-halo.md).
All15 forward/dX occurrences cost13.706→18.598ms,35.69% slower with original
packing/output work.503,808 F64 outputs and257,949,696 B512 paired outputs pass;
math, K order,32/64 tiles, staging/scatter and all eight dW products stay.
The largest shared allocation47,104B fits, but logical image requests falling
9.461→1.642GB does not establish reduced DRAM traffic or a useful pipeline.
Additional shared/address/preload/resource costs are not individually measured.
Six passing guards/two warnings, no actions/updates; no full-update trial or
favorable-subset promotion. Original B1 kernel-selection coverage failure and
CPU receipt errors remain; a separately declared exact-schedule repair passes.
Two isolated kernel series shift within the128 samples, all retained. A CPU
audit finds complete RGB first/last32 means54.260/54.198ms; no comparable drift
explains the whole2.04x gap. Do not infer a driver/clock cause from timing alone.

**Current complete-execution attribution now passes:** [result](results/2026-10-15-whole-update-attribution.md).
Fresh ordinary54.113ms; coarse55.150ms,+1.916% instrumentation. GPU-pass union
47.699ms versus7.453ms uncovered. Cached programs46.484ms dominate; categorical
means.059ms and optimizer.464ms cannot explain the gap. All272 reports/292
initial-first-final tensors and paired plans/pipelines/groups match exactly.
Two guards/no warnings,544 synthetic updates, zero actions. Timed chunking
adds three intermediate waits; pass coverage is not SM utilization or an exact
ordinary decomposition. The corrected two-prefix/three-backward description
and original CPU failures remain. Current world graph matches the retained
valid-value convolution attribution, not an additive savings estimate.

**Current B32 graph-execution control passes:** [result](results/2026-10-15-jax-command-buffer-b32.md).
JAX command buffers on/off: core26.550/29.042ms, inclusive27.223/29.505ms.
Exact272 reports/288 initial-final tensors,940 static fusion computations,
112 library configurations/physical shapes and reachable heavy work match.
Native remains1.83x slower even without JAX graphs. Their2.282ms inclusive
effect cannot explain most of the residual; old tiny-chain/B8 ratios do not
scale into a dominant B32 advantage. Historical26.632ms was JAX's core boundary,
not caller-inclusive; this correction is not a native speedup. Two guards/two
warnings,544 synthetic updates, zero actions. CPU serialization failure and
qualified dependency-graph/JSON canonicalization repair remain; no GPU retry.

**The connected image cost is now isolated with valid boundary values:**
[all-output](results/2026-10-15-spatial-value-cost.md) and
[closed-region](results/2026-10-15-spatial-boundary-cost.md) interventions pass.
The former keeps3.154GB of logical intermediate outputs and shows only.530ms
from replacing non-heavy image arithmetic. The closed cut preserves every
escaping value but omits internal writes: non-heavy substitution saves4.173ms;
combined image/convolution substitution reduces fixed-world34.195→6.564ms.
Only46 outputs/7.014MB escape the combined region. This is not measured DRAM
traffic or an attainable optimization; changed copies/dispatch work also matter.
Independent runtime version proofs, poison checks and exact full outputs/tensors
pass. Interaction−3.552ms rules out additive attribution. Three guards/no warnings
across both studies, two initializer updates/1460 fixed replays, zero actions.
Controller-only provenance and CPU/lint failures remain; no GPU rerun. Production
and complete RGB/JAX54.113/27.223ms remain unchanged.

**The [image-interface audit](results/2026-10-15-image-layout-contract.md) is complete.**
All23 JAX convolution outputs use BF16 NHWC/KRSC; native returns NCHW and F32
derivatives, then converts/repacks. Eight identical-version packs duplicate
83.886MB of logical writes. This is a compiler dataflow/layout issue, not proof
that layout alone explains the library's feeding advantage. A concrete packing
launch cliff affects four products; repairing it saves only2.13% of complete
RGB updates,54.117→52.963ms, below the declared3% gate. CDP saves3.08%. Structured
F64 checks and exact retained-control trajectories pass, but full timing-program
exports were disabled, leaving that gate incomplete. Keep the repair isolated;
no retime/promotion or claim that the gap has meaningfully closed. Six guards/
four warnings,1088 synthetic updates/zero actions; original CPU observer failure
and separate repair remain. Control076f3135 and inclusive1.99x remain current.

**Connected producer/consumer views now qualify:** [complete result](results/2026-10-15-image-views.md).
RGB54.205→51.945ms,4.17%; CDP47.699→47.112ms,1.23%. All272 reports/targets and
292/346 tensors at initial/first/final boundaries match exactly, including
qualification prefixes and retained controls. Full-program review independently
reconstructs13 changed images/31 removed dispatches in RGB and5/11 in CDP;
convolution selection/arithmetic and non-world programs remain unchanged.
CPU proof covers253.755M addresses;144 existing/432 new shader variants pass.
876.610MB fewer logical writes is not measured DRAM traffic or a VRAM saving.
Eight guards/seven warnings,1184 synthetic updates, zero actions; timing ends
12:30:39UTC and CPU review12:31:48, no worker. Keep isolated67559ce5; no production
or learning promotion. The original Clippy failure/source remain.

This removes2.26ms, not the remaining24.72ms over inclusive JAX. Next distinguish
the native interface's expressible data-feeding pipeline from the generated
algorithms' efficiency, using the complete forward/dX/dW contracts and measured
resource limits. A credible parity route requires that combined pipeline, not
another isolated tile. Convolution feeding and non-image execution remain separate
targets. No rejected family, full im2col, halo, invalid-mask, geometry/precision/
flag sweep, new GPU allocation or Naga project follows automatically.

The [CPU feeding-pipeline check](results/2026-10-15-feeding-pipeline.md) now
separates an actual compiler restriction from Vulkan capability. Nine typed
subgroup-matrix schedules validate offline; two operand slots plus scratch fit
all reduced-precision layouts under48KiB. Packed F32 variants exceed it and
remain unsupported by this schedule. Two loading/four computing subgroups with
ordinary full-group barriers pass1.646M schedule-event checks/five negatives.
Naga rejects subgroup-uniform consumer branches for all three types, while
whole-group controls pass; the newer memory-model branch has the same analyzer.
That CPU result is not performance evidence and changes no compiler. Preserve
three CPU preparation failures and repairs. Its [actual complete-family follow-up](results/2026-10-15-feeding-family.md)
now **rejects the schedule**: native/GLSL serial/pipeline GPU totals22.412/24.418/
46.671ms, all23 products included. Aggregate serial calibration+8.95% passes;
every distinct contract regresses against both controls. All4.036M F64 comparisons,
1.033B paired values and120 guards pass, but there is no useful speedup or reason
to start the Naga feature project. Preserve the stopped original allocation's
subgroup-ID/local-ID assumption and the separately qualified repair; enforcing
subgroup size alone does not establish that mapping. No full-update trial,
favorable selector or loader/geometry sweep follows.

The [output-contract CPU audit](results/2026-10-15-output-contract.md) now passes
all23 actual contracts/280.986M outputs. Nine typed direct-store SPIR-V variants
need zero explicit output scratch; nine existing-Naga fragment-store IR variants
use4KiB independent of full tile size. The latter needs no new compiler builtin,
but adds two barriers per phase.1.442M independent bit-rounding checks pass;
neither route has GPU numerical/timing qualification. Preserve the original
missing-rustc preparation failure and reviewed absolute-path repair.

Removing the full output tile is necessary for some wider reuse schemes, not
sufficient for a good pipeline. Applying current packed-kernel padding, two slots fit
3/9/13 of22 known library geometries under48KiB with full/fragment/direct output.
Unpadded payload with fragment/direct fits18/22 or22/22, but that is only a storage
lower bound, not a qualified bank-conflict-free mapping. Four larger dW tiles
also represent384 logical F32 values/compute lane with retained compensation;
not measured registers/spills. Its earlier removal saves only.89% and fails
numerical checks: keep it. No accumulation sweep or unsupported cuDNN claim.

**The resulting fixed policy now fails its complete-family timing gate:**
[design](experiments/2026-10-15-reuse-policy.md) and
[result](results/2026-10-15-reuse-policy.md). It includes all23 products,
not just already-packed ones: four-channel packing, resource-bounded wider reuse,
one input slot and4KiB output fragments. K32, split ranges and compensation stay;
forward/dX terms may reorder and retain the original independent numerical gates.
120 offline modules and104.808M CPU addresses pass; source-state bounds are64
spatial/96 compensated values per computing lane, not measured registers.
Padded work increases1.45%,374.467G versus369.099G. Native/GLSL/candidate totals
22.327/24.313/27.177ms include packing/reduction; aggregate calibration passes,
but18/20 contracts regress.120 guards/no warnings and independent final audits
pass; numerical15/20 contracts are exact, five pass unchanged nonexact limits.
All samples/transients and CPU observer repairs remain. Timing finishes16:10:46UTC.
No compiler project, integration, selector, sweep or learning promotion. Keep
67559ce5 and the1.91x whole gap. Native counter evidence above now guides the next
implementation contract; fitting resources does not establish an efficient kernel.

### What is established

- **We are not doing four times as much model arithmetic.** The
  [reachable executable audit](results/2026-10-14-executable-work.md) finds only
  1.40% more nominal heavy MACs. Shapes/precision/rounding and scalar work still
  matter; MAC counts are not instruction counts or utilization.
- **The convolution implementation differs fundamentally from the library.**
  [Actual full-JAX counters](results/2026-10-15-library-convolution-feeding.md)
  show10.33×/8.27× fewer warp instructions than same-algorithm CUDA translations
  of our dX/dW, despite31.6%/29.2% more tensor instructions. Library SASS uses
  asynchronous128-bit global-to-shared copies and multistage block reuse.
  CUDA/native timing bands fail, so these are not native counters or an8×
  speed promise. The [native iterator improvement](results/2026-10-14-convolution-iterators.md)
  demonstrates part of the supporting-work cost, not complete library parity.
- **Whole-program execution also mattered.** [Staged posterior reuse](results/2026-10-13-staged-world.md)
  removes duplicate computation; [command reuse](results/2026-10-14-command-cache.md)
  reduces complete-update time10.16%; GPU [RNG](results/2026-10-14-device-rng.md)
  and [EMA](results/2026-10-14-device-ema.md) remove measured host boundaries.
  These are separate experiments, not additive savings. Mixed CDP cannot
  automatically reuse a different-precision posterior.
- **Real dependencies remain expensive.** [Execution controls](results/2026-10-13-execution-boundaries.md)
  measure~2.05µs for tiny dependent Vulkan/ordinary-CUDA kernels versus~.65µs
  in CUDA graphs. [Precise barriers/events](results/2026-10-14-dependency-paths.md)
  do not supply the missing multiplier. Never remove actual dependencies.
  Many fewer dispatches or groups have repeatedly yielded only~1% whole gains;
  counts alone do not justify another fusion sweep.
- **Precision alone does not explain the residual.** [Wider JAX image intermediates](results/2026-10-14-image-precision-control.md)
  add only.92% to complete updates; the earlier wider matrix/intermediate
  control adds6.19%. These change some compiler/library choices, not storage
  alone. Native connected [pooling tape](results/2026-10-14-spatial-tape.md) and
  [image training](results/2026-10-14-image-training-region.md) help, but their
  individual4.02%/3.0015% complete gains do not close the gap.

The [retained full B32 profile](results/2026-10-14-jax-image-profile.md), before
the latest image/order improvements, measures68.637ms complete,56.948ms GPU-pass
union and11.688ms outside, including3.752ms CPU distribution decoding and1.632ms
target construction. JAX has23.099ms kernel/copy union inside27.617ms captured
intervals. Native interacting family allocations and JAX kernel sums have
different coverage; do not subtract them into a promised savings budget or
call them SM utilization. Host work cannot be the sole explanation.

### Rejected route: do not integrate the workgroup prototype

The [fixed-address/K32 derivative](results/2026-10-15-convolution-qualification.md)
qualifies against broader BF16 inputs:196,608 F64 outputs, worst relativeL2
1.412e−6, all pointwise gates. Fresh native/proof2.14890→1.29280ms is39.84%
less GPU time with all transforms/scratch. But it occurs once in the full
graph: its.856ms saving is only1.43% of the retained update before integration
effects. This does not justify a compiler project.

The subsequent [shared spatial-family check](results/2026-10-15-convolution-family.md)
covers all13 distinct forward/dX contracts and15 actual occurrences with one
fixed typed implementation. It **regresses12.833→18.131ms,41.29%**, including
all packing and output conversion. All503,808 F64 values/416 probes and
257,949,696 exact output pairs pass;27 guards/five warnings, zero learner
updates/actions. Keep all per-shape regressions; no cherry-picked selector,
geometry sweep, Naga/Blade integration or complete-update trial follows.
The native dX samples themselves transition~6.13→2.15ms; retain every sample,
do not relabel the mean steady state or assign a hardware cause without evidence.

This narrows the conclusion: larger workgroups, high-level tensor addressing
and a tile copied from one library operation do not reproduce the library's
feeding pipeline across the model. The three-input-channel cases waste padded
contractions, but aligned cases also regress. Shared typed arithmetic is still
the right design; one fixed schedule is not automatically a good implementation.
Earlier store, alignment, BF16/F16 and compiler-statistics diagnostics remain
closed. The local-memory statistic fails a scalar control and is not spill data.
Native executable queries returned no representation; no unsafe profiler retry.

### Latest result: remove bulk CPU distribution decoding

[GPU categorical means](results/2026-10-15-categorical-gpu.md) reduces complete
B32 updates59.963→54.224ms,9.57%, with heavy graphs unchanged. Three readbacks
shrink25.1MB→98KB; imagination/targets17.814→11.945ms. Native Blade computes
detached means with mirrored bins/stable exponential differences. All28,860
F64 checks and873 exact symmetric zeros pass, including actual trained logits.
This deliberately replaces F64 CPU arithmetic with F32: first targets/tensors
are exact, later trajectories differ. No learning-equivalence claim.

Four guards/three warnings,545 synthetic updates, no game actions/workers.
The failed raw-JSON plan assertion is retained; exact semantic maps, ordered
constants and pipeline checks pass without a GPU repeat. Target building,
percentile state and other host dataflow remain unchanged, not a fully
GPU-resident learner. Keep the candidate isolated; do not restart Atari.

### Whole-family audit: a concrete missing transfer contract

World training still takes30.20ms, longer than JAX's complete26.63ms update.
The [completed CPU audit](results/2026-10-15-convolution-contract.md) reconciles
all23 products, actual parameter/gradient paths and library kernels over eight
captured updates. Library forward/dX and dW use different reuse/staging schedules.
Native14 packed occurrences use8-byte BF16 vector expressions. The failed
spatial family adds47.06% padded tile work across its15 occurrences; aligned
cases also regress with identical tile work, so padding is not the whole cause.
Current native23-product tile work is369.099G versus226.492G nominal; neither
is a native/JAX instruction ratio. Preserve the corrected first audit.

Six CPU cases identify an actual compiler-interface limitation: Naga discards
the requested cooperative matrix scalar and derives it from the pointer.
Thus F32/F16/BF16 matrix loads from packed `vec4<u32>` become integer matrices,
although all three equivalent Vulkan SPIR-V cases validate. No GPU execution
or Naga edit occurs. Source vector width does not prove hardware instruction
width or a speedup; this contract is necessary for that representation, not
proven to explain the entire gap.

The [transfer observation](results/2026-10-15-convolution-transfer.md) now
measures all14 packed occurrences/11 contracts:4.933→4.170ms,15.48% less complete-
operation GPU time. The generic control calibrates+3.66% GPU/+4.00% queued wall.
But the.763ms saving is only1.41% of the retained full update as a scale comparison,
not measured whole improvement. Keep this local benefit isolated; it is not
the missing multiplier or a reason for another compiler project.

Qualification remains failed:40 forward and1,277 dX cancellation values exceed
original absolute envelopes in native and both variants, with bit-identical
outputs. Exact integer contractions confirm every failed reference. Both stopped
allocations remain; the separate cost-only observation repeats no numerical
cohort.132 guards/six warnings,18.819M F64 comparisons,5,115 optimizer-free calls,
zero actions/learner updates/workers. No gate relaxation or promotion. The CPU
aggregate membership error and corrected audit stay alongside all original data.

### Resource contract confirmed; fixed library control rejected

A separate guarded physical query confirms **48KiB explicit Vulkan workgroup
shared memory**, versus76/80KiB in the captured library dX/dW. In fact19/23 main
library convolutions exceed48KiB. One guard, no warnings, no logical device or
dispatch. The exact library allocation cannot be copied into explicit Workgroup
variables. This does not establish the slowdown caused by capacity alone or
rule out a different legal schedule/driver-managed cooperative-matrix path.

The [fixed library control](results/2026-10-15-library-pipeline.md) reconstructs
original cnn1 engines40/33, shapes/strides/knobs/workspaces. Both stage-choice1
configurations fail3000 at engine-configuration finalization, despite advertised
individual ranges0–7. No candidate execution plan, kernel or timing exists.
Two memory-metadata queries also fail; all four failed guards, their truncated
failure-log snapshots and separately reviewed follow-ups remain. Zero new
warnings, convolutions or learner updates. Local serialized GPU profiles confirm
CUDA99KiB opt-in shared memory versus Vulkan48KiB, not the latency it causes.
No stage/engine/tile sweep, silent library upgrade, invented enum or CUDA learner.
Installed9.24 lacks the newer documented memory-limit attribute. The serialized
frontend=false flag does not imply legacy execution. Preserve the earlier
resource audit's compact/raw-schema failure and corrected review too.

### Exact-value attribution: convolution is still a large cost

The [saved-output comparison](results/2026-10-15-convolution-value-bypass.md)
passes: fixed-world GPU34.191→14.282ms,58.23%, after replacing all23 products
with copies of their exact recorded outputs. Original allocations, consumers,
4,023 descriptors/1,258 groups and dependency boundaries remain. All live/output
fingerprints, three292-tensor boundary pairs, first targets/reports and full
plan/pipeline semantics pass. This avoids the invalid data in earlier zero-work
masks. Two guards/zero new warnings; the original binding smoke failure and CPU
whole-object audit OOM remain, with separate binding/streaming corrections.
No GPU timing repeat, relaxed gate or raised CPU memory limit.

This is not a58% full-learner improvement: it is optimizer-free fixed replay,
with no actual convolution computation. Copy-only2.005ms includes all original
barriers but omits nonselected commands; it is not a subtractable kernel cost.
The14.282ms remainder and other learner stages remain real. Latest complete
54.224/26.632ms stays unchanged; no native SM-utilization claim.

The [encoder/decoder comparison](results/2026-10-15-convolution-role-cost.md)
now passes: normal/encoder-substituted/decoder-substituted/both fixed-world
intervals34.216/27.285/21.142/14.280ms. The symmetric conditional allocation is
6.896ms encoder/13.039ms decoder, with only.069ms interaction. About65% of the
convolution intervention belongs to RGB reconstruction, which CDP removes.
All four conditions preserve exact live/full outputs and three292-tensor pairs;
first targets/reports, complete graph semantics and1,157 source seals pass.
One guard/zero warnings,512 retained samples/647 full-world replays and one
initializer update. No new failed process or timing retry. This is not pure
kernel cost or a whole-learner/CDP speedup; the14.280ms remainder stays real.

Source review confirms CDP already has the October11 exploration projection
reuse/hidden centering; its old18-way first-projection diagnosis is obsolete.
Mixed's separate BF16 inference/F16-world path does not inherit staged RGB
posterior reuse. The policy was selected before BF16 backward and staging
support existed. The [bounded recheck](results/2026-10-15-cdp-precision-recheck.md)
now finds Mixed/BF16 full updates78.206/50.122ms,35.91% less time. Posterior+world
49.788→21.715ms accounts for the change; imagination stays23.94ms. Four guards,
one warning,592 synthetic updates and exact346 initial tensors pass. Each arm
reproduces its own smoke prefix; cross-policy trajectories differ. Actual model
parameter relativeL2 is.0001757 after one step/.05677 after272, distinct from
optimizer moments. The corrected CPU label/audit and all data remain. No source
rebuild or game actions. This is cost evidence, not independent numerical
qualification or a comparison to the RGB/JAX ratio. The subsequent
[BF16 boundary/staging checks](results/2026-10-15-cdp-bf16-qualification.md)
pass: eight independent cases/5,585,254 values and two actual whole/staged pairs,
86 raw gradient tensors/14,688,256 elements per pass. Late-input NaN poisoning,
changed targets, exact restoration, three346-tensor pairs and1,200 source seals
pass. The original JSON observer assertion remains failed: its5,618,720 F32
values were exact, but in-memory and parsed F64 representations differed.
Separate persistence/parsing repair and CPU regression pass without relaxing
equality or repeating the independent boundary. Two passing/one failed guard,
one warning,13 raw backwards/two initializers including failure; zero actions.
No new timing, full-model independent oracle, learning parity or promotion.
The [completed main-path attribution](results/2026-10-15-cdp-imagination-work.md)
now identifies12.066ms of conditional imagination cost in the additional
ensemble, with all outputs preserved. Its pass-order/view-composition miss above
is the next concrete compiler test. Continue the separate structural feeding/
execution investigation; do not resurrect the old18-way state-projection diagnosis.
Do not make RGB decoder work the main-path bottleneck by assumption. Further
algorithm work must address substantial data-feeding or execution structure,
not another small tile/iterator variation. No new subset/geometry sweep,
production promotion or learning restart follows.

The [offline capability check](results/2026-10-15-categorical-gpu.md#what-remains-fundamental)
rules out a misleading shortcut: SPIR-V explicit async-copy/wait/event needs
Kernel, not Vulkan Shader. The untyped variant still needs the event contract.
Matrix2's compiler-managed staging is not explicit control of the library's
four-stage async schedule. This is not proof the driver cannot emit asynchronous
copies, nor evidence that async copies alone close the gap. Do not start Naga
feature work merely because an instruction exists in the combined SPIR-V spec.

Remaining scalar targets, non-heavy device work and dependent execution remain
separate costs. The measured5.739ms host-boundary saving does not explain the
remaining27.592ms complete difference, and component allocations are not an
additive savings budget. The original root-cause/performance goal stays open.
No more broad profile, tile/row/dtype/hint sweep, CUDA learner, unsafe barrier
weakening or automatic feature project. Latest upstream99e3de27/e7c3f7b is
unchanged; check again before diagnosing a new backend issue. The reserved-WGSL-
keyword probe failure and original48-channel padding audit remain. Keep failed
sources/gates/tails and historical pins; no recursive repinning.

Historical experiment-by-experiment decisions remain in their linked reports
and git history. The following sections retain earlier production integration,
starting controls and architectural work packages; they are not an active queue.

## Retained qualified implementation boundaries

**[Typed cooperative matmul consolidation qualifies](results/2026-10-13-typed-matmul.md).**
One checked F32/F16/BF16 generator, explicit storage/multiply/output types and
shared typed partition selection/search replace the BF16-only dense path.
Specialized fused/hardware-specific algorithms and legacy search exclusions
remain. RGB52.803→53.047ms/CDP124.997→126.182ms are0.46%/0.95% slower in the
single matched pair, within the declared2% gate; no speedup claim. All272
reports, targets and292/346 tensors match exactly in these fixtures. Thirteen
guards pass; two warnings and corrected development failures remain. Pin056562b,
unchanged Blade/Naga, cache30/memo3. Both locks/native checks, Clippy and193
binding tests pass. Native F32 device execution on5080 and game quality are
not established. Next resume measured matrix/grouped efficiency, derivative
precision and command-stream work, not another dtype-specific path or Atari
campaign. CI stays on PR31.

**First post-diagnosis optimization qualifies: [BF16 dense split-K](results/2026-10-12-bf16-split-k.md).**
Complete RGB12M/B8 updates56.269→53.065ms (5.69% less time), CDP12M/B32
126.880→125.432ms (1.14%). Independent859,216 F64 values, long gradients,
ten passing guards and finite272-update fixtures pass; two corrected development
failures remain. Historical pin9e05731 (runtime identical to timed87a3236), with
Blade/Naga unchanged. Different summation order changes stochastic trajectories,
especially pretrained CDP; retained game quality is untested. Its54.21ms
imagination stage is unchanged. Next address remaining dense/grouped efficiency
and reduced-precision derivatives; no tile sweep, repeated profiler campaign
or Atari restart. PR31 tracks installation and CI.

**Historical operation-group accounting: [full JAX gap](results/2026-10-12-jax-gap-accounting.md).**
Model size matches exactly; a same-executable input-transfer control explains
only.262ms of the~44ms gap. Host preparation/targets/EMA matter, but measured
GPU work dominates. Matched matrix cases show native grouped F32 loses even
to JAX F32, and native BF16 dense loses to BF16 library kernels. One matched
convolution confirms backward precision coverage and forward kernel efficiency
both need work. Prioritize dense launch geometry/split-K/coalescing and grouped
precision alongside convolution derivatives, then posterior reuse and GPU
targets/EMA. Current counterfactual graph costs are17.32ms convolution,14.04ms
matrix,6.84ms remaining operations and5.89ms zero-work command stream. Full
execution44.088ms matches44.101ms of complete-update graph passes; explicit
host/transfer/optimizer and profiling-boundary accounting reconciles the44.04ms
ordinary gap. These are operation-group allocations, not pure kernel timings or
independent speedup promises. Single-dispatch timestamp sums are rejected; paired
blocks retain0.434ms/0.98% excess. Include fusion/recording/safe barrier scheduling
in the implementation work. Keep four production world submissions and learning
stopped; continue useful whole-update optimizations, not another profile sweep.

The [combined result](results/2026-10-12-backend-integration.md) reduces ordinary
CDP12M/B32 updates **161.32→135.17ms (16.21%)** and RGB12M/B8
**72.69→61.15ms (15.88%)**. Native RGB is still2.52x/4.90x slower than the
fresh JAX FP32/BF16 controls. Useful changes from all three work packages are
integrated; no claim of reaching JAX, full-BF16 CDP world training, or faster
learning to a score. The final CPU/GPU audit and all runtime CI355 jobs pass.
This bounded round is complete; the wider performance target remains open.

**Current decision: [adopt the qualified split-convolution tile fix](results/2026-10-12-split-convolution-tiles.md).**
On refreshed main20af998, complete RGB updates61.027->56.058ms and
CDP135.070->126.739ms save8.14%/6.17%, with exact272 numerical reports/tensors
and ten full-plan comparisons. All17 guards pass; four allocation warnings
and extra96 short-export updates remain. Integrated5a493322 is pinned, with
Blade663afd2/Naga c9fd6a85a unchanged. Both locks, native checks, Clippy and193
installed-binding tests pass; CI status stays on the PR dashboard.
Standalone [PR255](https://github.com/kvark/meganeura/pull/255) reuses the existing
occupancy rule and leaves low-occupancy cases and split counts unchanged.
No new shader or precision change; the full JAX gap and learning quality stay open.

The preceding [Nsight investigation](results/2026-10-12-nsight-root-cause.md)
identified a causal kernel bottleneck.
Counter access is enabled. RGB weight gradients cost12.62ms versus JAX1.44ms;
matrix products15.03 versus2.37ms in diagnostic captures. Tiny scalar tiles
retain poor register reuse after split-K supplies ample parallelism. One
isolated16->32 tile check reduces ordinary RGB60.98->55.97ms and CDP135.13->126.61ms,
with exact272 reports/tensors. That was the diagnostic precursor to the
qualified change above. Graphics
teardown crashes remain failed guards; Systems timestamps remain rejected.
Profiled counters are not ordinary utilization or complete explanation of the
JAX gap. No permission blocker, GPU wedge observed, or new learning allocation.

Next: efficient matrix/grouped-RSSM kernels, precision and layout/fusion,
alongside reduced-precision convolution derivatives, followed
by posterior reuse and device-resident inputs/targets/EMA. Use complete-update
timing and numerical checks; do not revive PR248 from its isolated kernel result
or launch a tile/batch sweep. Upstream now supports measuring per-step input
refresh, which matters when comparing resident and repeatedly uploaded inputs.
CDP imagination now costs54.22ms versus world training45.94ms; optimize total
update cost, not only the world model.

## Starting point

The [measured baseline](results/2026-10-11-cdp-performance.md) uses Meganeura
`d7a7293`, Blade `56f0565`, 12M/B8/T16/H15 for matched-shape RGB:

| Stack | Ordinary full update |
| --- | ---: |
| Optimized native, F16 matrix-input permission | 71.37 ms |
| JAX FP32, highest matmul precision | 24.36 ms |
| JAX BF16 | 12.40 ms |

These are framework measurements, not identical graphs or numerical parity.
Native CDP B32/F16 takes 162.00 ms per 512 replay positions. The previous
2.66x throughput gain is real, but faster learning to a score is unestablished.
F16 permission alone saved about 1–2%; it does not change activation storage.

The user stopped the new Freeway screen before this work: B8/Auto seed1009
ended cleanly at 39,568 actions / 9,831 updates, with a partial checkpoint.
No learner finished and no other arm or frozen cohort started. Preserve the
declaration and artifacts; do not resume a checkpoint as the same lifetime.

## Earlier production integration state

[First-round evidence](results/2026-10-11-backend-training-performance.md):

- Convolution: [Meganeura PR248](https://github.com/kvark/meganeura/pull/248)
  passes kernel checks and speeds isolated gradients, but not full updates.
  CDP161.40→161.27 ms; RGB71.38→72.76 ms. Draft, not adopted.
  [Complete-trace repeat](results/2026-10-12-convolution-attribution.md) confirms
  this: CDP161.31→161.48, RGB71.11→72.82 ms; world GPU intervals52.93→52.81 ms.
  Selection passes; the isolated gain is not hidden by CPU overhead. The
  kernel-level cause remains unresolved; do not promote or sweep this candidate.
  A separate [hardware-row alignment fix](results/2026-10-12-convolution-subtiles.md),
  [PR252](https://github.com/kvark/meganeura/pull/252), now qualifies a useful
  CDP161.51→158.12 ms/RGB71.58→70.06 ms improvement. No new shader; forward/input
  oracles pass. Its broader unchanged scalar weight-oracle failure stays explicit.
- Layout: [Meganeura PR249](https://github.com/kvark/meganeura/pull/249)
  removes repeated action copies. CDP161.40→157.30 ms, about1.05 GiB less
  imagination allocation, exact272 reports/final tensors. The isolated change
  is now pinned in the combined stack, still unmerged. Independent of PR248;
  RGB effectively unchanged in that isolated test.
- Mixed precision: [BF16 dense learner updates qualify](results/2026-10-12-bf16-learner.md).
  Explicit selection covers all five learner sessions; F32 masters/optimizer,
  convolution/grouped products and sensitive reductions remain. Pinned full
  CDP updates161.49->149.55 ms (7.40%); RGB71.33->66.47 ms (6.81%). Native
  partial-tile/gradient checks and272 finite complete updates per arm pass.
  World training regresses in this dense-only version. Auto default and live
  acting stay unchanged; the extensions below address useful storage coverage.
  Restoring Auto for unconverted operations does not fix CDP149.26→150.38 ms;
  RGB66.45→64.32 ms improves. Keep NativeF32 for the main path; no per-title
  precision branch or sweep. Native F32 cooperative shapes are absent on5080,
  but capability availability alone does not establish an end-to-end gain.
  [Convolution storage now qualifies](results/2026-10-12-bf16-convolution.md):
  shared native BF16 loop,16 F64 cases, six guards. Full CDP149.27→146.31ms and
  RGB66.51→61.02ms improve; only RGB world training improves. CDP world61.87ms
  remains slower. Keep F32 derivatives; do not claim lower overall allocations.
  The [fixed mixed-stage check](results/2026-10-12-mixed-world-precision.md)
  passes: CDP146.13→140.73ms, world61.78→56.53ms. Retain BF16 learner
  perception/imagination/behavior with the faster F16-input/F32-storage
  world-gradient path. RGB keeps its useful all-session BF16 candidate.
  This is an explicit precision choice, not a hidden per-title branch.
- Fresh JAX RGB12M/B8 control: FP32/highest24.35 ms, BF1612.42 ms. Native dense
  BF16 was still2.73x/5.35x slower, without identical graphs/weights/RNG. The
  combined comparison above supersedes that timing, not its retained evidence.
  No time-to-score claim. Upstream review branches remain unmerged.
- Profiling: correct the ordinary GPU fields to say **last submission only**.
  World training uses four submissions. Initial Perfetto captures also miss
  intermediate query pools. A timing-only harvesting repair now qualifies all
  four chunks with exact reports/tensors; its diagnostic waits add~2.7% wall
  time. Recorded world intervals52.80/57.86 ms confirm a GPU-side BF16 regression.
  Historical raw captures remain; these intervals are not SM utilization.

The [final integration check](experiments/2026-10-12-backend-integration.md)
combines BF16, row addressing and hardware-row alignment, not rejected PR248.
Meganeurad68f1a5 incorporates current maineb8c90f; Blade663afd2 incorporates
main7909fc1; Naga forkc9fd6a85a remains pinned with the user's permission.
Both dependency locks use git revisions, not path overrides. The four native
qualification processes and all six fresh native/JAX timing guards pass; three
allocation warnings remain. The independent CPU audit passes in4.767s.
Meganeura BF16/row PRs and Blade BF16 CI pass after rebasing without merge commits.
The installed Python binding supports `bf16` and `mixed`;193 runner tests pass.
Remove `bf16_dense` from the current CLI, retaining its historical source/evidence.
No game worker remains. Do not restart the interrupted Freeway comparison.

## Work packages

The first round below is retained for provenance. The current ordering above
comes from Nsight and the complete-update tile check; it supersedes this
original implementation sequence without changing its recorded outcomes.

### 1 Accelerate convolution training

- Add a cooperative-matrix weight-gradient path in Meganeura. Use existing
  FP16 operands / FP32 accumulation first, without waiting for BF16 support.
  Share existing dispatch, capability and split-reduction machinery; no
  im2col-sized global temporary, new backend or Kindle-specific shader.
- Cover the profiled encoder shapes first, then RGB decoder shapes, including
  batch reduction, padding, stride and partial tiles. Preserve scalar F32
  fallback and the explicit reduced-precision policy.
- Earlier per-dispatch costs: 13.48 ms of instrumented RGB weight gradients
  versus approximately 1.48 ms in the BF16 JAX trace; CDP B32's largest single
  dispatch is 12.29 ms (512 images, 32→48 channels, 32×32, 5×5).
- Check outputs against independently indexed F64 derivatives, including
  FP16-rounded operands to isolate accumulation/indexing from input rounding.
  Check strict-policy fallback and shader bounds. Benchmark the actual hot
  shapes and then ordinary complete CDP/RGB updates, not only a fast kernel.
- The weight-gradient candidate fails complete-update acceptance twice.
  The [next bounded check](experiments/2026-10-12-convolution-subtiles.md) changes
  only a conservative selection boundary:48-channel forward/input gradients
  align with16-row hardware stores despite not aligning with32-row workgroups.
  Qualify across image boundaries before timing; no new shader or shape sweep.

### 2 Implement useful mixed precision

- The bounded Vulkan check passes: driver580.178.04 exposes
  `VK_KHR_shader_bfloat16`, BF16 types and cooperative matrices on the5080
  (not BF16 dot products). Kindle's original Naga pin had no BF16 type
  support; the qualified compiler fork adds it, not just a Blade feature bit.
  BF16 weight loading converted to F32 is not native BF16 execution.
  Do not change/reload the driver or emulate a CPU learner to claim support.
- Add the smallest Blade capability/shader plumbing actually required, then
  Meganeura dtype, conversion, storage and arithmetic support. Keep FP32 master
  parameters, optimizer state, accumulators and sensitive reductions/losses.
- Propagate reduced-precision activations through supported forward/backward
  regions, converting at boundaries rather than at every operation. Measure
  memory traffic, scratch use and total update time. FP16 is a useful supported
  path even if native BF16 is unavailable through this driver.
- Use format-aware numerical checks, finite gradients and a short repeated
  synthetic-update stability check. Include small gradients, large values and
  cancellation; never silently demote strict F32 operations. Reduced-precision
  learning need not match the old F32 trajectory bit for bit.

The implementation boundary is explicit:

1. Naga: a distinct BF16 scalar encoding, validation, conversion and SPIR-V
   emission. BF16 is not another name for IEEE FP16. Initially support storage,
   conversions and cooperative products; widen ordinary elementwise arithmetic
   to F32. The [SPIR-V BF16 extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_bfloat16.html)
   restricts which instructions can consume BF16 values.
2. Blade: update Vulkan bindings as needed, query/enable the exposed
   [BF16 feature](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_bfloat16.html)
   and enumerate supported cooperative operand/accumulator combinations.
   No requirement for the unsupported BF16 dot-product feature. Unsupported
   devices retain the existing path; do not expose an unexecutable dtype.
3. Meganeura: typed BF16 buffers and boundary conversions, F32-master training,
   then connected reduced-storage regions through the actual convolution /
   dense / activation / layout pipeline. First qualify conversion edge cases
   and F64-referenced matrix gradients, then repeated complete updates.

The native compiler/Blade foundation, Meganeura storage/matrix consumers and
explicit dense/pointwise forward/backward regions pass scoped checks. Forward
convolution storage is now native BF16 too; RGB world training benefits, CDP's
full-BF16 world path does not. Keep the measured mixed-stage alternative for
CDP. Convolution derivatives and grouped products retain F32; broader CDP
world/backward reduced storage remains an optimization opportunity, not a reason
to force the slower mode. No driver change, CPU emulation or isolated cast
benchmark substitutes for these complete-update measurements.

### 3 Reduce intermediate traffic and dispatch overhead

- Reprofile after the convolution change. Rank remaining costs across dense /
  recurrent products, transposes, normalization, activations and synchronization.
  Separate host waits, GPU timestamps and ordinary latency; none is SM usage.
- Implement the largest measured opportunity using existing fusion/layout
  machinery where possible: avoid a materialized transpose, fuse a producer
  epilogue or consumer prologue, or remove a redundant synchronization.
  Preserve buffer lifetime, visibility and gradient dependencies.
- Compare dispatch counts, intermediate bytes and ordinary whole-update time.
  Do not retain complexity for a microbenchmark win that regresses the learner.

## Execution and acceptance

- [x] Preserve the old controls and stop the current learning service.
- [x] Recheck both upstreams; use an isolated Meganeura worktree to preserve
  neighboring user changes. Only the user merges upstream PRs.
- [x] Qualify a useful complete-update convolution improvement: hardware-row
  selectionPR252 passes scoped checks and both full-update timings. PR248 stays
  unadopted; the retained unrelated weight-oracle failure is not erased.
- [x] Establish BF16 capability and qualify native compiler/storage/matrix primitives.
- [x] Select BF16 dense regions across learner sessions and qualify complete updates.
- [x] Extend connected reduced storage through convolution and measure whole
  world training: RGB benefits; CDP retains the faster F16 world path under
  `mixed`. Full CDP world/backward BF16 coverage is not achieved.
- [x] Implement and qualify the selected row-addressing layout improvement;
  upstream merge/adoption remains separate.
- [x] Repeat the fixed native/JAX comparison and report remaining gaps (October12).
- [x] Finish the combined pinned-stack comparison, audit and runtime CI review
  (CI355 passes). Do not add individual percentages or treat finite synthetic
  updates as learning.
- [x] Attribute the native GPU bottleneck and qualify the minimal split-tile
  fix on refreshed main, including exact full-update and plan comparisons.

Use small reviewable upstream changes with portable fallbacks. Reuse the
existing benchmark and guard rather than create another orchestration framework.
For each mechanism, one correctness suite and matched warmed timing suffice;
retain failures and review them before another candidate. GPU work remains
serialized in bounded, non-restarting services with >=2 GiB sampled Vulkan
budget headroom. No compilation during matched timings. No separate NVML polling.

The engineering target is to approach the 24.36 ms JAX FP32 update first, then
the 12.40 ms BF16 reference, without assuming either number is guaranteed.
Report kernel gains and full-update gains separately. Synthetic updates and
backend qualification are allowed during this pause; Atari training and frozen
game rollouts stay stopped until this backend round is reviewed. Return to a
fresh bounded learning check only afterward, retaining all three learner seeds
and score-versus-actions/time evidence before changing Kindle defaults.
