# Kindle: a single actor that learns while acting

Updated October 16, 2026. This is the authoritative roadmap.
[PR31](https://github.com/kvark/kindle/pull/31) is the done/running/next dashboard;
[experiment reports](experiments/README.md) retain detailed evidence and failures.
[AGENTS.md](../AGENTS.md) gives working rules. No separate status document.

## Performance pivot

**Current work: [backend training performance](backend_training_performance_plan.md).**
Fresh isolated complete RGB12M/B8/BF16 updates take **22.780 ms versus JAX
12.737 ms: still 1.79×**. Production remains on its earlier qualified~53 ms
runtime; no Atari restart or learning-quality claim. CDP/B32 is a different
workload and must not be compared directly with this JAX number.

**Latest complete B32 result:** [typed word staging](results/2026-10-16-word-staging.md)
reduces RGB51.763→48.068ms,7.14%, and CDP46.929→45.505ms,3.03%, with exact
reports/targets/tensors and full plans/shaders. RGB remains1.77× inclusive JAX.
Keep isolated native834241b6 with word staging/image views/device means/cached
recording; no production or Atari promotion. The124 guards pass with five
warnings. Native counters establish instruction/shared-feeding improvements,
but substantial instruction overhead remains. No GPU worker remains.

The subsequent [packet-axis test](results/2026-10-16-packet-axis.md) passes
numerics but regresses the complete convolution family11.15%. Restoring uniform
instructions in dW increases total instruction work; reduced dX L2 traffic does
not improve time.120 guards/22 warnings and all offline audits pass, with no
full-update trial or promotion. Keep834241b6 and1.77×.
The [subsequent CPU graph audit](results/2026-10-16-nonconvolution-contract.md)
finds training-prior work unnecessarily unrolled:144 matrix products versus9
batched JAX products, identical arithmetic outside the posterior dependency.
The [isolated prior-batching implementation](results/2026-10-16-prior-batching.md)
reduces standalone products144→9 and passes all paired checks, but independent
F64 gates fail in both arms. CPU rounding diagnosis is partial; preserve failures
and resolve the reference before actual-world qualification or full timing.
One guard/no warnings,16 raw backwards, no optimizer updates/actions. New Blade
workgroup-matrix support is a separate feeding-contract review, not a repin.
Basic head batching is already present. All eight JAX
updates reconcile, but native pass intervals and JAX kernel sums are not
subtractable savings. No new layout/tile sweep or automatic GPU allocation;
convolution feeding and parity remain open.

**Preceding B32 result:** [connected image views](results/2026-10-15-image-views.md)
reduces RGB54.205→51.945ms,4.17%, and CDP47.699→47.112ms,1.23%. Exact reports,
targets, saved tensors and full-program audits pass. RGB remains1.91× inclusive
JAX27.223ms. Keep isolated67559ce5 with image views/device means enabled; no
production or Atari promotion. Kernel data feeding and device execution remain
structural open work, not another BF16 switch.
The [CPU pipeline review](results/2026-10-15-feeding-pipeline.md) identifies a
specific Naga limitation: a subgroup-uniform producer/consumer split validates
as Vulkan SPIR-V but is rejected by Naga's cooperative-operation analysis.
Reduced-precision double buffering fits the retained resource limit, but the
[actual full-family trial](results/2026-10-15-feeding-family.md) now rejects this
schedule:22.412/24.418/46.671ms native/GLSL serial/separate loaders. Every contract
regresses, despite passing numerical checks and aggregate calibration. No Naga
project, full-update trial or favorable selector follows. The subsequent
[CPU output audit](results/2026-10-15-output-contract.md) proves exact layouts can
avoid a full shared output tile, including a4KiB fragment alternative supported
by current Naga. The resulting [complete resource-bounded policy](results/2026-10-15-reuse-policy.md)
now passes numerical checks but loses21.72%:22.327→27.177ms across all23 products,
18/20 contracts regressing.120 guards/no warnings; no integration or favorable
selector. Source staging drops33.93%, but output barriers and in-loop coordinate
work increase; these are not hardware-counter costs. Keep compensation, numerical
gates, isolated67559ce5 and the1.91x whole gap. No new kernel sweep or Atari run.
The supplied Perf SDK now enables [direct native counters](results/2026-10-16-native-counters.md)
on the existing driver: all20 contracts/23 occurrences pass, with exact retained
outputs and no faults. Native CNN1 dX/dW execute15.94x/9.04x the warp instructions
of the retained full-JAX kernels. Low DRAM throughput and substantial shared-bank
conflicts point toward scalar/on-chip data feeding, not another BF16 switch.
The word-staging result above now qualifies that contract and narrows the gap
to1.77x. Preserve tiles, rounding and numerical gates; parity remains open.
No tool/approval blocker or active worker remains.

**Earlier B32 step:** [GPU categorical means](results/2026-10-15-categorical-gpu.md)
reduces complete RGB updates59.963→54.224ms,9.57%, retaining the earlier
[general ordering](results/2026-10-14-convolution-scheduling.md) improvement.
Three distribution readbacks25.1MB→98KB; all28,860 independent F64 checks and
873 exact symmetric zeros pass. Initial/first292 tensors and targets match,
but later F32-vs-F64 trajectories differ; no learning-equivalence claim.
Four guards/three warnings,545 synthetic updates, zero actions/workers.
Still2.04× retained JAX26.632ms. Keep isolated99f8aab0 with device means enabled;
production, B8 and learning are not promoted/remeasured by that result. The
separate CDP recheck follows below. Targets remain on CPU.

**New CDP cost finding:** [Mixed→BF16 on the current binary](results/2026-10-15-cdp-precision-recheck.md)
reduces full updates78.206→50.122ms,35.91%. The old Mixed policy excludes later
BF16 derivative/staged-posterior improvements; posterior+world falls49.788→
21.715ms, imagination stays23.94ms. Four guards/exact346 initials/finite states
and each arm's own smoke prefix pass;592 synthetic updates, no game actions.
Trajectories differ. The [scoped BF16 qualification](results/2026-10-15-cdp-bf16-qualification.md)
now passes5.59M independent projection/loss values and exact actual whole/staged
gradients, late-target isolation and unchanged346 tensors. The original JSON
observer failure remains, with5.62M exact underlying F32 values and a separately
reviewed parser repair. Two passing/one failed guard, one warning; no full-model
independent oracle, learning parity or promotion. The [imagination attribution](results/2026-10-15-cdp-imagination-work.md)
now preserves all outputs while replacing the closed ensemble branch: fixed
GPU20.771→8.705ms, conditional12.066ms, not a learner speedup. Existing ensemble
projections add64.42G MACs; four repeat/transpose chains also materialize1.132GB
of logical intermediates because address lowering and typed normalization fusion
do not compose. The [qualified compiler fix](results/2026-10-15-repeated-row-composition.md)
now saves4.83% of complete CDP updates,50.151→47.728ms, with exact272 reports and
346 initial/first/final tensors. RGB54.129→54.245ms remains2.04x JAX. Conditioned
F64 and exact pairs pass; both compilers' original near-zero scalar gate failure
stays retained, not relaxed. Eight passing/two failed guards/five warnings,
1,136 synthetic updates, no game actions/workers. Keep isolated076f3135, with
99f8aab0 as control; no production or learning promotion.
The [explicit matrix feasibility screen](results/2026-10-15-convolution-matrix-feasibility.md)
rejects that route: all23 products' patch construction alone costs34.348ms,
before GEMM or output work. All6.647M exact values/47,288 actual samples and
two guards pass, no new warnings/actions/updates. The subsequent
[fixed-geometry halo-cache trial](results/2026-10-15-convolution-halo.md) also
loses: all15 forward/dX occurrences13.706→18.598ms,35.69% slower, despite exact
F64/output pairs. Six guards/two warnings, no actions/updates. Failed B1
kernel-selection coverage and its separately reviewed repair remain. No
full-update trial, geometry sweep, CUDA learner or learning restart follows.
The [refreshed complete trace](results/2026-10-15-whole-update-attribution.md)
now passes: ordinary54.113ms, still2.03x JAX. Coarse instrumentation adds1.916%;
47.699ms of55.152ms lies inside GPU passes, cached programs46.484ms. Helpers
and host work alone cannot explain the residual. All272 reports/292 tensors
at three boundaries and plans match exactly; two guards/no warnings,544
synthetic updates, zero actions. Timed chunking adds three waits; coverage
is not utilization. The [current B32 command-buffer control](results/2026-10-15-jax-command-buffer-b32.md)
now passes: JAX inclusive27.223ms with graphs/29.505ms without. Native remains
1.83x slower even without graphs; exact reports/tensors/fusion computation/
library settings pass. Two guards/two warnings,544 synthetic updates, no actions.
Historical26.632ms was JAX's core boundary; the current inclusive comparison
is54.113/27.223ms,1.99x, not a new native speedup. The subsequent
[closed-image attribution](results/2026-10-15-spatial-boundary-cost.md) now passes:
non-convolution image substitution saves4.173ms; replacing the connected image/
convolution region with exact escaping values reduces fixed-world34.195→6.564ms.
Only7.014MB of its3.154GB logical outputs escape; poison/version/exact-output
checks pass. These interacting oracle effects are not implementable full-update
savings or CDP timings. The [image-interface audit](results/2026-10-15-image-layout-contract.md)
now finds repeated NCHW materialization/casting/packing where JAX keeps BF16 NHWC
interfaces. Eight duplicate packs are version-proven, not measured savings.
Fixing a four-product packing launch cliff saves only2.13% of complete RGB
updates, below its3% gate; CDP saves3.08%. Structured numerics/control trajectories
pass, but full timing-program exports were disabled and that gate remains
incomplete. No retime/promotion; keep076f3135 and the inclusive1.99x comparison.
Next prove direct producer-to-consumer converted/expanded packed views on CPU,
including forward/both derivatives and required saved/external values, before
a complete-update allocation. Layout alone does not reproduce library data
feeding. Production and all quality/budget gates remain unchanged/open.
The [connected-view qualification and timing](results/2026-10-15-image-views.md)
now pass: RGB4.17% faster, CDP1.23%, with all272 reports and initial/first/final
292/346 tensors exact. Full program audits reconstruct only declared image-view
changes; convolution algorithms remain unchanged. Eight guards/seven warnings,
1184 synthetic updates, zero actions; no worker. Keep isolated67559ce5. Its31
removed dispatches/876.610MB logical writes are not DRAM measurements; the
measured2.26ms gain leaves a1.91x gap. No production promotion or learning restart.

**New root-cause evidence:** [the same derivative algorithms remain slow in CUDA](results/2026-10-14-convolution-backend.md).
Native dX/dW2.148/2.019ms versus CUDA1.872/2.441; exact outputs pass, but both10%
timing bands fail, so CUDA counters are not native measurements. They show~58%
integer instructions and~9% tensor activity: feeding the tensor operations is
still costly. A physical-device query confirms an unused higher-level option:
cooperative-matrix2 workgroup matrices/tensor addressing, with BF16→F32 support.
Our Naga writer currently hard-codes subgroup scope. The [native proof rejects](results/2026-10-14-workgroup-convolution.md):
actual dX2.14876→8.71198ms despite exact outputs and independent checks. Do not
start compiler/backend feature work or tune that prototype. The retained full
JAX trace instead shows four-stage256x32/64x256 convolution tiles versus our
single-stage32x32, with8–10× fewer blocks. Layout/vector/reuse/staging differ
together; tile ratios are not speedup estimates. The [actual library captures](results/2026-10-15-library-convolution-feeding.md)
now verify asynchronous128-bit global-to-shared copies and10.33x/8.27x fewer
warp instructions than the current CUDA translations, despite more tensor
instructions. Both exact272-report/288-tensor/HLO checks pass. This is substantial
supporting-work overhead, not simply low occupancy. Native instruction counts
remain unknown; the retained driver query returned no internal representation.
The [aligned/reused native proof rejects](results/2026-10-15-aligned-convolution.md):
2.147→7.433ms, with only.152ms in conversion; the tensor interval also includes
strided stores. Exact outputs pass; explicit alignment metadata has no material
effect. Output-store and BF16/F16 controls do not help. A
[fixed-valid-input contract](results/2026-10-15-convolution-address.md) then cuts
the proof7.438→1.877ms including GPU halo padding—fourfold relative to that
failed proof, only12.59% faster than retained native2.147ms. Qualified trace
assigns.179ms packing/1.700ms to the remaining tensor interval. Both forms
report255 registers; smaller code supports different compiler lowering, not
measured spill bytes or exact instruction attribution. Testing the observed
library K32 reduction step for operand liveness yields another improvement: that
[fixed test now saves31%](results/2026-10-15-convolution-contraction.md), to1.293ms
including transforms,39.79% below retained native2.147ms. The [broader F64
qualification and fresh native pair](results/2026-10-15-convolution-qualification.md)
now pass:196,608 outputs and2.14890→1.29280ms,39.84%. But this operation occurs
once; its.856ms saving is only1.43% of the full update before integration effects.
The [actual spatial-family check](results/2026-10-15-convolution-family.md) now
rejects that shared replacement:12.833→18.131ms across13 distinct contracts/
15 occurrences,41.29% slower, despite all numerical gates and257.95M exact
output pairs passing. Keep per-shape regressions and the nonstationary native
dX samples; no cherry-picked selector, geometry sweep or Naga/Blade integration.
Device-side categorical decoding now qualifies above; world training remains
30.20ms, longer than the entire JAX update. The [whole-family audit](results/2026-10-15-convolution-contract.md)
now matches all23 products and exposes a concrete Naga limitation: matrix loads
cannot retain F32/F16/BF16 arithmetic types independently of packed integer
storage. Six CPU reproductions and three validated Vulkan SPIR-V controls pass;
no compiler change or GPU speedup is claimed by that audit. The subsequent
[transfer observation](results/2026-10-15-convolution-transfer.md) finds a real
but small local benefit:4.933→4.170ms across all14 packed occurrences,15.48% less
complete-operation time. That.763ms is only1.41% of the retained full update as
a scale comparison; no complete update is remeasured. Qualification remains
failed at40 forward/1,277 dX cancellation values shared identically by native
and both variants. Exact integer references confirm the failures; the separate
cost-only observation repeats no cohort and authorizes no compiler promotion.
132 guards/six warnings, zero learner updates/actions/workers; all failures stay.
A separate properties-only query confirms a more structural constraint:
Vulkan exposes48KiB explicit workgroup shared memory, while19/23 captured
cuDNN products exceed it, including76/80KiB dX/dW. A literal port does not fit;
this is not proof of a causal slowdown or inability to reach parity. The
[fixed library control](results/2026-10-15-library-pipeline.md) reconstructs
original engines40/33, but both reduced-stage configurations fail before
execution-plan construction. Two metadata-query failures also remain; four
failed guards, zero executions/new warnings and four truncated failure-log
snapshots. Local plans confirm CUDA99KiB opt-in capacity. No stage/engine/tile
sweep or timing follows. Installed9.24 lacks the newer memory-limit attribute;
no invented enum or silent upgrade. Serialized frontend=false is not proof
of legacy execution. The [exact saved-output comparison](results/2026-10-15-convolution-value-bypass.md)
now passes: replacing all23 products with recorded-value copies cuts fixed-world
GPU34.191→14.282ms,58.23%, preserving allocations/consumers/dependencies and all
checked outputs/tensors. This resolves the old invalid-intermediate masking
problem, not ordinary training speed or pure kernel latency. Copy-only2.005ms
includes original barriers and is not a subtractable saving. Two guards/zero
new warnings; original binding-contract failure and CPU audit OOM remain beside
their reviewed repairs. The [role comparison](results/2026-10-15-convolution-role-cost.md)
now assigns6.896ms to encoder and13.039ms to decoder substitution, with.069ms
interaction; normal/both34.216/14.280ms. About65% belongs to RGB reconstruction,
absent from CDP. These are valid-value intervention differences, not pure kernel
latencies or measured CDP gains. All four conditions, live/full outputs, three
292-tensor pairs and complete graph/source audits pass; one guard/zero warnings,
647 fixed replays/one initializer/zero actions. No worker or production change.
The source review and existing-policy comparison above reconcile the next CDP
contract: exploration projection reuse is already present, while Mixed excludes
the staged BF16 path. No decoder-only CDP claim or new subset/geometry/iterator
sweep follows.
The earlier offline capability audit confirms
explicit SPIR-V async-copy/wait/event is not a Vulkan Shader primitive; Matrix2
compiler-managed staging is different. No claim that the driver cannot generate
async instructions or that exposing one API would close the gap. Production
unchanged, no8x forecast or full-gap closure.

The [retained whole-update diagnosis](results/2026-10-14-jax-image-profile.md)
finds a structural compilation gap: native B32 spends56.948ms in GPU passes
and11.688ms outside them, including5.383ms host decoding/targets. Exact-state
native/JAX captures pass. Native materializes more image intermediates and
layout conversions, while JAX combines forward/backward work and retains
compact storage. Even zero host overhead would not close the gap. The first
[connected spatial lowering](results/2026-10-14-spatial-tape.md) now qualifies:
fresh complete B32 updates68.491→65.736ms,4.02%, with exact272 reports and
three292-tensor boundaries. It saves forward pooling decisions and combines
typed bias/backward gradient/partial-reduction work. The block improves30.27%,
but only6.67% of the whole gap closes; the candidate remains2.42× retained JAX.
The [spatial-vector convolution intervention](results/2026-10-14-convolution-spatial-staging.md)
is rejected: its targeted decoder pair is0.62% slower with exact outputs, so no
whole-update trial follows. The numerical run's final count assertion stays
failed; independent CPU reconstruction passes all93,280 saved values without a
GPU retry. The subsequent [connected layout/reuse trial](results/2026-10-14-connected-layout.md)
also rejects:65.651→65.593ms, only0.089%. It removes eight packs/ten casts/two
transposes with exact reports/states and qualified staged backward lifetimes,
but those static redundancies were not a major cost. Keep3551bc53, not rejected
212713c4. [Packed shape specialization](results/2026-10-14-packed-specialization.md)
also rejects:65.684→65.207ms,0.726%, with exact reports/states/plans and13.3M
F64 comparisons. No shape sweep. The omitted [wide-image JAX control](results/2026-10-14-image-precision-control.md)
now measures26.632→26.876ms, only0.92%, with unchanged heavy multiplication
work and exact288 initial tensors. All544 updates and22,120 independent
image-tail values/derivatives pass; no new native kernel or learning claim.
JAX stays fast with F32 image intermediates while still combining their work.
The [connected image forward/backward candidate](results/2026-10-14-image-training-region.md)
now passes independent numerical checks and barely clears its whole-update
gate:65.609→63.640ms,3.0015%, still2.39× retained original JAX. It retains the
existing precision/heavy arithmetic across eight encoder/decoder regions;
reassociated reductions change subsequent tensors/trajectories. Keep it isolated,
without robust-speed, learning-equivalence or production promotion claims.
The subsequent exact-restoring masked diagnostic cannot rank the remainder:
removing unrelated operations makes image-enabled execution slower, producing
a negative formal allocation for other work. Do not interpret it as kernel
latency. The [intact sparse trace](results/2026-10-14-middle-convolution.md)
preserves exact results but fails its3% overhead gate at3.346%; its mixed scopes
are not ordinary component costs. Exact actual intermediate encoder derivatives
separately take2.148/2.015ms versus BF16 cuDNN.479/.550ms, with all16.8M rounded
outputs agreeing. Both miniature library tunings differ from full JAX; an
auxiliary F32/Winograd control fails and remains unqualified. A shared-group
control finds no interference: grouped/serialized both4.165ms with100.9M exact
checks. The [captured full-group follow-up](results/2026-10-14-convolution-boundary.md)
now isolates a context that control omitted: three auxiliary operations. Their
standalone cost is only.0186ms, but one barrier before the two convolutions
changes the exact group7.387→4.168ms. It also delivers the complete-update
gain above. Real/synthetic input controls and original allocation mappings
pass; separated timing is an intervention, not passive attribution. The
[ordering follow-up](results/2026-10-14-convolution-scheduling.md) now retains
the full local gain with empty memory-barrier arrays, and qualifies the general
rule above. Explicit access scopes are unnecessary; hardware/cache causes remain
unmeasured. JAX's retained trace is effectively non-overlapping, not strictly
serial or native-overlap evidence. The derivative/backend study above follows
that audit; its workgroup-matrix follow-up rejects above. The library's complete
data-feeding algorithm, not a tile/split/geometry sweep, is the next question.
No blanket serialization or masked ranking.
Host decoding/targets remain a separate GPU-residency opportunity.
No pool/row/grid/tile/dtype sweep or broad capture. Production and Atari stay
unchanged; the full performance goal remains open.

The [same RGB comparison at B32](results/2026-10-14-batch-granularity.md) gives
68.456ms native versus27.187ms JAX,2.52×. Larger batches improve throughput
1.33× versus1.87×, so they do not resolve the implementation gap. Four native
convolutions lose packing at a one-dimensional launch limit:65,536 workgroups
versus the selector's65,535 portable bound. The [qualified two-dimensional
correction](results/2026-10-14-packing-grid.md) now saves only1.49% of fresh
complete B32 time,68.344→67.326ms, below3%; do not promote it. Candidate remains
2.48× retained complete JAX. Actual5080 X limit is2,147,483,647, so these were
conservative selections, not invalid launches. Independent exact packing/F64
checks and seven guards pass; B8 smoke/control states are exact, B32 candidate
trajectories change. This does not explain the dominant gap. The
[completed cached attribution](results/2026-10-14-batch-residual.md) finds
convolution and other-than-matrix work supply89.0% of the fixed-graph B8→B32
increase; empty command cost does not grow. World layout/casts, reductions
and pointwise/image work allocate4.442/5.379/5.893ms. Exact full-plan/state
restorations pass; these are not pure kernel times or removable savings.
Actual first-CNN dataflow writes640MiB of summed logical intermediate outputs,
while JAX combines conv+bias and pooling+statistics in channels-last storage.
Not peak memory/DRAM traffic or identical rounding; JAX still has backward
tapes/work. The [complete first CNN block](results/2026-10-14-image-block.md)
now measures6.218ms; exact-restoring masks allocate78.4% outside convolution.
Small F64 checks pass, but the miniature JAX large-gradient gate fails, so
there is no cross-stack block speed claim. The actual complete B32 JAX trace
and native whole-update accounting now finish above; the subsequent connected
lowering passes its whole-B32 gate but leaves most of the gap. No batch/grid
sweep or learning promotion. The historical B32 control is68.344/27.187ms,
2.51×; the new isolated candidate is65.736ms,2.42×.

The [executable audit](results/2026-10-14-executable-work.md) finds only1.40%
more native nominal heavy arithmetic. The gap is execution engineering, not
a much larger Dreamer workload. Three structural differences have evidence:

- Kernel algorithms: the [latest matched convolution study](results/2026-10-14-boundary-algorithms.md)
  isolates decoder forward/dW at1.310/1.742 ms versus cuDNN.261/.338 ms.
  Its checked CUDA bridge executes17.84× as many warp instructions, mostly
  integer indexing, although cuDNN issues12.5% more identical BF16 tensor
  instructions. Bridge counters are not native Vulkan counters; two primitive
  library algorithms differ from the full executable, dominant dW matches.
  The [native iterator intervention](results/2026-10-14-convolution-iterators.md)
  now cuts decoder dW37.15% and complete updates4.45% with exact reports/states:
  repeated indexing is a demonstrated cause, not the whole explanation.
  The [matrix attribution](results/2026-10-14-matrix-cost.md) finds the checked
  small-product path costs2.903ms/63.54% of matrix time for12.01% of matrix MACs;
  all already use BF16. The [matched matrix counters](results/2026-10-14-matrix-library.md)
  now find identical tensor work/near-identical occupancy, but3.40× instructions
  in a CUDA translation reproducing native grouped-forward cost: scalar16-bit
  staging versus cuBLAS128-bit vectors. Four actual contracts pass exact outputs;
  whole JAX layouts/addends/storage differ, counters are not native Vulkan.
  The [shared typed vector-staging intervention](results/2026-10-14-matrix-staging.md)
  now saves4.75% of complete updates with exact reports/states. Same arithmetic,
  tiles and dependencies; grouped forward19.27% faster, no fresh counter claim.
- Host construction: [command reuse](results/2026-10-14-command-cache.md)
  saves10.16% of complete update time with exact reports/states. It removes
  repeated bindings/recording, not GPU operations or dependencies.
- Device execution: [dependent-kernel controls](results/2026-10-13-execution-boundaries.md)
  find about2.05µs per dependency for Vulkan/ordinary prequeued CUDA versus
  .65µs for CUDA graphs. Host command reuse does not establish that advantage.
  The [legal dependency-path comparison](results/2026-10-14-dependency-paths.md)
  finds synchronization2/exact-buffer barriers unchanged and immediate events
  about2× slower. All3,204 counters pass; close this API-flag detour. This is
  not a universal Vulkan limit or a fresh complete-learner measurement.

The [wider-intermediate control](results/2026-10-14-precision-flow.md) now traces
an actual forward/backward block and measures complete JAX12.684→13.469ms,
only6.19% more. F32 matrix derivatives/interfaces can remain close to JAX speed;
both original stacks already accumulate F32 master gradients. Independent
numerical/work checks pass. Explicit algorithm selection also changes compiler/
library choices: not a storage-only ablation or exact native clone. No further
precision sweep or conditional capture; changed trajectories are not learning
equivalence. All failures remain.

The [current cached capture](results/2026-10-14-cached-residual.md) now attributes
19.697ms to GPU-pass intervals versus3.458ms uncovered, with0.50% instrumentation
overhead and exact reports/states. Fixed-graph costs include4.295ms empty command
stream and5.463ms other-than-heavy work, alongside5.587ms convolution/3.388ms
matrices. These are interacting allocations, not additive removable costs or
SM utilization. The connected heavy-projection comparison finds separate JAX
tape/reduction/backward kernels too; counts alone do not justify a new fusion
trial. The packing-grid intervention above supplies only1.49%; the completed
B32 attribution points to connected image-training dataflow. The five checked
convolutions occupy only~2.50ms; prior row-only fusion saved~1%. Keep matrix
efficiency, materialization and device execution in scope. No iterator/tile/
vector-width/occupancy sweep, scalar fallback revival, BF16-only fork or tolerance relaxation.
Detailed successful and rejected interventions remain in the backend plan and
linked results; individual savings are not additive. The full goal stays open.

### Retained causal findings

The [execution-design diagnosis](results/2026-10-13-execution-gap.md) finds a
fundamental implementation gap: basic kernels/incomplete training precision
and fragmented learner dataflow versus optimized libraries/compiled device
work. Directly translating our kernels to CUDA does not fix their substantial-
kernel latency; matched cuBLAS products remain faster. JAX12.42ms becomes only
22.78ms with our missing precision regions and command-buffer advantage removed,
versus native~53ms. Current priority is efficient type-aware kernel families,
grouped/backward reduced precision and connected learner dataflow—not a new
dtype-specific shader, CUDA learner or more local tile tweaks. Latest compiler
refresh brings no speedup; all changes remain diagnostic. No game restart or
speed-parity claim. The earlier accounting located cost but did not establish
complete causal closure; the new report retains its limitations and failures.

The [shared tiled prototype](results/2026-10-13-tiled-gap.md) now makes one actual
transposed product2.92x faster, but saves only2.59% RGB/6.43% CDP full-update time
with exact reports/tensors. World training is untouched; RGB is still4.16x JAX.
Next isolate convolution-backward coverage, then grouped products and connected
learner dataflow. No production adoption or dense-tile sweep; investigation stays
active and the complete performance target remains open.

The isolated [backward-coverage experiment](results/2026-10-13-conv-backward-gap.md)
then saves8.01%, reaching48.90ms—not close to JAX12.42ms.90.51% of nominal
matrix/convolution MACs select BF16 kernels, showing that dtype coverage alone
does not supply efficient execution. Numerical checks pass after a retained
long-reduction failure/correction; learning quality is untested. Next examine
convolution kernel counters, grouped algorithms and connected learner dataflow.
No production adoption, combined-candidate speed claim or game restart.

[Counter-guided convolution interventions](results/2026-10-13-conv-kernel-cause.md)
give concrete algorithm evidence: layout/iterator and reuse improve a CUDA twin
from~629 to211µs across two controlled pairs, including packing and exact outputs.
Instructions fall84%, while occupancy also falls. Useful tensor work, not more
resident threads, is the target. Port through shared native algorithm families
and measure whole updates; this is not already a3x learner improvement.

The [typed cooperative matmul consolidation](results/2026-10-13-typed-matmul.md)
qualifies: shared checked F32/F16/BF16 generation and typed partition tuning,
with explicit storage/arithmetic and specialized algorithms retained. Complete
RGB52.803→53.047ms/CDP124.997→126.182ms updates are within the declared2%
non-regression gate, not faster. All272 reports/targets and292/346 saved tensors
match exactly;13 guards pass, two warnings and corrected development failures
remain. Pin056562b, unchanged Blade/Naga. No native-F32 device qualification
on5080, universal parity claim or game training. Both locks/native checks,
Clippy and193 binding tests pass; CI stays on PR31.
The [BF16 split-K optimization](results/2026-10-12-bf16-split-k.md) qualifies:
RGB56.269→53.065ms / CDP126.880→125.432ms complete synthetic updates, independent
matrix/gradient checks and ten passing guards. Historical pin9e05731 with unchanged
Blade/Naga; two development failures remain. Summation order changes stochastic
trajectories; game quality is untested. CDP's dominant54.21ms imagination stage
is unchanged. Continue matrix/grouped efficiency and derivative precision work;
no training restart or claim of JAX parity.
The [full JAX gap audit](results/2026-10-12-jax-gap-accounting.md) confirms equal
trainable model size and rules out input transfer as the main explanation.
Both incomplete reduced-precision coverage and inefficient native BF16/F32
kernels are measured; repeated posterior work and CPU targets/EMA add overhead.
The current operation-group accounting closes the full measured gap: graph
costs17.32ms convolution/14.04ms matrix/6.84ms remaining plus5.89ms zero-work
command stream match complete-update passes within0.014ms. Host work and
profiling/boundary adjustments remain explicit. This is cost accounting,
not causal closure, JAX speed parity or independently achievable savings.
Keep four-chunk scheduling and learning stopped; next address
matrix launch geometry/coalescing and grouped precision alongside convolution
derivatives and command-stream reduction, then posterior reuse/device residency.
No concurrent-seed workaround or indefinite profiling gate.
The [upstream split-tile fix](results/2026-10-12-split-convolution-tiles.md) is
qualified and pinned on refreshed main: RGB61.027->56.058ms/CDP135.070->126.739ms,
exact numerical controls and all17 guards. Native checks, both locks, Clippy
and193 installed-binding tests pass. No training restart or learning-quality claim.
The latest [Nsight investigation](results/2026-10-12-nsight-root-cause.md) identifies
expensive convolution derivatives and matrix kernels, not just host overhead.
One isolated split-tile change reduces RGB60.98->55.97ms and CDP135.13->126.61ms
with exact metrics/tensors; the qualified fix above supersedes that diagnostic
variant. Counter access is
enabled, but Graphics shutdown crashes limit captures to diagnostic evidence;
Systems native timestamps stay rejected. Tile selection is now adopted;
the new matched controls above refine the remaining execution work.
The remaining JAX gap and learning quality remain open; no game training resumes.
The user prioritizes all three remaining opportunities before more Kindle
training: cooperative convolution weight gradients, real mixed precision /
BF16 where supported, then measured fusion/layout/dispatch improvements.
The Freeway screen below is stopped at39,568 actions/9,831 updates on its first
learner; the partial checkpoint remains. No completed learner, later arm or
frozen cohort. Synthetic backend qualification continues, not game training.
The [combined backend result](results/2026-10-12-backend-integration.md)
qualifies native BF16 storage, repeated-row addressing and hardware-row
convolution selection: CDP12M/B32 **161.32→135.17ms (16.21% less time)** and
RGB12M/B8 **72.69→61.15ms (15.88%)**. CDP uses `mixed`: BF16 learner inference/
behavior with the faster F16-input/F32-storage world-gradient path. RGB uses
`bf16`, including forward convolution storage. Auto default/live acting,
F32 masters/optimizer/reductions stay unchanged; no full-BF16 learner claim.
Ten guards/three retained warnings and the independent CPU audit pass. JAX RGB
FP32/BF16 **24.29/12.47ms** still leaves a2.52x/4.90x gap. The failed PR248
weight-gradient kernel stays out; all earlier failures remain linked in the
result. Current git pins include latest upstream baseseb8c90f/7909fc1; only
the user merges the review PRs. No learning/time-to-score or SM-utilization
claim, and no return to game training yet. Remaining performance work concerns
world/backward and recurrent costs, not another precision-policy sweep.

The user has selected a wall-clock performance round on the5080. The stopped
Pong queue stays cancelled. Optimize one useful learner before adding GPUs or
concurrent seeds; retain the current quality results as controls, not a mandate
to repeat a long mastery campaign.

1. Initial October11 update: Meganeurad7a7293; Blade56f0565 unchanged. The initial
   430f14b refresh adds two compiler-search fixes; five more Vulkan/packed-F16/
   convolution fixes land during this session. Preserve old runtime controls
   and retain the revisions used by each completed measurement.
2. Reuse each exploration head's state projection across all18 actions, then
   center in hidden space and project only the selected effect. Keep all-action
   centering, four heads, targets and coefficient unchanged. This is
   a new candidate under the user's revised arithmetic contract, not retroactive
   qualification of the rejected October7 result.
3. Expose checkpointed training precision. The original F16-input permission
   was not reduced storage; the newly qualified `bf16`/`mixed` paths above are.
   Keep F32 accumulators/master weights/optimizer/reductions. Check finite gradients,
   losses and checkpoints; judge adoption by learning and time-to-result, not
   bitwise trajectories or the old2e-5 end-to-end gate.
4. Compare B8, B16 and B32 at T16/R32, with full-batch centered CDP and no
   world microbatching. B32 consumes512 replay positions per update and updates
   every16 actions, versus128/every4 for B8. It reduces optimizer-step frequency
   and changes centering statistics; equal replay ratio is not identical
   learning. Stop sizes that violate the2GiB sampled headroom floor. Report
   time per replay position and whole-agent actions/s, not just update latency.
5. Profile matched native/JAX RGB work and native CDP/exploration separately,
   including compile time, ordinary complete updates, CPU/GPU gaps and hot
   kernels. Attribute differences in precision, model, exploration and batch;
   the old1M RGB comparison is not a current12M CDP comparison. Separate
   instrumented attribution from ordinary timing. Identify concrete Meganeura
   work rather than assuming that wider matrices are free.
6. The [performance result](results/2026-10-11-cdp-performance.md) selects
   B32/F16 for a now-stopped [three-seed paired Freeway screen](experiments/2026-10-11-cdp-performance-learning.md),
   200k actions per arm/seed against optimized B8/Auto. The synthetic replay
   throughput gain is2.66x; learning/time-to-score is not yet established.
   Keep all failures; no new five-game campaign or automatic extension.

Concurrent seeds can fill otherwise idle hardware without changing each seed's
batch, but do not inherently speed one learner and may compete for bandwidth
and memory. Wider batches are the first experiment. Neither idle capacity nor
room for multiple12M workers follows from a sampled VRAM estimate.

Cleanup so far: removed regenerable Rust debug builds from `target/debug` and
`python/target/debug`; filesystem free space549→612GiB (rounded). Rebuilding
restores them. Run evidence, checkpoints, videos and user-owned files remain.

## Preserved pivot boundary

**October11,20:25 UTC: training stopped at the user's request.** The existing
five-game goal is paused, not achieved. Nothing in that allocation will restart
automatically. The performance work above is newly authorized; the long-run
stages below are retained context, not an active training queue.

- **Preserved boundary:** Pong1009 stops gracefully at189,576 actions /
  47,333 updates after1h29m6s. Partial and actual-initial checkpoints, logs,
  declaration and source pins remain; saved partial hashes match the final log.
  Seeds2017/3019 and all evaluations are cancelled. There is no completed2M
  learner or new frozen result. See the [stop receipt](results/2026-10-11-cdp-pong-stop.json).
  A restore lacks replay/RNG/live belief and would not continue the same lifetime.
- **Reusable baseline:** qualified native CDP/RGB, eight batched environments,
  one GPU resize of native pixels,1M fast controls and12M quality candidates.
  The [game table](#current-game-status) links completed seed results and videos;
  Qbert's latest2M cohort and frozen diagnostics are complete. Preserve those
  controls rather than deleting evidence or changing their historical gates.
- **Unresolved:** reliable five-game quality, a matched RGB compute advantage,
  and transfer to native games. The interrupted Pong run settles none of these.
- **Decision to make for the pivot:** distinguish sample efficiency from wall
  time. Use matched-frame learning comparisons and time-to-score for any new
  study; keep the old long-run targets as historical references, not evidence
  that short-budget learning is broken. Current12M updates take about110ms and
  imagination, not world training alone, is the largest measured component.
  The performance round above supersedes this earlier stop boundary; the old
  long-budget queue remains cancelled and does not authorize concurrency.

The next work is the bounded performance round above, not an extension of Pong.

## Retained direction: CDP is the main path

Learn to play games through experience, using sparse explicit rewards and human
guidance where needed, and eventually useful intrinsic motivation. Keep learning
and inference native in Rust on Meganeura + Blade. Favor one effective actor,
small models and short experiments before larger budgets, concurrency or swarms.

**Adopt Dreamer-CDP as the main development architecture.** Keep Dreamer's RSSM,
replay and imagined actor/critic learning; train a small visual CNN jointly with
a deterministic latent predictor instead of reconstructing RGB. This advances
the JEPA bet on cheaper useful prediction. It does not require LeVJEPA, a frozen
video encoder, DINO or a detached visualization decoder.

The selected small recipe is Size1M, eight environments, B8/T16/H15/R32,
microbatch 8, replay 100000 and full BPTT. CDP has 804,785 trainable parameters,
including its CNN; the current exploration ensemble brings the whole learner to
1,057,201 parameters. There is no extra5.5M or303M frontend. Retain the
[qualified CDP loss and split learning rates](experiments/2026-10-04-cdp.md),
AGC and `ac_grads=false`. Change one factor at a time.

The CLI currently still defaults to learned-RGB reconstruction. **New main-path
experiment declarations must use `--cdp`.** This roadmap change does not silently
change runtime defaults. RGB remains the numerical/learning reference and a
fallback, not a mandatory extra arm in every exploration experiment. Causal
LeVJEPA is an optional future video/3D hypothesis, not a prerequisite for progress.

The causal model is: previous belief plus executed action produces a prior and
predicted embedding; the current frame's CNN embedding then forms the posterior.
The prediction target is detached and cannot enter its own prior. Encoder
gradients still arrive through world/task/value and temporal source paths.
Posterior state estimates are not forecasts.

## What is established

- **Fast iteration and a faithful control exist.** Shared/fused learning and a
  small batched recipe are complete. The original 12M 3x speed target remains an
  unmet, user-accepted stretch target. The small RGB port has
  [upstream numerical and three-seed learning evidence](results/2026-09-30-small-replication-learning.md).
  The old 45-run matrix is cancelled: 24 completed, 21 unstarted; never resume it.
- **Frozen Tiny did not earn its 2D cost.** The
  [three-seed comparison](results/2026-10-01-frontend-decision.md) found no clear
  pretraining advantage and 18% longer whole-agent time than RGB. The subsequent
  [joint-Tiny](results/2026-10-03-joint-tiny-learning.md) and
  [direct-policy-gradient](results/2026-10-03-policy-tiny-learning.md) early
  screens did not establish a useful benefit. Those results do not refute video
  priors or latent prediction. The unstarted posterior-Tiny ablation is deferred.
- **CDP improves the current Seaquest screen.** Three fresh paired seeds,
  200,000 actual actions / 49,939 updates per run: final online mean 543.6 versus
  RGB 318.1, paired difference +225.5 [62.8,330.4]. Every pair favors CDP, with
  35.9% less world-training time and 13.6% less wall time. The
  [complete result](results/2026-10-04-cdp-learning.md) includes all episodes,
  tails, curves, qualification failures and independent audits.
- **CDP's world representation is more useful, not solved.** Frozen probes find
  much more readable player state in the RSSM. Fifteen-step latent forecasts beat
  persistence, the constant training mean and unrelated actions. One-step
  persistence, zero-reward MAE and matched privileged coordinate persistence
  remain stronger. Sparse reward/terminal counts limit conclusions; there is no
  bullet/full-state sufficiency claim.

The controlled CDP/RGB comparison is one short game and three learner seeds,
not Atari-wide superiority or a reproduction of the authors' full benchmark.
The completed five-game exploration screen below now measures broader retained
learning, with important failures still unresolved.

## Current execution order — strategy reset

The [September strategy reset](strategy_reset_plan.md) remains the rationale:
iteration speed, useful representations, exploration/reward, video priors, then
real-time deployment. Phases 0–2 and the CDP follow-up are complete. The user's
October 4 direction puts CDP on the main path through the remaining stages.

### 1. Atari learning, exploration and budget (Phase 3)

The October6 goal is to reach **DreamerV3 quality on Boxing, Pong, Freeway,
Breakout and Qbert with CDP**, retaining the earlier budget comparison. All five remain in scope. Neither learning a subset nor completing a
queue establishes this goal.

**The first breadth screen is complete:** fifteen200k-action learners, including
all three retained Freeway cohorts. Frozen means are Boxing69.74, Pong−18.53,
Freeway24.14, Breakout4.36 and Qbert403.13. Four games improve over actual initial
controls in every seed; Pong improves in only one of three. All45 guards,720
selected natural frozen episodes, zero updates/cutoffs, exact saved tensors,
full trajectory replays and whole videos pass. [Results, curves and videos](results/2026-10-05-cdp-five-game-screen.md).

**No long-run DreamerV3 target is reached.** Keep the
[published references](results/2026-10-05-dreamerv3-five-game-reference.json)
separate: Atari57 uses200M frames; Atari100k uses400k frames and a different
nonsticky/minimal-action protocol. Our200k actions are about800k frames, not
less experience than Atari100k. At approximately the same early frame count,
published Atari57 is also weak on Pong/Breakout; our means are similar on
Boxing/Pong/Breakout, Freeway is ahead and Qbert somewhat behind. Different
online averaging windows and protocols prevent an exact parity claim.
Same-hardware/protocol RGB controls are required for a compute-saving claim.

**The longer Pong test is now complete:** three fresh500k-action learners
retain frozen scores−21.00/−2.375/−20.9167,7/72 wins. Only2017 improves;
the two failed seeds' KL drops further while reward separation stays weak.
Nine guards,144 selected natural episodes, unchanged tensors and full
replays/videos pass. [Result and curves](results/2026-10-06-cdp-pong-budget.md).
The online mean−15.27 is below the published−7.16 near2M frames, with protocol
differences retained. No further raw-cosine budget extension is declared.

**Frozen diagnosis is complete:** all four models preserve their346 tensors
over131,072 diagnostic actions. Every CNN retains readable ball/paddle state;
the two failed seeds' RSSMs do not reliably retain ball state or predict reward
events. Constant means nearly match their tiny cosine losses. Seed2017 retains
useful state and h15 forecasts. [Results and all controls](results/2026-10-06-cdp-pong-world.md).
This is not proof of complete encoder collapse. No privileged labels enter
learning; old-trained models are probed on newly qualified Meganeurab684ffd9.

**Centered CDP passes its three-seed early-learning screen:** the
[six fresh200k-action runs](results/2026-10-06-cdp-centered-learning.md) finish
October6 08:41 UTC. Frozen mean−8.4583 versus−18.6528 unchanged CDP; paired
gain+10.1944 [7.5,12.375]. Every centered seed improves over its actual initial
and matched control. All18 guards/288 selected natural episodes, exact tensors
and full replay/videos pass. Only4/72 centered matches are wins: not mastery.
Centering the training objective is the only changed mechanism; source CNN
gradients, capacity, rates, replay and exploration remain fixed. Promote
`--cdp --cdp-centered` as the next candidate, not the unchanged paper objective.
The ordinary `--cdp` control and CLI default remain unchanged.

**Paired world probes are complete:** six guards/exact346 tensors pass over
196,608 diagnostic actions, zero actor updates. All centered models retain
readable ball state and useful h15 negative-event forecasts; the raw2017
control remains stronger on some coordinate probes, and positive/terminal
evidence stays sparse. [Full result and controls](results/2026-10-06-cdp-centered-world.md).
One bounded
synthetic profile has completed; imagination alone has3,100 dispatches and
10.91ms ordinary GPU /13.28ms synchronized wall time. Kernel instrumentation
perturbs timing; do not turn its family shares into whole-agent percentages.
**The bounded throughput trial passes:** existing grouped B128 products reduce
ordinary synthetic network updates30.49->28.01ms (8.12%) and imagination
dispatches3,100->1,210 on the same updated Meganeurac6376542 backend. All14
guards, independent arithmetic/upstream checks and raw/centered production plus
frozen smokes pass. [Result and limits](results/2026-10-06-cdp-imagination-throughput.md).
Keep the change, without claiming a measured whole-game speedup or utilization.

**The five-game centered500k allocation is complete after evaluation repair:**
[result, curves and every whole video](results/2026-10-07-cdp-centered-five-game-budget.md).
All15 learners improve over actual initial controls; Boxing/Freeway pass the
historical gates in every seed, Pong remains unstable, Breakout/Qbert remain
weak. No long-run quality target is reached. Cost7.5M actions/1,874,085 updates /
16h24m7s training, plus all evaluation and earlier development compute. This is
longer centered-package learning, not a matched five-game loss/RGB ablation.
The repaired Breakout3019 natural cohort preserves its original cutoff and
exact207,288-action prefix; both CPU audit failures and extra collection remain
disclosed. All720 selected natural episodes and46 GPU guards pass.

**The frozen Breakout diagnosis now completes:** [result and limits](results/2026-10-07-cdp-breakout-world.md).
All six models remain exactly frozen. Ball state and h15 forecasts are weak,
but the pixel control is ill-conditioned: per-pixel whitening amplifies rare
changes to4,021 despite bounded RGB input. An F64 replay agrees with its saved
GPU predictions. The fixed-range2,048-update follow-up completes in5.01s with
no new actions/actor updates: paddle R².979, ball still near zero even in
training. This head is not an information ceiling. A fixed diagnostic color
centroid finds all483 eligible held-out open-playfield balls within.63/.55
source-pixel x/y RMSE; the GPU input retains those balls. No detector or RAM
enters the actor. The [existing12M preflight](experiments/2026-10-07-cdp-capacity.md)
passes195 finite updates and exact frozen restore with no new warnings.
**The three-seed12M capacity comparison now completes:**
[results, curves and whole videos](results/2026-10-07-cdp-capacity.md).
Frozen Breakout48.17/15.88/50.58, mean38.21 versus retained1M5.14;
paired+33.07 [12.38,44.17]. Every seed improves, but none reaches381.81.
The same500k-action recipe costs3.58x the training wall time:11h39m37s total.
All nine guards/144 natural frozen episodes/exact tensors/full videos pass;
three allocation warnings and all tails remain. No worker or automatic
extension remains for that Breakout cohort. This tests whole-model capacity,
not isolated CNN capacity;12M forecast quality remains untested. The other-game
follow-up is separate, below.

Use12M as the next quality candidate and1M as a fast engineering control.
The [projection-reuse trial](results/2026-10-07-cdp-exploration-throughput.md)
fails fixed numerical gates before timing and is not adopted. Original graphs
are restored; no tolerance relaxation or optimization sweep. The
[latest-backend control refresh](results/2026-10-07-meganeura-control-refresh.md)
passes20 guards/2,824 upstream comparisons and three train/frozen smokes.
Paired12M updates107.85->108.00ms, with exact first outputs,272 metric reports
and346 final tensors in the fixture. Adopt upstreamf104/native83be without a
speedup claim; this does not erase the stronger full-head failure.

**The five-game12M capacity comparison is complete:**
[Freeway result, combined comparison and all video links](results/2026-10-09-cdp-freeway-capacity.md).
The [four-game allocation](experiments/2026-10-07-cdp-12m-four-game-capacity.md)
finishes October9 at16:08 UTC in47h1m32s; its automatic CPU review finishes
16:12 UTC. With retained Breakout, fifteen500k learners cost7.5M actions /
1,874,085 updates /58h22m14s training,3.56x retained1M. All45 guards,
720 natural frozen episodes, exact saved tensors and30 whole replays/videos
pass. Every tail,35 excess episodes,52 allocation warnings and earlier
reporting/repair failures remain. No worker or automatic extension remains.

Capacity raises every paired reward mean on Breakout/Pong/Qbert, but Qbert's
first-pyramid successes still come only from3019. Boxing's gain interval
includes zero and its win rate was already72/72. Freeway regresses on average:
27.54 versus30.26, with crossing gates72/72 ->48/72. Keep the weak2017 seed,
not just1009's33.58. The game table below retains every result. This is not
universal capacity superiority, an exact same-binary ablation or tested12M
prior-forecast quality. Historical model/backend identities remain unchanged.

At approximately2M frames, the completed12M online means are ahead of released
DreamerV3 bins on Pong/Freeway/Breakout and close on Boxing/Qbert. Different
protocols/windows prevent a parity claim; all released seeds, fixed early bins
and the original last10% of200M targets remain in the
[combined data](results/2026-10-09-cdp-five-game-capacity.json).
All five quality targets and the RGB compute-saving question remain open.

**October9 backend qualified:** [refresh](results/2026-10-09-meganeura-refresh.md),
Meganeura31026833/Blade56f0565/native7311547d. All20 guards,2,824 upstream
comparisons and three train/frozen smokes pass. One paired12M timing measures
107.97->108.23ms, with exact first outputs,272 metric reports and346 final
tensors in this fixture. Adopt the fixes without a speedup or learning-cause
claim. Three allocation warnings and the corrected CPU provenance failure
remain; historical learning identities are unchanged.

**T64 does not advance:** [preflight](results/2026-10-09-cdp-sequence-preflight.md)
passes validity/frozen checks but saves only4.6% per replay position, below its
10% gate, with2.05GiB minimum sampled headroom. Keep T16; no sequence sweep.
The [one-shot cost capture](results/2026-10-09-cdp-imagination-cost.md) attributes
38.7% of instrumented imagination time to the four all-action ensemble first
projections. This is neither SM utilization nor a reason to retry the rejected
factorization or relax numerical gates.

**Qbert budget study complete:** [three fresh2M-action learners](results/2026-10-11-cdp-qbert-budget.md)
finish October11 at16:10:24 UTC on unchanged12M/T16. Frozen mean7,367.01
versus retained500k2,657.99, paired+4,709.03 [3,382.29,6,260.42]; all three
scores improve. First-pyramid completions0/0/24 ->20/22/20: more experience
helps the weak seeds, but3019 loses four successes. No seed meets the full
historical or long-run gate. Online6,545.50 is below the fixed published
8M-frame bin14,607.50, with protocol/model/window differences retained.

The cohort costs46h40m1s training,4.002x the retained500k cost. All nine guards,
144 natural frozen episodes, exact tensors and six whole replays/videos pass;
221,648 frozen actions, nine excess episodes, all tails and ten warnings remain.
Final CPU review passes in52.02s; no worker remains. Historical controls retain
their backend pins; this is not an exact same-binary budget ablation.
The [completed frozen forecast comparison](results/2026-10-11-cdp-qbert-forecasts.md)
finds useful h15 reward predictions in every2M seed: MAE/RMSE beat zero, mean,
persistence and unrelated actions; positive-event AUC.951/.967/.973. Later-stage
forecasting improves strongly over500k; survival remains unresolved with sparse
terminal evidence. Six guards/exact346 tensors pass,386,496 replayed actions,
zero updates,24m16.84s; original lifecycle failure/source and extra actions
remain. This is not universal world-model quality or a new competence result.
The [six-evaluation action-mode check](results/2026-10-11-cdp-qbert-action-mode.md)
is complete: greedy raises first-pyramid completions62/72→69/72, but mean
score falls7,367.01→6,953.47; paired−413.54 [−2,092.71,580.21]. All six guards,
144 natural episodes, exact tensors and whole videos pass;218,936 actions,
zero updates,3m32.63s total. Keep sampled evaluation as standard.
The [live-value check](results/2026-10-11-cdp-qbert-values.md) now compares
exact batched beliefs with full real discounted returns, including intrinsic
reward. Every critic beats zero/calibration mean; R².435/.927/.531. Bias differs
by seed/stage, not a shared collapse; the measured intrinsic fraction is<.03%.
Five guards/exact346 tensors/72 natural episodes/full traces and videos pass;
193,248 actions plus1,024 smoke, zero updates,88.87s capture. CPU review repairs
only wall-clock comparison; no GPU retry. No worker remains.
The [bounded measured-build check](results/2026-10-11-cdp-measured-build.md)
now completes. Adopt qualified ordinary Meganeura2c42b448/Blade56f0565/native
0f26ee63:21 guards,2,824 upstream comparisons, three train/frozen smokes and
exact first outputs/272 metric reports/346 final tensors pass. Updates remain
107.93→108.30ms; no speedup. Search's original logical-memory preflight fails;
one declared correction stops at value/bonus numerical tolerances. Both failed
guards/source bundles remain, with zero search optimizer updates/game actions;
ten total warnings and no GPU recovery. The test-only hook is removed. No
further search, tolerance relaxation or rejected-factorization retry.
**Pong2M budget study is stopped for the pivot:** the
[fixed declaration](experiments/2026-10-11-cdp-pong-budget.md) remains unchanged,
but its unstarted work is cancelled. Seed1009 runs October11 18:56:09–20:25:19
UTC and saves a partial checkpoint at189,576 actions/47,333 updates, zero debt.
No frozen evaluation runs;2017/3019 never start. The [stop receipt](results/2026-10-11-cdp-pong-stop.json)
retains counters, checkpoint identity and interruption limits. The service has
no workers or automatic restart. Its interrupted guard has no terminal verdict;
no new allocation warnings are logged. Keep the20.45 historical target and
completed500k controls; partial online progress is not a new competence result.
Do not automatically extend all five games, resume an old lifetime or start a
capacity/representation sweep. All five long-run targets and the RGB budget
claim stay open.

[Seed1009's500k prefix](results/2026-10-09-cdp-qbert-prefix.md) now matches the
retained run exactly: initial346 tensors,62,500 batched transitions,124,939
non-timing learner reports and671 episodes, online last50 mean1,809. The
CPU-only review does not establish universal backend parity, pixel equality,
final500k tensor equality or later/frozen competence. The completed2M evidence
above is separate from that earlier prefix check.

The exploration mechanism is a small GPU-native action-conditioned ensemble.
It predicts detached CNN embeddings and subtracts each head's all-action mean
before disagreement, removing action-independent uncertainty. One actor uses
the bonus in imagination and replay-value targets; the world reward head stays
extrinsic-only. No second encoder, scripted action, game label, CPU feature
readback or external reward shaping enters learning. This is an adaptation
inspired by [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html), not
a reproduction of its separate exploration actor.

Freeway's zero-reward blocker is resolved across all three seeds without action
aids. [Its full200k result](results/2026-10-05-freeway-effects-200k.md) preserves
the stricter mastery limitation: only3019 passes. The earlier
[all-zero baseline](results/2026-10-05-cdp-freeway-exploration.md),
[soft/visual-target failures](results/2026-10-05-freeway-embedding-learning.md)
and [weak32k frozen result](results/2026-10-05-freeway-action-effects-frozen.md)
remain additional development compute. The200k result is not a matched
extrinsic-only ablation, and Freeway was used to develop the mechanism.

The stopped45-run CDP/RGB/Tiny comparison and cancelled historical12M matrix
remain stopped, with all completed/interrupted evidence retained. Venture,
Tiny comparisons and new representation matrices are deferred by the current
five-game goal. After breadth is established, a separately declared held-out
exploration task with fresh extrinsic controls can test generalization of the
bonus. Do not repeat Seaquest or tune old mastery gates merely to fill a queue.

### 2. Video priors for dynamics and behavior (Phase 4)

Build on the online CDP agent, rather than returning by default to a large frozen
frontend. First test a single world/dynamics initialization against fresh CDP at
equal online experience; later consider inferred actions or a behavior prior.
Action-free video and action-labelled recordings are different supervision.

Use whole-recording train/validation/test splits. Disclose corpus, offline
compute and same-title exposure; missing actions or rewards are not NOOP/zero.
World-only pretraining must not silently update actor/critic. Success is fewer
online interactions or less total time to useful retained gameplay learning,
not only a lower latent error. Confirm a promising method on a second game
before scaling it. Causal Tiny must earn any return through this comparison.

### 3. A useful native-game actor (Phase 5)

Use `/x/Code/mind-games`: **vkQuake2 first**, vkQuake only as the transport
reference, then **TMNF**. Recheck the current integration API before changes.
Complete the reward/terminal/action adapter and GameSession integration; measure
kills/objectives or track finishes, not just movement.

The target is game GPU -> GPU capture/preprocessing -> encoder/RSSM/policy ->
CPU input events. Qualify the current v4 real-producer ownership/reuse path
before relying on it; the old v3 vkQuake plumbing success is not v4 validation.
Sparse explicit game rewards and disclosed human guidance remain acceptable.

Start with one learner and batched/serialized acting. For free-running games
without time control, measure p50/p95/p99 capture-to-action latency, observation
gaps, actual action durations, effective replay ratio and learner debt.
Introduce an asynchronous actor/learner **only if these measurements require it**;
it is not a prerequisite for the next Atari experiment or first native-game
learning check.

### 4. GOG games, then held-out transfer and retention

Apply the same actor to a small predeclared GOG/Wine panel with explicit sparse
rewards. Menus, startup scripts and input overrides are assistance and must be
reported. Look for repeated learning success rather than one favorable clip.

Reserve an unseen shooter before tuning a general FPS actor. Compare a fresh
agent, transferred world with fresh policy, and transferred world plus policy
at matched adaptation budgets. Declare action mappings and optimizer,
normalizer, replay and belief resets. Report zero-shot behavior, adaptation
speed and forgetting on source games. A held-out map is not a held-out title.

### 5. Swarm learning, last

Only after strong single-actor native/GOG learning and measured transfer/
retention, test immutable experience sharing or several actors feeding one
learner. Keep independent causal histories and explicit data/weight ownership.
Do not build swarm infrastructure or a concurrent learner service now.

## Current game status

These are current sticky/full-action centered CDP500k results, except Qbert's
separately allocated2M follow-up. All five titles retain12M/1M500k controls;
the qualified backend refresh prevents an exact same-binary capacity ablation.
This is not a comparison against raw CDP or RGB. Frozen means cover three learner seeds and24 natural
episodes/model; streams/episodes are not independent learner replicates.
The [combined capacity report](results/2026-10-09-cdp-freeway-capacity.md#complete-five-game-conclusion)
links every game's trained/initial-control videos and score/action/time curves.
Links into `runs/` require this workspace.

| Game | Current measured result | Unresolved question / historical gate | Evidence and whole videos |
| --- | --- | --- | --- |
| Seaquest | Three extrinsic CDP/RGB pairs: online543.6 vs318.1; CDP13.6% less wall time and35.9% less world-training time. Frozen world probes complete. | Controlled result on one title, not frozen policy mastery or five-game speed superiority. | [Learning and world report](results/2026-10-04-cdp-learning.md) |
| Boxing |12M frozen85.26 vs1.96 initial; seeds73.17/92.33/90.29,72/72 wins. Paired gain over1M+7.57 [−1.29,20.71]; training costs3.57x. | Both sizes pass historical gates in every seed; larger-model advantage is inconsistent. Published99.61 remains unmet. | [Capacity results and videos](results/2026-10-09-cdp-boxing-capacity.md) |
| Pong |12M frozen+12.40 vs−20.21 initial; seeds+7.21/+12.96/+17.04,69/72 wins. All beat retained1M controls at500k actions; training costs3.57x. | Only3019 passes historical>=90% wins/mean>=15. Published20.45 remains unmet;12M forecasts untested. | [Capacity results and videos](results/2026-10-08-cdp-pong-capacity.md); [earlier1M paired world report](results/2026-10-06-cdp-centered-world.md) |
| Freeway |12M frozen27.54 vs0; seeds33.58/17.25/31.79. Retained1M30.26; paired−2.72 [−13.29,4.08], training costs3.57x. | Crossing gates fall72/72 ->48/72;2017 regresses. No universal capacity advantage. Published33.40 remains unmet by the cohort. | [Capacity results and videos](results/2026-10-09-cdp-freeway-capacity.md); [exploration evidence](results/2026-10-05-freeway-effects-200k.md) |
| Breakout |12M frozen38.21 vs1.56 initial; seeds48.17/15.88/50.58. All beat retained1M4.00/3.50/7.92 at500k actions; training costs3.58x. | Published381.81 and historical two-wall/864-point gate remain unmet. Capacity helps;12M forecasts remain untested. Earlier weak-state diagnosis applies to1M, not automatically to12M. | [Capacity results and videos](results/2026-10-07-cdp-capacity.md); [1M world diagnosis](results/2026-10-07-cdp-breakout-world.md) |
| Qbert | At2M actions,12M frozen7,367.01 vs200.69 initial; seeds5,172.92/5,812.50/11,115.63. All beat retained500k scores; first pyramids20/22/20 versus0/0/24, training costs4.002x. | Published193,220.77 and historical>=90% first pyramids/mean>=15,000 remain unmet.3019 loses four first-pyramid successes despite its higher score. Useful h15 rewards and live values in all seeds, with calibration errors; survival remains unresolved. | [Budget results and videos](results/2026-10-11-cdp-qbert-budget.md); [forecasts](results/2026-10-11-cdp-qbert-forecasts.md); [values](results/2026-10-11-cdp-qbert-values.md); [retained500k capacity](results/2026-10-08-cdp-qbert-capacity.md) |

Historical Tiny results used nonsticky protocols and250k same-title video
observations (45k train+5k validation/game). Freeway also used a.5-probability
random action held for64 steps in training. Preserve them as historical, not
fresh/unassisted CDP or matched controls. Their gates are unchanged for interpreting
old claims, not perpetual development exit gates. Achievements before a cutoff
remain achievements without relabeling the episode natural; retain all tails.

## Runtime, speed and world-model checks

Eight small-recipe environments share batched perception/policy and one learner;
their recurrent states, resets, RNG and replay histories remain independent.
Uncapped, step-driven playing plus training already works. The completed five-game
CDP+exploration recipe on Meganeura592a2f5a achieves about8x aggregate /1x per-stream realtime.
Mean update31.62ms includes16.19ms imagination (51%),8.45ms world training (27%)
and4.33ms posterior work (14%). The older extrinsic-only CDP Seaquest comparison
used28.91ms updates; keep its settings distinct. GPU utilization is unmeasured,
not inferred from frame-clock ratios. Target measured whole-agent bottlenecks,
especially imagination, not just the now-cheaper world loss.
The October6 matched Pong comparison on Meganeurab684ffd9 averages32.11ms
raw /32.13ms centered updates and about28min per200k learner. Centering has no
material observed cost increase; it is not itself a throughput optimization.
The subsequent [grouped B128 trial](results/2026-10-06-cdp-imagination-throughput.md)
on Meganeurac6376542 reduces matched synthetic network updates30.49->28.01ms.
The completed1M five-game allocation measures29.70ms/update and127 actions/s,
8.46x aggregate /1.06x per-stream realtime. The12M capacity follow-up measures
110.03ms/update,55.4% imagination/29.9% world training/9.2% posterior, about
2.38x aggregate /.30x per-stream realtime. This is a capacity/cost tradeoff,
not an unchanged-learning slowdown or measured GPU utilization. The shared
exploration state-projection trial was rejected at its numerical gates before
timing; the qualified latest-backend control retains the original graph.

Atari emulation/frame upload is a CPU-environment fallback; GPU preprocessing,
batched acting and resident replay collection are implemented. Native GPU
capture integration exists, but current v4 producer qualification remains open.
The [capture report](results/2026-10-02-matching-external-allocations.md) keeps
the allocation warning and historical v3 success separate. Only selected actions
leave the acting hot path; rewards, diagnostics, checkpoints and sampled learner
data may still cross the host. Some scalar targets/slow-critic work remains
host-side. Do not claim a completely host-free learner.

CDP uses one GPU resize from native Atari frames to RGB 64, not a downscale/
upscale detour. A future encoder must receive its intended native-detail input;
do not silently feed RGB 64-upscaled frames into LeVJEPA. Tiny's16-arrival chunk
reset is perception-only, not an environment or RSSM reset.

World-model checks accompany meaningful learning milestones, without becoming
a new qualification campaign. Forecast before observing targets; test held-out
real trajectories at short and longer horizons against persistence, constant
means, unrelated actions and zero-reward/always-continue controls. Report
representation spread, fitted-state readability, reward/terminal counts,
MAE/MSE and matched cohorts. Privileged observers are evaluation labels only.
Posterior estimates, predictable features and imagined RGB are different things.
Existing CDP probes and [older world reports](experiments/README.md) retain
negative results; do not select only successful trajectories.

## Execution discipline and evidence

- Screen small with one changed mechanism and at least three learner seeds.
  Report return versus actual interactions and wall time, seed-bootstrap
  uncertainty, configuration, every aid and all episodes/tails. A lower replay
  ratio, new loss or smaller model is a learning tradeoff, not an unchanged-
  learning speedup. Increase capacity or budget for evidence of a limitation,
  not automatically to12M.
- Frozen evaluation never updates model/optimizer tensors. Predeclare evaluation
  cohorts and report untrained controls when claiming competence. Development
  curves, numerical smokes and rollout videos alone are not mastery.
- Use the GPU and keep Meganeura/Blade current before diagnosing old bugs.
  The new synthetic-performance stack is Meganeurad68f1a5/Blade663afd2/Naga
  forkc9fd6a85a; see the [combined qualification](results/2026-10-12-backend-integration.md).
  It is not new game-learning evidence. Historical Qbert production pins are
  Meganeura31026833/Blade56f0565; the
  [October9 control refresh](results/2026-10-09-meganeura-refresh.md)
  passes scoped independent and production/restore checks, with no measured
  speedup or new learning comparison. Nativee9ebbf5d adds only the qualified
  read-only value probe; learning evidence retains7311547d. Tiny retains its
  historicalb684ffd9 qualification.
  Retain original backend identities for old results.
  No repeated upstream learning replication without a relevant change.
- Serialize bounded native jobs under the [host guard](gpu_incident_response.md)
  in persistent systemd user services. Require the expected device and >=2 GiB
  sampled Vulkan budget headroom; this is not physical free or peak VRAM.
  Ordinary GPU/JAX initialization is allowed; separate NVML polling stays off.
  October4 permission allows GPU use unless wedged: record standalone allocation
  warnings without an approval gate. Actual API/numerical failures, hard faults
  and deadlines still stop the affected job for review. No blind retry, reset,
  driver change or host recovery.
- Preserve failures and compact JSON/Markdown reports; leave large artifacts in
  `runs/`. Keep current status in the PR, not in chronological roadmap appendices.
  Do not rebuild unchanged native code for docs or create a new pin/framework
  layer per experiment. Only the user merges.
- Checkpoints preserve weights/moments, not a complete replay/RNG/live-belief
  lifetime. Interrupted restore is not equivalent to uninterrupted learning.
  Natural game deaths/respawns are allowed; cloning/rewinding a live game for
  training is not. Full-lifetime recovery remains later work.

The initial CDP roadmap adoption was documentation-only. The user's subsequent
five-game comparison goal authorizes the separately declared training/evaluation.
CLI defaults remain unchanged; old artifacts retain their original scope.

Core implementation: [agent](../kindle/src/dreamer/agent.rs),
[vector collection](../kindle/src/dreamer/agent/vector.rs),
[world model](../kindle/src/dreamer/world.rs),
[networks](../kindle/src/dreamer/networks.rs),
[behavior](../kindle/src/dreamer/behavior.rs),
[replay](../kindle/src/dreamer/replay.rs).
