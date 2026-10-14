# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **Current derivative algorithms stay slow through CUDA:**
  [comparison, counters and matrix2 readiness](docs/results/2026-10-14-convolution-backend.md).
  Same B512/CI32/CO48/K5 checked dX/dW: native2.148/2.019ms,CUDA1.872/2.441,
  retained cuDNN.479/.550. All16.82M outputs/backend and512 F64 checks pass;
  both10% timing bands fail (−12.86/+20.92%), so CUDA counters are not native
  counters or a calibrated time model. Current incremental dW/split/compensated
  reduction and twelve algorithm sources match; no native rebuild. CUDA captures
  show1.007/1.105B warp instructions,58.15/57.65% integer,~9% tensor activity,
  no reported spilling. Retain clock/cache/optional-sampling warnings; no native
  occupancy/utilization claim or renewed CUDA/geometry sweep. Five guards pass,
  one allocation warning;292 native/330 CUDA calls plus two17-pass captures,
  zero learner updates/actions/workers. CI411 passesb2390c3; upstream unchanged.
  **5080/580 exposes cooperative-matrix2:** workgroup scope, flexible dimensions,
  tensor addressing and BF16→F32;128 threads advertise32/16/16 M/N/K multiples.
  Current Naga hard-codes subgroup scope; Blade does not expose matrix2. Exact
  capability/source records remain. Query creates only an instance, no logical
  device/allocation/pipeline/dispatch; support is not numerical/performance
  qualification. Next one bounded native compiler-managed matrix-load/staging
  proof before Naga/Blade/Meganeura feature work; shared typed path, no tile/split
  sweep or automatic adoption. This offers a concrete higher-level alternative
  to manual staging, not proof of efficient convolution mapping or parity.
  Keep isolated28864210 and its [general ordering gain](docs/results/2026-10-14-convolution-scheduling.md):
  63.597→60.041ms,5.59%, exact272 reports/initial-first-final292 tensors,20
  automatic boundaries with unchanged allocation/submissions. Execution-only
  ordering retains the local gain; hardware/cache causality remains open. No
  blanket serialization/additive-savings forecast or revival of failed sparse/F32
  controls. Full B32 remains2.25× JAX26.632ms, not remeasured this round.
  Production/Atari, root-cause and five-game goals stay open; host targets distinct.
- **Wide-image JAX control complete:** [result](docs/results/2026-10-14-image-precision-control.md).
  Complete RGB12M/B32 original/wide26.632→26.876ms,0.92%; retained native
  65.736ms is still2.45× wide JAX. F32 image bias/pool/norm/activation does not
  explain the multiplier. Heavy multiplication shapes/dtypes/MACs unchanged;
  eight convolution bias operands move out, so not a storage-only/exact-native
  control.288 initial tensors exact,272 finite reports/arm;283 final tensors
  differ. Eight independent image-tail cases/22,120 F64 values/derivatives pass.
  Three guards pass/one allocation warning; two CPU process failures and two
  CPU report failures retained.544 synthetic updates, zero actions/workers/
  native change. Next implement connected image forward/backward lowering on
  existing precision, including decoder blocks: remove remaining norm/activation/
  layout intermediates, share backward state/reductions, preserve pooling
  derivatives/staged lifetimes. One shared typed path, independent numerical
  checks and complete-update gate. No dtype/layout/row/tile sweep, miniature-JAX
  retry, broad capture, Atari restart or goal closure. Keep3551bc53; convolution
  staging/algorithms and host decode/targets remain separate measured gaps.
- **Packed shape specialization rejected:** [result](docs/results/2026-10-14-packed-specialization.md).
  Complete RGB12M/B32 control/candidate65.684→65.207ms,0.726%, below3%;2.40×
  retained JAX27.187ms. Keep3551bc53, not rejectedf8e6e663/212713c4. All14
  packed-convolution keys change, five plans/cache counts/272 reports/first
  outputs/three292-tensor boundaries stay exact.36 specialized/24 retained
  shader variants and40 GPU cases/13,304,541 F64 values/seven partials pass.
  Initial guard fails in CPU12→16-word parameter binding; fixed via existing
  binding conversion, unchanged oracle/gates. Three guards pass/one warning,
  544 synthetic updates/40 zero-rate numerical steps/zero actions/workers.
  Earlier dispatch-route CPU preparation stop retained. Upstream99e3de27/
  e7c3f7b unchanged. Both this and layout reuse have sub1% whole gains; no
  shape/layout/tile sweep or promotion. Next one connected image-training
  reduced-storage/fusion contract: actual JAX pooled values32MiB BF16 versus
  native64MiB F32, plus different bias/rounding/statistics boundaries. Static
  extents, not measured savings; JAX still retains a pooling mask. The earlier
  wider-JAX control left these islands unchanged. Define rounding/pooling
  subgradients and independent quantized forward/backward checks; deliberate
  precision changes need not preserve legacy trajectories. No dtype sweep,
  miniature-JAX retry, Atari restart or whole-goal closure.
- **Connected layout reuse rejected:** [result](docs/results/2026-10-14-connected-layout.md).
  Complete RGB12M/B32 control/candidate65.651→65.593ms,0.089%, below3%; still
  2.41× retained JAX27.187ms. Keep3551bc53, not rejected212713c4. World4,089→
  4,069 dispatches removes eight packs/ten casts/two transposes; all heavy
  descriptors/protected metadata,272 reports/first outputs/three292-tensor
  boundaries/cache counts exact. Other three plans/keys exact.162 typed-pack
  cases/279,936 exact values and12 connected pairs/14,877,300 exact outputs/
  6,912 F64 probes pass, including staged lifetimes/aliasing/changed inputs.
  Three guards pass/one CPU fixture shape failure retained; only unstarted
  connected cases run after reviewed test-only repair. Two CPU build failures
  retained; zero warnings/actions/workers,544 synthetic updates. No promotion,
  layout sweep or gap closure. Next isolate packed-convolution shape/index
  specialization: its fixed dimensions remain uniforms/software division while
  scalar convolution already specializes them. Hypothesis, not established
  causality; retain shared typed arithmetic and a complete-update gate.
- **Connected image lowering qualifies, most of the gap remains:**
  [result and limits](docs/results/2026-10-14-spatial-tape.md). Fresh complete
  RGB12M/B32 control/candidate68.491→65.736ms,4.02%; still2.42× retained JAX
  27.187ms. Only6.67% of that gap closes. Keep the isolated candidate3551bc53,
  not a solved architecture or production promotion. B8 is not newly measured.
  One shared typed pool lowering folds bias, saves original forward decisions,
  and emits rounded gradient/F32 spatial partials together. Same precision,
  reduction tree, model and tensor products. Four regions replace20 dispatches
  with8;4,081 others retain descriptors. Posterior/world plan labels alias the
  same staged session, not duplicate work. All272 reports/first outputs/three
  292-tensor boundaries/cache counts exact; other three plans/keys exact.
  First complete block6.218→4.336ms versus retained timing; all large outputs,
  5,104 F64 values and staged restores pass.24 generic cases/25,632 reference
  values/15 shader variants qualify; F16 is shader-only coverage.686MiB fewer
  logical scheduled output extents, not DRAM traffic or peak memory. Four guards,
  zero warnings/GPU failures,544 synthetic updates/248 block calls/zero actions
  or workers. Seven CPU preparation/report failures and corrected sources remain.
  No pool/row/grid/tile/dtype sweep. The following staging intervention closes
  below its gate; next preserve physical layout/reuse across connected image
  training, including padding/rounding/consumers and staged backward lifetimes.
  [Prior whole-program attribution](docs/results/2026-10-14-jax-image-profile.md)
  retains56.948ms GPU-pass union/11.688ms outside and5.383ms host decode/targets;
  actual JAX23.099ms kernel/copy union uses different coverage. No exact causal
  subtraction, SM-utilization or additive-savings claim. Host targets remain
  a separate measured GPU-residency opportunity. No broad capture, automatic
  NHWC rewrite, production repin, Atari restart or full-goal closure. Earlier
  miniature JAX failure stays unqualified/no retry. Upstream99e3de2/e7c3f7b
  rechecked; no identified fix for this path. CI403 passes on46e1463.
- **Convolution spatial-vector staging rejected:** [result](docs/results/2026-10-14-convolution-spatial-staging.md).
  Fresh decoder forward/dW sum2.03269→2.04531ms,0.62% slower; encoder dW also
  regresses. All70,264,192 retained outputs/five plans exact; no whole-update
  trial, promotion or vector-width sweep. Candidate680ea39a stays rejected.
  CPU303 shader variants/2,152,448 component-address checks pass.200 GPU cases/
  600 calls finish their individual checks, then fail my final count assertion:
  93,280, not100,960. Keep the failed guard/source; independent CPU F64/exact/
  change reconstruction passes all saved values at3.36e-7 maximum relative L2.
  No GPU retry, numerical/API fault or tolerance change. Two timing guards pass;
  three preparation failures remain (Clippy/BF16 literal/stale build cache).
  2,060 kernel calls/zero learner updates/actions/warnings/workers. Production
  unchanged; retained whole B32 remains65.736/27.187ms,2.42×. CPU audit of the
  qualified4,089-dispatch world plan finds ten transpose→BF16→pack chains and
  eight duplicate packs/80MiB repeated logical output extents. Not traffic,
  peak memory, measured savings or the whole38.55ms explanation. Next one shared
  typed connected-layout/reuse intervention with a complete-update gate, not
  a duplicate-count-only sweep or automatic global NHWC rewrite. Goal stays open.
- **Packing cliff is not the dominant gap:** [qualified intervention](docs/results/2026-10-14-packing-grid.md)
  changes exactly four B32 convolutions but saves only1.49%,68.344→67.326ms,
  below3%; no promotion/sweep. Candidate remains2.48× retained complete JAX
  27.187ms. The [batch comparison](docs/results/2026-10-14-batch-granularity.md)
  retains B8 native/JAX22.780/12.737ms and1.33×/1.87× throughput scaling with
  exact4× nominal work. The selector's65,535 X bound is conservative; actual
  5080 limits[2147483647,65535,65535] admit all retained grids. No invalid-launch
  claim. Shared typed2D packing preserves memory/precision/accumulation bounds.
  All536,894,080 exact packing values/40 F64 derivative cases/seven partials
  pass; B8 smoke and B32 control reproduce exact reports/plans/states. B32
  candidate has exact initials/finite states but changed arithmetic/trajectory.
  Seven guards/two warnings,592 synthetic updates/zero actions/workers. Four
  CPU failures remain; repaired graph and complete-plan audits pass, no GPU
  retry. Native9ad4b2b6 stays unpromoted; e5d890dd remains B32 control. Next
  cached B32 family attribution is complete above; follow actual connected
  image operations against reachable JAX—not another row-only/
  packing/tile/dtype sweep. No production change, Atari restart or full-goal
  closure. Upstream7574ad1/e7c3f7b unchanged; CI399 passes on238a0d4.
- **Current cached execution attributed:** [result](docs/results/2026-10-14-cached-residual.md).
  Profiling previously disabled the command cache; isolated primary-boundary
  timestamps retain it. Ordinary/coarse23.039/23.154ms,0.50% overhead, all272
  reports/plans/pipelines/first outputs/three292-tensor boundaries exact.
  GPU-pass union19.697ms, uncovered3.458ms; not SM utilization. Fixed graph
  18.733ms allocates4.295 empty stream/5.587 convolution/3.388 matrix/5.463 other.
  All1,536 measured masks verify cache reuse; full outputs/tensors restore.
  Four guards/one retained warning,569 synthetic updates/2,380 optimizer-free
  replays/zero actions/failures/workers. Nativec3b7975b, production unchanged.
  One-world-chunk diagnostics are not a new four-chunk speed result: retained
  22.881ms/fresh JAX12.684ms remains1.80×. Next actual connected heavy-projection/
  layout/rounding/backward regions against reachable JAX, then a general measured
  lowering intervention. No row-only/dtype/tile/barrier sweep, additive savings
  forecast, Atari restart or whole-goal closure. Upstream7574ad1/e7c3f7b unchanged.
  CI397 passes on preceding9044f84; publication/source/link audit passes.
- **Wider-intermediate JAX control completes below its follow-up gate:**
  [result](docs/results/2026-10-14-precision-flow.md). Complete12.684→13.469ms,
  +6.19%, below10%; no further precision capture/sweep. Actual obs0 contracts
  show native F32 versus JAX BF16 input gradients, but both accumulate F32
  master gradients; native forward partials narrow before bias/norm. Diagnostic
  F32 interfaces/matrix derivatives retain BF16 products/rounded forwards and
  unchanged BF16 convolution islands. Explicit algorithm changes compiler/
  library choices: not a storage-only ablation or exact native clone.4,809 F64/
  71 frontend/1,272 convolution values pass; all18 work-count differences
  reconcile padding/scalar lowering/packing exactly, no tolerance relaxation.
  Same288 initials;272 finite reports/arm, changed trajectories. Ten guards
  (four pass/six retained failures), zero allocation warnings,544 synthetic
  updates/66 fixture calls/zero actions/workers. Source/accounting audit passes.
  Retain cast-elision, grouped arithmetic, classifier and preparation failures.
  Native22.881ms unchanged,1.80× fresh JAX; no production/learning promotion.
  Upstream7574ad1/e7c3f7b unchanged; CI396 passes on4237eed. Next refresh the
  current candidate's low-perturbation attribution, then follow its largest
  connected heavy-kernel/compiler costs. No dtype/row-only/tile sweep or
  whole-goal closure. Device execution and kernel efficiency remain in scope.
- **Shared typed vector staging qualifies:** [result](docs/results/2026-10-14-matrix-staging.md).
  Fresh complete RGB24.022→22.881ms (4.75%), now1.78× retained complete
  JAX12.836ms. All272 reports/first outputs/five plans/initial-first-final292
  tensors exact; only eligible checked pipeline keys change. Same arithmetic,
  precision, tile/K partitions/dependencies; four-value physical-major shared
  staging, not a BF16-only engine.416 GPU cases/1,981,440 F64 comparisons,
  four actual contracts/5,345,280 baseline F64 values,243 shader variants/16
  selection cases/17,350,704 CPU address checks pass. Grouped forward19.27%,
  dense/grouped gradients24.43%/10.05% faster; small-row forward2.27% slower.
  Seven guards/one retained warning,592 synthetic updates/3,588 matrix calls,
  zero actions/workers. Initial CPU template collision/source remains.
  Native42f00066; keep isolated, no production/CDP/learning promotion.
  The [preceding counters](docs/results/2026-10-14-matrix-library.md) find equal
  tensor work/occupancy but3.40× instructions in a timing-matched CUDA translation;
  not native Vulkan counters or full-JAX algorithm parity. Both failed controls
  remain; no precision relaxation. Next trace actual matched forward/backward
  precision/materialization contracts, then separately measure that policy;
  native247 BF16/469 F32 versus JAX456/96 outputs are not one-to-one evidence.
  Keep device-execution/convolution/compiler costs in scope. No vector-width/
  tile sweep/Atari restart or whole-goal closure. CI395 passes on a219e08.
- **Checked small products dominate matrix cost:**
  [current attribution](docs/results/2026-10-14-matrix-cost.md) assigns521 checked
  products2.903ms/63.54% of matrix cost for12.01% of nominal matrix MACs;195
  aligned products1.666ms. All BF16;302 eight-row products,160 additional
  reduction-eight products. World recurrence supplies2.568ms checked cost.
  All plans/pipelines/reports/restored outputs/initial-before-after292 tensors
  exact. One guard/zero warnings, one synthetic update/1,468 optimizer-free
  replays/zero actions; two CPU classifier tests pass. Native1339b0b7, unchanged
  iterator backend/compiler. No new whole-update or logged-validation claim.
  Standard Vulkan executable query separately passes logged compile-only guard
  but exposes no representations; retain uninterpreted statistics, no native
  assembly/driver change. Two total guards/zero warnings, no worker. Next
  matched small-row/short-reduction dense/grouped JAX algorithms and contracts,
  then justified general algorithm or gradient aggregation—not a tile sweep.
  Full reachable JAX HLO confirms96 reduction-eight BF16-input/F32-output
  cuBLAS calls with `beta=1`; no claim they avoid our F32 accumulation.
  Complete RGB/JAX24.340/12.836ms remains1.90×; full goal open.
  CI393 passes on precedingaa498bc.
- **Precise dependency APIs do not close the device-execution gap:**
  [four-path check](docs/results/2026-10-14-dependency-paths.md) finds legacy/
  synchronization2-global/exact-buffer barriers all~2.05µs per dependent kernel;
  immediate event set/wait~4.10µs. All3,204 exact counters/two guards pass,
  zero warnings/actions/updates; four CPU command tests pass. Failed initial
  Rust build/source remains. No20% long-chain gate passes: close this API
  detour, no full-learner integration/flag sweep. Native765e6294, isolated
  Blade96c1dc7/Naga a97559b; production unchanged, no worker. No universal
  Vulkan limit or SM-utilization claim. Current whole RGB24.340ms/retained
  complete JAX12.836ms stays1.90×; root-cause goal open. Next remaining kernel
  staging/reuse and useful work per execution unit, preserving real dependencies.
  CI392 passes on precedingf2a8a3c.
- **Incremental checked-convolution addresses qualify:**
  [result](docs/results/2026-10-14-convolution-iterators.md). Same cached complete
  RGB25.475→24.340ms (4.45%), now1.90× retained complete JAX12.836ms.
  All272 reports/first outputs/plans/pipelines/initial-first-final292 tensors
  exact. Five boundary kernels3.747→2.501ms; decoder dW37.15%/forward28.43%
  faster, unchanged dX. Same arithmetic/precision/tiles/K partitions/barriers;
  shared typed cursors, not a BF16-specific engine.70,264,192 exact compared
  values/1,280 F64 probes,40 derivative cases/17 partials,32 typed/alias cases
  and288,675 CPU address checks pass. F32 storage qualifies, not native F32
  cooperative multiplication. Seven guards/zero warnings;592 synthetic updates,
  1,460 boundary calls/zero actions. No worker/production/CDP/learning promotion.
  Nativeb9152764, numerical5e178d83. Keep isolated candidate; next distinguish
  remaining staging/reuse and device-execution costs, not an iterator/tile
  sweep. Full gap stays open. CI391 passes on preceding7f28899.
- **Checked convolution's dominant cost is address-generation work:**
  [matched operations and Nsight counters](docs/results/2026-10-14-boundary-algorithms.md).
  Five native kernels total3.748ms; encoder forward/dX are competitive, while
  decoder forward/dW1.310/1.742ms versus cuDNN.261/.338ms dominate. Dominant
  dW matches the full JAX algorithm; two other miniature algorithms differ.
  Its CUDA translation executes1.785B warp instructions/1.306B integer versus
  cuDNN100.06M/61.09M. cuDNN executes12.5% more identical BF16 HMMA operations,
  with4.65->39.71% tensor activity and61.93->15.99% occupancy. Padding/occupancy
  alone are not the cause. Bridge2.374ms is36.2% slower than native; counters
  are not Vulkan instructions. All51,914,112 JAX values match at declared
  rounding;1,664 F64 probes/640 coordinates and changed-input checks pass.
  F32 JAX gradient-output controls widen operands: no storage-policy isolation.
  Five guards pass; first fixture's BF16-input assertion fails before dispatch,
  repaired via internal buffers with source/binary/failure retained. No actor
  updates/actions/production change/worker. The general incremental typed
  iterator intervention above now demonstrates the native causal benefit.
  No scalar revival, geometry sweep, dtype-specific kernel or precision
  relaxation. The checked footprint does not explain the whole gap. CI390 passes.
- **Command reuse removes a structural host bottleneck:**
  [result](docs/results/2026-10-14-command-cache.md). Same RGB complete updates
  28.416->25.530ms (10.16%), exact272 reports/first outputs/plans/pipelines/
  initial-first-final292 tensors. Nsight CPU-API capture shows descriptor
  updates5351->65 and pipeline binds5297->11;15 submissions/update unchanged.
  All5286 GPU graph dispatches still execute. Eight graph chunks recorded once,
  2168 reuses; dynamic optimizer commands remain live and barriers stay intact.
  Selected CPU API1.562->.116ms is not all host cost or GPU acceleration;
  capture perturbs control+7.85%, so use ordinary timing.16 lifecycle cases/
  278528 exact values pass twice, second with repaired validation logging;
  eight guards/four warnings,1152 synthetic updates/zero actions. CPU failures
  remain. Keep isolated candidate, not production/portable/CDP qualification.
  No worker/recovery; RGB still1.99x retained complete JAX. Next investigate
  actual remaining library algorithms/checked boundary convolutions and device
  dependencies, not another cache/geometry sweep or Atari restart. CI389 passes.
- **Narrow-product alternative rejected:**
  [result](docs/results/2026-10-14-narrow-products.md). A shared typed scalar
  alternative removes padded rows in two checked decoder kernels, but complete
  RGB28.488->30.492ms is7.03% slower. All90 numerical cases/4,705,658 values,
  90 shader variants/eight selection checks/45 existing shader tests pass.
  Five guards/three warnings,592 synthetic updates/90 numerical executions,
  zero actions; first/final trajectories differ, no learning parity. No grouped
  GPU or F32 cooperative qualification. Reject without a geometry/CDP sweep;
  production unchanged, no worker. Command recording is addressed above;
  GPU algorithms and dependency cost remain open. CI388 passes on2429a48.
- **Irregular convolution paths remain expensive:**
  [result](docs/results/2026-10-14-heavy-issued-work.md). Five checked RGB-boundary
  kernels contain11.1% of convolution nominal MACs but56.6% of measured cost:
  3.785ms versus2.906ms for18 packed kernels. All already multiply in BF16.
  Exact source/grid accounting finds84.756G useful versus152.574G cooperative
  product slots (1.80x), including10.67x padding in two three-output kernels.
  Not hardware counters or a JAX issued-work ratio; padding is not the sole
  proven timing cause. One guard/zero warnings, exact plans/pipelines/outputs/
  292 tensors, one original update/367 optimizer-free replays, zero actions.
  CPU count-typo failure remains. The general typed alternative above fails
  the full-update speed gate; no further narrow-product/tile sweep, production change, Atari restart
  or whole-gap closure. Device dependency/host costs stay in scope.
- **Connected-region learner trial closes below its gate:**
  [result](docs/results/2026-10-14-connected-learner.md). Automatic BF16-safe
  forward/backward integration selects127 regions,5,286->4,908 dispatches,
  but full RGB28.280->27.999ms saves only.99%, below3%. No CDP follow-up or
  further row sweep/promotion. All56 cases/10,251,192 numerical values pass,
  staged late-dy/alias poison included; heavy descriptors/packing provenance
  match. Disabled control reproduces all272 reports/292 tensors; candidate
  later trajectories differ, final aggregate L2.08319, no learning parity.
  Nine guards/five warnings,640 synthetic updates/120 block replays including
  the retained partial implementation; failed builds/CPU audits remain.
  No worker or production change; whole2.20x gap remains open. CI387 passes.

- **Connected training-region prototype passes:** [result](docs/results/2026-10-14-connected-row-region.md).
  Existing general reduction gains column prologues and extra row/element
  outputs; RMSNorm/SiLU saves forward values and emits both derivatives.
  Same8x256 complete block12.841->8.280us (35.5%),6->3 dispatches/4->3 groups.
  All281,052 GPU values/12 alias-poison cases and70,263 CPU values pass;
  29 CPU tests/two guards, two allocation warnings,556 optimizer-free replays,
  zero actions/updates/workers. Failures/source snapshots remain. Explicit plan
  construction on main7574ad1, not automatic compiler or BF16 learner adoption.
  The20% block gate passes; subsequent whole-graph integration above preserves
  rounding/live outputs/staged backward lifetimes but saves only.99% overall.
  No block sweep, additive savings claim or Atari restart. Whole2.20x gap stays
  open; production unchanged. CI386 passes on preceding1e71e74.

- **Training compiler boundary diagnosed:** the [latest-main CPU control](docs/results/2026-10-14-training-residuals.md)
  proves JAX carries forward normalization/activation tapes into backward;
  native135 SiLU/norm derivative regions recompute them. Primitive expansion
  alone worsens the representative plan14->28/32 dispatches and7->16/17 groups.
  All198,150 F64 comparisons in six plans pass; initial132,100 retained.
  Recomposition/shared-output/broadcast boundaries need connected lowering,
  not a decomposition flag or more scalar tuning.141 descriptor-identical
  pure dispatches in retained learner plans are not a timing claim. No GPU
  allocation, production change or worker; full2.20× gap remains open.
  Meganeura main7574ad1's BF16/shared-widening fixes are present in this CPU
  fixture; exact upstream regression passes. Not full-main GPU qualification.
  CI385 passes. Keep heavy-kernel and device dependency costs in scope;
  row/reduction's whole2.329ms allocation cannot alone explain the gap.

- **Comparable math, different execution machinery:** the
  [reachable executable audit](docs/results/2026-10-14-executable-work.md)
  finds84.756G native versus83.589G JAX nominal heavy MACs, only1.40% extra,
  despite retained28.213/12.836ms (2.20×). All23 convolution shapes/counts and
  129 grouped products match; dense587 versus423 calls account for the delta.
  Actual JAX dataflow packs reward/continuation/value/slow-value projections
  into one wide GEMM; native typed products/sessions stay separate. This is
  not a measured large speedup or permission to revive rejected factorization.
  Count scalar/layout work separately; nominal MACs are not hardware instructions.
  CPU-only audit passes after a retained tuple-boundary proof failure; no GPU
  jobs/actions/updates. The [separate non-heavy attribution](docs/results/2026-10-14-materialization-cost.md)
  now allocates1.432ms to1,744 layout/cast dispatches,2.329ms to1,557 row/
  reductions and1.631ms to1,246 other operations. Full fixed GPU21.171ms;
  heavy work/unchanged commands15.778ms is not an optimized-program lower bound.
  All plans/pipeline keys/outputs/292 tensors match; one guard, zero warnings,
  one original update/2,380 optimizer-free replays, zero actions/workers.
  Row/backward regions with explicit layouts/rounding/multiple live outputs
  are a measured compiler priority, not another scalar cap or uncosted wide GEMM.
  Time stacking is already balanced. No automatic implementation allocation.
  Keep kernel efficiency, device dependency cost and host construction distinct;
  no small tuning sweep, additive savings forecast or production/Atari restart.
  Latest Bladee7c3f7b/Nagacbbac959 adds Metal BF16 only; no Vulkan/SPIR-V
  changes. Current attribution retains96c1dc7/a97559b. CI384 passes.

- **Compensation and existing prefetch are not the missing large cost:** the
  [bounded diagnostic](docs/results/2026-10-14-accumulation-cost.md) measures
  fixed-world16.169->16.025ms, only.89%, below its10% follow-up threshold.
  Plain dW retains14 F64 pointwise failures/layout; keep compensated arithmetic.
  No Nsight/accumulation sweep or alternative-gradient optimizer updates.
  Two guards/two warnings, one original preparatory update/443 optimizer-free
  replays, exact292 tensors and restored outputs/68 gradients, zero actions.
  Complete source snapshot repairs18 inherited omitted additions; failed CPU
  build/incomplete patch remain, historical measurements unchanged.
  The [separate prefetch check](docs/results/2026-10-14-conv-prefetch.md)
  completes: RGB28.436->28.714ms (.98% slower), all272 reports/first outputs/
  plans/292 tensors exact. Existing next-tile loads, not a multistage async-copy
  implementation. Independent derivatives/CPU shaders/Clippy pass; five guards,
  one warning,568 synthetic updates, zero actions. Close its conditional follow-up;
  no CDP/counter/tile sweep or promotion. Upstream2214412/96c1dc7 unchanged.
  No learning/production promotion; full gap remains open. CI383 passes.

- **Fresh residual attribution confirms a GPU execution gap:**
  [result](docs/results/2026-10-14-residual-reassessment.md).
  Isolated ordinary28.213ms versus complete retained JAX12.836ms remains2.20×.
  Single-chunk ordinary/coarse30.066/30.115ms (.16% trace cost) preserves all272
  reports/first outputs/plans/initial-first-final292 tensors. All272 updates
  include20 steady-state passes and async tails;45 scalar probes/three checkpoint
  copies are accounted separately. GPU-pass union22.144ms versus JAX10.295ms;
  RNG+copies+EMA only.430ms. Current masks allocate convolution6.888ms,
  matrices4.475ms, other5.545ms plus4.272ms empty command stream; not pure
  kernel times or additive savings. Four guards/one warning,569 synthetic updates/
  2,380 optimizer-free replays, zero actions/workers. CPU coverage assertion and
  unused first instrumentation build remain; no GPU retry. Production unchanged.
  No row/grid/binding sweep or invalid Vulkan Nsight/Graphics retry.
  The completed compensation check above rules out that large-cost hypothesis;
  no alternative gradients ever update an actor/optimizer. Keep gates/failures.
  The [actual-grid trial](docs/results/2026-10-14-dense-launch-grid.md)
  stays unpromoted (~1% gain, changed trajectories). Retainb3a086f/compiler
  96c1dc7/a97559b. No Atari allocation or whole-goal closure; CI382 passes.

- **Latest compiler does not materially close the5080 gap:**
  [result](docs/results/2026-10-14-storage-propagation.md). Blade96c1dc7/
  Naga a97559b changes storage propagation, not graphs: RGB28.197→28.314ms,
  CDP92.965→93.049ms, all272 reports/first outputs/plans/292/346 tensors exact.
  Six numerical groups pass; row-fusion's original F64 gate fails identically
  on both compilers. A separate unchanged-tolerance diagnostic explains three
  BF16 bin crossings; it is not an unconditional qualification pass. Keep both
  failed guards, CPU parser OOMs and build failure.13 guards pass/three warnings,
  1,136 full synthetic+144 small staged updates, zero actions/recovery/workers.
  Keep nativeb3a086f in isolated experiments; production remains unchanged.
  RGB2.21× retained complete JAX; no new JAX/utilization/learning claim. CI380
  passes. The separately completed actual-grid check above uses this refreshed
  compiler, not a combined compiler/geometry intervention. Broader goal stays active.

- **Row selection qualifies, but dependency counts overstate its benefit:**
  [result](docs/results/2026-10-14-row-selection.md). Fresh RGB28.777→28.479ms /
  CDP93.938→93.001ms: only1.04%/1.00%, despite groups2,271→2,058/2,483→2,253.
  Both24-update smokes and all272 full-run reports/first outputs/initial-first-
  final292/346 tensors are exact.32 CPU tests/52 shader cases,144 primitive
  cases/2,922,192 exact one-hot values/21 F64 derivatives and eight guards pass.
  Retain the original failed infinity-row guard: static(1,0,1) binding zeroed
  runtime-one/table fields; bounded diagnosis and preserving those fields fix
  it without gate relaxation. Five warnings,1,136 synthetic updates, zero
  actions/recovery/workers. One RGB backward addend moves into its pointwise
  consumer; failed strict inventory and reviewed exact exception remain.
  Candidate6d892a3d stays isolated; RGB2.22× retained complete JAX12.836ms,
  no fresh JAX or production/learning change. CI379 passes.
  **No row-operation tuning campaign.** A CPU audit identifies55 RGB/53 CDP
  dense products whose split selector assumes32×32 despite actual64×64 tiles;
  grouped products already use their actual grid. The completed bounded check
  above applies the same rule to the actual tile and yields only~1%; no
  threshold/algorithm sweep or underutilization claim. Broader goal stays active.

- **Residual target boundary attributed; prioritize compiler execution depth:**
  [result](docs/results/2026-10-14-target-boundary.md). Unchanged RGB29.084ms/
  CDP94.001ms captures match all272 reports, first outputs, plans and292/346
  initial/first/final tensors exactly. CPU decode/bonus1.217/3.774ms; selected
  host target/preparation/upload regions2.027/7.520ms cannot explain the~16.3ms
  RGB gap. Readback5.019/23.909ms includes outstanding GPU work, not pure copy
  cost. Posterior readback also feeds CPU replay/truncated training: retain it.
  RGB imagination839 groups/839-node longest logical chain; full world including
  prefix1,387/1,383. Reordering cannot remove real dependencies; these are
  unweighted counts, not GPU critical-path times. The subsequent row-selection
  intervention above limits their predictive value; do not continue a chain-
  count-driven sweep. Device targets remain useful, not the whole-gap explanation.
  Two guards, zero warnings/faults,544 synthetic updates, zero game actions;
  bounded CPU audits/build/Clippy pass. Native17f1e5e is instrumentation only,
  production unchanged, no worker. CI378 passes; broader goal stays active.

- **Exact-sequence GPU RNG removes a measured host bottleneck:**
  [result](docs/results/2026-10-14-device-rng.md). Isolated wider-control RGB
  32.246→30.053ms (6.80%) and CDP104.048→95.269ms (8.44%); saved2.194/8.779ms
  scales almost exactly with B8/B32 draw counts. Existing compiled plans,
  all272 reports, first targets and292/346 initial/first/final tensors are exact.
  Explicit pinned ChaCha12 fills session inputs on the shared GPU queue:
  2,107,118 primitive values, canaries/counter wrap/split fills/three no-host-wait
  consumer handoffs, four CPU tests and both24-update smokes pass. Seven guards,
  three allocation warnings,1,136 full synthetic updates, zero actions/faults.
  Retain the CPU lint failure and initially incomplete backend patch; corrected
  complete snapshot exactly matches the qualified wider source, no GPU retry.
  Native57a1a1a3 stays isolated; no production/default/learning change.
  RNG-only RGB remains2.34× retained complete JAX12.836ms.
  **[Device EMA now qualifies in the isolated candidate](docs/results/2026-10-14-device-ema.md):**
  fresh RGB30.260→29.169ms (3.61%), CDP95.473→94.059ms (1.48%), mostly in
  behavior/sync. General dense-F32 blend preserves compiler-derived copies and
  tracks both sessions' lifetimes.5,267,639 primitive values and853,503 first
  blend values/workload pass F64 bounds; four CPU tests/both Clippy checks pass.
  Initials/first non-EMA tensors/plans exact, later trajectories differ:
  82/272 RGB and1/272 CDP reports exact; final relative L2.0724/.8690, finite.
  No learning-equivalence or proven near-zero LaProp claim. Seven guards/six
  warnings,1,136 full synthetic plus one small SGD update, zero actions/GPU
  faults. The repaired offline audit's shadowed-handle failure remains.
  Nativea1c366a5 stays isolated; RGB now2.27× retained JAX, no fresh JAX capture.
  Keep GPU RNG/EMA. The subsequent target attribution above puts row/indexing/
  backward work first; device targets remain useful. No worker at publication;
  broader performance goal stays active.

- **Remaining gap is structural; scalar fan-out is not the main limit:**
  [graph-execution controls](docs/results/2026-10-13-execution-boundaries.md),
  [multi-output scalar result](docs/results/2026-10-14-pointwise-regions.md).
  Tiny dependent kernels cost~2.05µs on Vulkan/fully prequeued ordinary CUDA
  versus~.65µs on CUDA graphs. All3,168 completed counters pass; the failed
  large prequeue and separately reviewed short control remain. Application
  feeding is excluded, not every driver-internal host contribution. Current
  zero-work stream4.709ms is unchanged by access-mask weakening. Real kernels
  retain full barriers; ordinary Vulkan command reuse has no demonstrated
  equivalent graph-execution advantage.
  Isolated wider RGB32.103ms versus complete JAX12.836ms remains2.50×;
  production stays on its earlier qualified~53ms runtime. The next multi-output
  scalar trial barely helps: fresh RGB32.384→32.312ms/CDP104.135→103.054ms,
  0.22%/1.04%. RGB dispatches5,500→5,322 but groups2,271→2,239; CDP6,200→5,955/
  2,483→2,422. Shared scalar values are not the primary remaining bottleneck.
  Do not promote or extend this trial. All912 F64 comparisons/148,274 exact
  RNE values, staged lifetimes,52 CPU tests and nine guards pass. RGB272
  reports/all292 tensors exact; CDP initials exact, later trajectories differ,
  no learning parity. Physical horizontal packing3+1→2+2 is disclosed; reviewed
  per-product descriptors match. Two Clippy failures, typed-fan-out assertion,
  physical-descriptor audit and8-vs256 controller error remain; only the
  unstarted control ran after repair, no GPU job repeated. Three allocation
  warnings,1,136 full synthetic+146 small optimizer updates, zero actions.
  Native0483cf64 remains isolated; upstream2214412/fe5f5b5 unchanged.
  GPU RNG qualifies above; next targets/slow-critic EMA, then row-level normalization/backward
  fusion and safe view/layout propagation. Preserve stream/update ordering,
  explicit rounding, public preparation roots and backward lifetimes. No more
  scalar/binding-cap/tile sweep, unsafe barriers, CUDA learner or Atari restart.
  CPU capture's on/off-CPU timings are not SM utilization; JAX metrics return
  one update late. Prior near-zero LaProp sensitivity stays unresolved. No
  worker at publication; broader root-cause/performance goal remains active.

- **Residual gap attributed; synchronization is a substantial cause:**
  [result](docs/results/2026-10-13-residual-gap.md). Current staged RGB33.776ms
  versus retained JAX12.417ms,2.72×. Complete coarse trace23.918ms GPU-pass union/
  11.748ms uncovered uses a separately timed single-chunk schedule, not ordinary
  utilization. Six full272-update controls preserve every report/first output/
  all292 saved tensors. Initial15-pass trace omitted four transfers; repaired
  19-pass capture and all255 complete samples qualify, original/final tails stay.
  Four-session masks restore exact outputs/states. Causal zero-work test keeps
  all6,148 dispatches but removes2,551 dependency barriers:5.295→.274ms. This
  5.021ms difference is not removable real-work latency; never remove dependency
  barriers from real work. Eight guards, three warnings,1,634 synthetic updates,
  zero actions/production changes/workers. Next general typed RSSM row-operation
  fusion and device targets/RNG/EMA, not another tile sweep or mere command replay.
  Imagination has1,070 groups/1,367 dispatches; generated RMSNorm is present.
  Keep rounding/backward lifetimes and qualify full updates. Remaining algorithms
  also matter; no additive savings, JAX parity or whole-goal closure.

- **Staged world reuse measured; full performance goal stays active:**
  [result](docs/results/2026-10-13-staged-world.md). Single-pass BF16 retains
  posterior activations across imagination/late targets, then finishes backward.
  RGB39.392→33.681ms (14.50%), still2.71× retained JAX12.417ms. Duplicate10.319G
  nominal MACs/895 dispatches removed; remaining matrix/conv descriptors match.
  Exact initials/first categoricals/all215 first world tensors; small deterministic
  differences and later trajectories remain, not learning parity. Mixed CDP's
  differing forward precision prevents direct reuse: fallback111.040→111.212ms,
  exact272 reports/all346 saved tensors, no speedup. No staged video/truncated
  qualification or production change. Residual attribution above is complete;
  next device targets/RNG/EMA and materialization; no kernel sweep, additive
  speedup forecast, Atari restart or whole-goal closure. Command-buffer replay
  alone is insufficient: retained JAX without it still takes16.163ms.
  Nine guards, four retained allocation warnings;1,136 full synthetic updates,
  146 small-graph optimizer updates and24 raw-gradient-only passes. Corrected
  the initial post-clipping/raw-gradient scope with a separate exact check,
  not relaxed tolerances. Publication audit passes; no worker remains.

- **Convolution training layout measured; connected execution is next:**
  [result](docs/results/2026-10-13-conv-training-layout.md). Shared typed packed
  dX/dW reduces complete RGB46.860→39.372ms (15.98%), world27.679→20.413ms.
  All95.074G nominal matrix/conv MACs now multiply in BF16, still3.17× JAX;
  coverage is not utilization. Same logical operations, exact initials/first
  outputs, changed later trajectories.80 reduced F64 cases/24 conv partials,
  strict/grouped controls and production alias/poison pass. A retained long-sum
  failure reproduces in checked/packed controls; shared256-position compensated
  F32 accumulation fixes the gate without extra global scratch or relaxed bounds.
  Eight passing guards include one numerical diagnosis; one numerical failure,
  one allocation warning and568 synthetic updates remain. CDP24-report/first-state
  control exact; no new CDP timing, learning, production change or worker.
  End layout tuning. Next staged late-input/posterior/target/backward reuse and
  device dataflow, preserving activation lifetimes/old-weight consumers. No
  additive savings promise, Atari restart or whole-goal closure.

- **Grouped training path measured; full gap remains:** [result](docs/results/2026-10-13-grouped-training.md).
  Existing block forward/derivatives now use shared typed checked/tiled kernels
  under explicit BF16 policy; F32 masters/strict fallback retained. Complete
  RGB49.551→46.780ms (5.59%), CDP115.519→111.105ms (3.82%). RGB still3.77× JAX.
  Same dense/native-forward control; separate backward-convolution prototype
  not stacked.48 F64 cases/96 partials, production alias/poison, seven guards
  pass; zero warnings,1,088 synthetic updates/zero game actions. Group shapes
  and convolution descriptors match; exact initials/finite states, changed
  trajectories, no learning claim. CPU failures/cancelled patch retained.
  No worker or production change. Next efficient convolution derivatives, then
  staged posterior/targets/backward/device work; no grouped geometry sweep,
  additive speed promise, Atari restart or whole-goal closure.

- **Native forward-layout transfer measured, wider goal active:** [result](docs/results/2026-10-13-native-conv-layout.md).
  Shared typed tiled family now packs channels/hoists coordinates/reuses64×64
  tiles; every activation/filter pack included. Complete RGB51.525→49.647ms
  (3.65%), CDP118.212→115.548ms (2.25%); RGB still4.00× retained JAX. CDP's
  mixed world stage46.92ms is unchanged. Eight guards, one retained numerical
  failure, one warning;1,088 synthetic updates/zero game actions. The original
  forward control reproduces both unsplit F32-gradient failures exactly;
  existing eight-partition numerical tests pass unchanged F64 gates, no learner
  split change. Exact initials/finite272 reports, but categorical samples and
  later states differ; no learning parity. Candidate isolated, production pins
  unchanged; no worker remains. Stop forward-only tuning. Next extend efficient
  training/grouped layouts, then staged posterior→targets→backward reuse with
  safe activation lifetimes; world loss needs imagination targets, so simply
  reordering train_world is not equivalent. No Atari restart or goal closure.

- **Convolution algorithm cause demonstrated:** [result](docs/results/2026-10-13-conv-kernel-cause.md).
  Matched CUDA-twin layout/iterator629.39→342.67µs and reuse347.64→211.16µs
  include packing every call; all6,291,456 native/F64-fixture outputs exact.
  Counter instructions388.58M→61.24M, tensor active8.79→28.75%, occupancy
  87.92→31.75%: parallel occupancy is not useful tensor throughput.2.98x is
  cross-pair, not native/whole-update speedup. Five guards, zero new warnings/
  game actions; name-filter/controller/parser failures retained. Micro-study
  closed, broader goal active: implement layout/iterator/reuse in shared native
  type-aware forward/backward/grouped families, then complete-update checks and
  connected learner work. No more CUDA/tile sweep or production/Atari restart.

- **Convolution-backward diagnostic passes after retained numerical failure:**
  [result](docs/results/2026-10-13-conv-backward-gap.md). RGB53.157→48.901ms
  saves8.01%, still3.94x JAX.90.51% nominal matrix/convolution MACs now select
  BF16 kernels—not utilization.40 reduced F64 cases/17 partials, strict controls,
  five guards pass;544 synthetic updates, zero game actions/new warnings.
  Long unsplit accumulation failed; mandatory ≤4096-position partials fix it
  without relaxed gates, with unchanged actual learner splits. Arithmetic/
  final trajectories change; no learning-equivalence claim. Isolated candidate,
  not combined with tiled changes or adopted. The [counter diagnosis](docs/results/2026-10-13-conv-kernel-cause.md)
  above completes the next causal check; native algorithm integration and
  connected learner work remain. Root-cause goal stays active.

- **Shared tiled algorithm diagnostic passes, root-cause work continues:**
  [result](docs/results/2026-10-13-tiled-gap.md). One actual transposed product
  168.487→57.747µs (2.92x), all6,981,632 kernel outputs/512 F64 probes exact.
  Complete RGB52.987→51.616ms/CDP126.203→118.089ms save2.59%/6.43%; world training
  stays30.15/46.95ms. All272 reports/first outputs/292/346 initial/first/final
  tensors match exactly.12 guards,18 new aligned typed F64 cases, one warning;
  1,088 synthetic updates/1,152 kernel calls, zero game actions. Shared aligned
  FP32/FP16/BF16 generator, fixed64x64/K32 prefetch, checked fallback; no sweep.
  Production pins unchanged; portable selection/shared-memory review remains.
  RGB is still4.16x retained JAX; this does not close the goal. The subsequent
  convolution precision/algorithm checks above pass; native shared-family
  integration, grouped products and posterior/device work remain. No more
  dense geometry or Atari restart.
  No worker remains at this publication boundary; investigation remains active.

- **Execution-design diagnosis, not another cost bucket:**
  [result](docs/results/2026-10-13-execution-gap.md). Native RGB~53ms/JAX12.42ms
  remains open. JAX with F32 convolution derivatives/grouped products and no
  command buffers takes22.78ms; both scheduling controls preserve all reports/
  states.48.76% of nominal native matrix/convolution MACs remain scalar F32.
  A direct same-algorithm CUDA translation reproduces substantial-kernel
  slowness: native/CUDA convolution.668/.626ms, transposed matrix.168/.179ms
  versus cuBLAS.070ms.6,981,632 exact outputs/512 F64 probes pass; this is not
  identical instructions, timer boundaries or universal Vulkan qualification.
  Library F32-output convolution promotes operands to F32 and is not a matched
  BF16 control. The earlier operation-group accounting was not causal closure.
  K128 staging regresses7.48%/6.79%; coalescing saves only.70%/1.38% RGB/CDP.
  Latest Bladefe5f5b5/Nagaa4b4183 qualifies unchanged kernels with exact tensors
  and no speedup; Meganeura main2214412's fix is already in056562b. Production
  pins/defaults unchanged; all new code is isolated diagnostic work.
  **Next:** generalize efficient tiled/prefetched algorithms across types and
  training operations, not just the checked template; connect grouped/backward
  reduced precision, then reuse posterior/device targets/EMA and fuse graph
  materializations. No CUDA learner, local-tweak sweep or additive speed promise.
  Thirty-five guards pass, two CPU graph-assertion guards fail, six allocation
  warnings and corrected preparation/reporter failures remain.4,920 synthetic
  updates, zero game actions; no worker or Atari restart. Whole-update speed
  parity and learning/time-to-score remain unestablished.
- **Typed cooperative matmul consolidation qualifies:**
  [result](docs/results/2026-10-13-typed-matmul.md),
  [Meganeura PR257](https://github.com/kvark/meganeura/pull/257), pin056562b.
  Shared checked F32/F16/BF16 generation and typed partition selection/search;
  no dedicated BF16 dense shader or implicit precision change. Specialized
  fused/hardware-specific algorithms and legacy search exclusions remain.
  RGB52.803→53.047ms/CDP124.997→126.182ms:0.46%/0.95% more time, within the
  declared2% gate, not a speedup. All272 reports/first targets/292/346 tensors
  match exactly in these fixtures. Six typed cases/probes, existing F64,
  gradient/convolution/Adam checks and13 guards pass; two warnings retained.
  Invalid test options and odd-halfword tuning exclusion are corrected and
  retained, with the CPU source-spelling failure. Native F32 device execution
  remains unqualified on5080; no universal parity or learning claim. Cache30/
  memo3; Blade/Naga unchanged. Both locks/native checks, Clippy and193 binding
  tests pass; CI stays on PR31. No Atari restart.
- **BF16 dense split-K qualifies:** [result](docs/results/2026-10-12-bf16-split-k.md),
  [Meganeura PR256](https://github.com/kvark/meganeura/pull/256). Complete matched
  RGB56.269→53.065ms (5.69% less time), CDP126.880→125.432ms (1.14%). One pair
  each, unchanged model/batch/precision; no profiler or timing repeat.32 full
  F64 cases/859,216 exact values,12 long-gradient cases and existing BF16 checks
  pass. Ten guards pass/zero warnings; first uniform-binding and subsequent
  test-topology failures remain. Initial292/346 tensors exact;272 reports and
  saved tensors finite, but stochastic trajectories differ substantially.
  No game-quality equivalence or SM-utilization claim. Pin9e05731, runtime
  identical to timed87a3236; Blade663afd2/Naga c9fd6a85a unchanged. Cache29 rejects
  older matrix uniforms. PR256 is stacked on qualified integration5a493322;
  only the user merges. Installation/CI stay on PR31. CDP imagination54.21ms
  is unchanged; next remaining dense/grouped efficiency and convolution
  derivatives, then command-stream/posterior/device work. No Atari restart.
- **Full JAX root-cause accounting complete at operation-group resolution:**
  [report and explicit bridge](docs/results/2026-10-12-jax-gap-accounting.md).
  Both RGB12M recipes have10,498,772 trainable parameters. Ordinary native56.46ms /
  JAX12.42ms; same-executable input refresh adds only.262ms. Matched independent
  controls show inefficient native F32/BF16 matrices and convolutions, not only
  missing BF16 coverage. Source/phase accounting locates duplicate posterior work,
  CPU RNG/preparation2.33ms, targets1.61ms and slow-critic sync~1.04ms.
  Current eight-subset counterfactuals retain bindings/barriers but zero selected
  workgroup counts:44.088ms full graph versus44.101ms in complete updates.
  Shapley allocations17.320ms convolution/14.036ms matrix/6.839ms remaining plus
  5.892ms zero-work command stream distribute measured interactions; **not pure
  kernel times or independent optimization savings**. Masked outputs are invalid
  learner outputs; restored full outputs and292 tensors remain exact. The full
  profiled42.184ms difference plus explicit boundary/scheduling adjustments
  reconciles44.038ms ordinary gap (43.276ms with inclusive JAX host inputs).
  Single-dispatch timestamp sums62.069 versus44.287ms are rejected. Paired32-
  dispatch blocks retain0.434ms/0.98% excess, not normalized. Keep four production
  world submissions; the diagnostic single submission is~1.51ms slower.
  Next implement dense launch geometry/coalescing and grouped precision alongside
  convolution derivatives, command-stream reduction, then posterior reuse/GPU
  targets/EMA. Qualify whole updates before learning; no more open-ended profiling
  or automatic Atari restart.19 guards pass, three diagnostic guards fail,17
  warnings retained;2,747 synthetic learner updates plus25,269 new optimizer-free
  session replays, zero game actions. Old Graphics failures stay failed. No JAX
  speed parity, learning-quality or ordinary SM-utilization claim. Production
  pins unchanged, CI361 passes on preceding docs head, upstream20af998/7909fc1
  unchanged; current-head CI stays on PR31. No worker at publication boundary.
- **Split-convolution tile fix adopted:** [result](docs/results/2026-10-12-split-convolution-tiles.md),
  [Meganeura PR255](https://github.com/kvark/meganeura/pull/255). Refreshed control
  RGB61.027->56.058ms/CDP135.070->126.739ms:8.14%/6.17% less complete-update time.
  All272 numerical reports/first targets and292/346 initial/first/final tensors
  match; refreshed controls also reproduce retained fixtures. Ten-plan audit
  changes only four RGB/two CDP shader/grid fields. All17 guards pass, four
  allocation warnings retained. Extra96 short export updates and both CPU
  preparation failures remain; no repeated ordinary timing or game training.
  Main20af998 plus refreshed BF16/row addressing and the fix is5a493322;
  Blade663afd2/Naga c9fd6a85a unchanged. Combined cache28 rejects older BF16
  parameter encoding. Standalone9d7caff and integration have identical patch IDs.
  CI974's stale oracle selection assertion fails on Metal/lavapipe; test-only
  8a9baad corrects the exact16/32 expectations and passes the guarded5080
  rerun in2.68s. No runtime/tolerance change or timing repeat; failed logs stay.
  Both locks, native configuration/pin checks, Clippy and193 installed-binding
  runner tests pass. CI is tracked on the PR dashboard; only the user merges.
  No tile sweep or new learning allocation. Next: reduced-precision derivatives
  and matrix/RSSM cost, then posterior reuse/device residency. CDP imagination
  54.22ms now exceeds world45.94ms. Full JAX gap, ordinary utilization and
  learning quality stay open.
- **Nsight identifies a causal kernel bottleneck:** [result](docs/results/2026-10-12-nsight-root-cause.md).
  User enables counters on unchanged580.178.04. Exact complete native captures
  contain6,570 RGB/6,916 CDP dispatches; RGB weight gradients12.62ms versus
  JAX1.44ms, matrix products15.03 versus2.37ms. Graphs/residency differ, not an
  additive explanation of the4.90x gap. Tiny scalar weight-gradient tiles have
  high occupancy/MIO stalls, not tensor activity. One isolated16->32 split-tile
  check reduces ordinary RGB60.98->55.97ms/CDP135.13->126.61ms, exact272 reports
  and292/346 initial/first/final tensors; split counts/arithmetic unchanged.
  That initial variant was diagnostic only; the follow-up above adopts a
  qualified fix. Original next steps: upstream split-K/tile
  selection plus reduced-precision derivatives, matrix/recurrent work, then
  posterior reuse/device residency. Rechecked Meganeura07445e8/Blade7909fc1;
  that initial refresh recommendation is completed by the follow-up above;
  historical evidence retains its pins.
  Graphics2025.4.1/2025.5 crash in injected userspace libraries after completed
  tests; preserve failed guards/captures, no crash suppression. Counter results
  are diagnostic windows, not ordinary utilization or clean-profiler qualification.
  Systems native timestamps remain rejected. No Xid/wedge observed or game
  training; no GPU worker remains. Do not reload/change drivers without approval.
- **Backend performance before more Kindle training:** follow the
  [implementation plan](docs/backend_training_performance_plan.md): cooperative
  convolution weight gradients, actual mixed-precision storage/execution
  (BF16 where Vulkan supports it), then measured fusion/layout/synchronization.
  No Atari learning or frozen rollouts during this round; bounded synthetic
  updates and backend GPU qualification are authorized. Preserve image detail,
  CDP/exploration semantics and strict-F32 fallback. Use isolated upstream
  worktrees; only the user merges. The Freeway screen below was stopped cleanly
  at39,568 actions/9,831 updates, no completed learner or later arm. Partial
  checkpoint/declaration remain; no automatic restart or same-lifetime claim.
- **Combined backend qualifies:** [result](docs/results/2026-10-12-backend-integration.md).
  CDP12M/B32 F16→mixed161.32→135.17ms (16.21% less time); RGB12M/B8
  F16→BF1672.69→61.15ms (15.88%). Same native workloads/initials,16+256 updates,
  not additive isolated gains or learning/time-to-score evidence. Fresh JAX
  RGB FP32/highest24.29ms/BF1612.47ms leaves native2.52x/4.90x slower; different
  weights/RNG/graphs. World training and imagination are now about40% each of
  CDP. No SM-utilization or universal memory-saving claim.
  Meganeurad68f1a5 combines BF16PR250/fd9608f, row addressingPR249 and
  hardware-row selectionPR252 on current maineb8c90f; Blade663afd2 is on
  main7909fc1; Naga forkc9fd6a85a was committed with explicit permission.
  Git pins only, both locks agree, no merge commits or upstream wgpu PR.
  All ten final guards pass/three allocation warnings retained; independent
  CPU audit passes in4.767s. Exact initial346/292 tensors,272 finite reports/
  targets/checkpoints; old controls exactly reproduce their previous fixtures.
  New RGB retains exact preceding BF16 metrics/tensors; new CDP's first imagined
  targets match, but its F16 alignment change alters the later trajectory.
  Thirteen backend/14 Kindle CPU checks, Clippy and193 installed-binding
  runner tests pass. Upstream CI965/966/961, Blade1172 and all Kindle runtime
  CI355 jobs pass. Native83a9227f, Pythonf2f7b999. No GPU worker remains.
- **Precision decision:** `mixed` keeps BF16 posterior/imagination/behavior and
  the measured faster F16-input/F32-storage CDP world-gradient path. RGB's
  `bf16` also reduces forward convolution storage. F32 masters/optimizer/
  reductions remain; convolution derivatives/grouped products are not all BF16.
  [Convolution storage](docs/results/2026-10-12-bf16-convolution.md) and
  [fixed mixed-stage](docs/results/2026-10-12-mixed-world-precision.md) reports
  preserve the isolated decisions. Auto default/live acting stay unchanged;
  obsolete `bf16_dense` CLI support is removed, historical evidence retained.
  Broader CDP world/backward reduced storage and JAX's speed remain open.
  No new precision sweep or automatic learning campaign.
- **Failures/limits retained:** [PR248 attribution](docs/results/2026-10-12-convolution-attribution.md)
  rejects the isolated-fast convolution weight-gradient candidate twice on
  complete updates; keep it unadopted. [Hardware-row checks](docs/results/2026-10-12-convolution-subtiles.md)
  qualify forward/input gradients, not the unchanged scalar weight-oracle failure
  reproduced on control. [BF16 foundations](docs/results/2026-10-11-native-bf16.md)
  and [dense integration](docs/results/2026-10-12-bf16-learner.md) retain odd-clear,
  coverage and CI timing-test failures. Timing-only harvesting now captures all
  four world submissions with~2.7% diagnostic overhead; ordinary GPU fields
  remain last-submission-only, not SM utilization. Full wgpu xtask lacks
  cargo-nextest; scoped Naga checks pass. Final CPU preparation's invalid-CLI
  fixture resets eight ALE environments before rejection, no actor/update,
  setup frames unrecorded; repaired tests prohibit environment construction.
  Missing service PATH and audit-field failures remain with their corrections.
- **October11 performance pivot:** the user authorizes upstream refresh,
  reduced-precision training, cheaper exploration, wider batches, a native/JAX
  profile and artifact cleanup. Optimize one5080 learner before concurrent seeds
  or more GPUs. Old long training queues remain cancelled; the five-game goal
  remains unachieved. The [current plan](docs/kindle_single_life_dreamer_plan.md#performance-pivot)
  supersedes historical “next”, strict-parity-only and no-factorization-retry
  directions for this new performance study, without rewriting their failures.
  Use numerical checks appropriate to the selected arithmetic plus finite
  training and three-seed learning/time evidence; reduced precision need not
  match the legacy F32 trajectory. Keep F32 weights/optimizer/reductions and
  record matrix precision explicitly. No blind retries or concurrent learners.
  Remove regenerable build caches, not user files or unique experiment evidence.
- **Performance candidate:** [report](docs/results/2026-10-11-cdp-performance.md).
  Cheaper all-action exploration, wide grouped RSSM, broader convolution
  splitting and explicit training precision. B32/F16 measures162.00ms per512
  replay positions versus original B8/Auto107.92ms per128:2.66x throughput,
  not learning quality. Latest Meganeurad7a7293/Blade56f0565; retain430f14b
  profiles and older controls. Auto remains default pending the separately
  declared, now stopped [three-seed Freeway comparison](docs/experiments/2026-10-11-cdp-performance-learning.md):
  optimized B8/Auto versus B32/F16,200k actions each, no old queue resumes.
  Half operands alone save little; cooperative convolution weight gradients
  and reduced-precision/fusion coverage are measured backend opportunities.
  No extraGPU or concurrent-seed assumption. SM utilization remains unmeasured.
- **October7 goal: reach DreamerV3 quality on Boxing, Pong, Freeway,
  Breakout and Qbert with CDP; retain the budget comparison.** All five
  and the budget question remain open. A completed screen, learned subset or
  smaller model does not complete this goal.
- **Five-game200k screen complete:** twelve new learners plus all three retained
  Freeway cohorts, seeds1009/2017/3019,49,939 updates each. Frozen means:
  Boxing69.74 vs.39 initial, Pong−18.53 vs−20.31, Freeway24.14 vs0,
  Breakout4.36 vs1.61, Qbert403.13 vs152.43. Four games improve in every seed;
  Pong improves in only one. All45 guards,720 selected natural frozen episodes,
  zero updates/cutoffs, exact346 saved tensors/model, full replay/video checks
  pass. New screen finishes12:36 UTC in5h43m45s; no worker remains. [Result,
  curves and whole videos](docs/results/2026-10-05-cdp-five-game-screen.md).
  Fifteen learners cost3M actions/749,085 updates/6h56m28s training, excluding
  earlier development work. None reaches the predeclared long-run reference.
- **Pong500k follow-up complete:** three fresh124,939-update learners finish
  October5 at17:42 UTC in3h32m29s. Frozen−21.00/−2.375/−20.9167 versus initial
  mean−20.375,7/72 wins. Online mean−15.2667 versus published−7.1622 near2M
  frames (protocol differences retained). Two weak seeds do not recover.
  All nine guards,144 natural frozen episodes, zero updates/cutoffs, exact
  tensors and full replays/videos pass. [Result and curves](docs/results/2026-10-06-cdp-pong-budget.md).
  That raw-cosine allocation stays closed; the new centered allocation below
  is a separately reviewed study, not an automatic extension.
- **Frozen Pong diagnosis complete:** all four guards/exact346 tensors pass,
  131,072 diagnostic actions and zero actor updates. Ball/paddle information is
  readable from every CNN, including initial weights, but ball state/reward
  prediction is weak in1009/3019's RSSM. Their tiny cosine errors mostly match
  a constant-mean prediction;2017 retains useful state and h15 forecasts.
  [Result and controls](docs/results/2026-10-06-cdp-pong-world.md).
  This is not complete encoder collapse. Batch-centered CDP cosine now
  [qualifies](docs/results/2026-10-06-cdp-centered-qualification.md): eight
  guards,32,768 independent centered derivatives,1,300 unchanged-CDP upstream
  comparisons and both production/frozen smokes pass. Natived32f3dc8.
  **Centered CDP's three-seed screen passes:** six fresh200k runs finish
  October6 08:41 UTC in2h55m56s. Frozen centered/control−8.4583/−18.6528,
  paired+10.1944 [7.5,12.375]; every centered seed improves over its actual
  initial and matched control. All18 guards/288 selected natural episodes,
  exact346 tensors/model and full replay/videos pass. Only4/72 centered wins,
  not mastery. [Result and all videos](docs/results/2026-10-06-cdp-centered-learning.md).
  Promote centered loss as the next candidate (`--cdp --cdp-centered`), not
  a solved architecture. No new encoder/capacity/decoder or blind extension.
  **Paired world probes complete:** six guards/exact346 tensors pass at09:02
  UTC in9m40s;196,608 diagnostic actions, zero actor updates. Every centered
  seed has readable recurrent ball state and useful h15 negative-event
  forecasts. Raw2017 remains stronger on some coordinate probes; positive /
  terminal evidence is sparse. [Result](docs/results/2026-10-06-cdp-centered-world.md).
  **Throughput trial passes:** grouped B128 imagination reduces ordinary full
  synthetic updates30.49->28.01ms (8.12%), dispatches3,100->1,210. Same upstream
  backend/control, exact first targets and272 metric reports,428,032 F64 block
  outputs,2,824 upstream comparisons, raw/centered train/frozen smokes and all
  14 guards pass. [Result](docs/results/2026-10-06-cdp-imagination-throughput.md).
  Keep it; no optimization sweep. This is not measured whole-game throughput
  or GPU utilization. Previous per-dispatch instrumentation perturbs timing.
- **Five-game centered500k allocation complete after evaluation repair:**
  [results, curves and whole videos](docs/results/2026-10-07-cdp-centered-five-game-budget.md).
  Fifteen learners,7.5M actions/1,874,085 updates/16h24m7s training; original
  service finishes October7 at02:30 UTC in16h45m23s. Frozen means Boxing77.69,
  Pong4.54, Freeway30.26, Breakout5.14, Qbert993.75; every seed beats its actual
  initial. Boxing/Freeway pass historical gates in all three seeds; Pong54/72
  wins remains unstable, Breakout/Qbert remain weak. None reaches its long-run
  reference. This is not a matched five-game centering/RGB ablation.
  All720 selected natural episodes, exact346 tensors/model, zero frozen updates,
  full replay/video checks and46 GPU guards pass;31 whole videos and all tails
  remain. Breakout3019's original8.0833 cutoff-containing cohort stays invalid.
  The corrected7.9167 reproduces its exact207,288-action prefix; extra209,504
  evaluation actions and the original timeout are retained. Host workers8->1
  in that rerun are disclosed; no unchanged-runtime speed claim. Vector v5
  counts natural episodes; v4 keeps its exact completed-episode receipt schema.
  Both CPU audit failures and pre-fix source snapshots remain. No learner retry.
- **Frozen Breakout diagnosis complete, interpretation limited:**
  [result and next bounded check](docs/results/2026-10-07-cdp-breakout-world.md).
  All six initial/final models finish in9m40s after a12.29s smoke. Seven guards,
  exact346 actor tensors/model and identical action/frame/pixel traces pass;
  196,992 diagnostic actions/61,600 readout updates/zero actor updates. No worker
  remains. Ball state is weakly readable and all three h15 ball readouts lose
  to constant mean; reward evidence is only7/8 positive events at h1/h15.
  The shared pixel control overfits: per-pixel whitening turns bounded RGB
  values into validation/test magnitudes up to4,021. F64 reproduces saved GPU
  predictions at9e-7 relative L2, implicating diagnostic conditioning, not an
  observed GPU prediction fault or proof of lost image information.
  The declared fixed-range follow-up finishes in5.01s:2,048 readout updates,
  selected1,792 by validation, one passing guard, zero new actions/actor updates.
  Paddle test R².979; ball near zero even in training, so this head is not an
  information ceiling. An independent fixed color centroid locates483/483
  eligible held-out open-playfield balls at.628/.542 source-pixel x/y RMSE.
  RGB64 retains those balls; no full-frame guarantee or actor-side detector/RAM.
  Keep original failed heads and source snapshots; no readout sweep.
  [Follow-up data](docs/results/2026-10-07-breakout-pixel-control.json), artifacts
  `runs/cdp-breakout-pixel-range-20261007.kQHipWpq`.
  **12M preflight passes:**1,024 training/195 finite updates in22.23s, then
  1,024 frozen actions/zero updates/exact346 tensors; two guards, no new
  warnings. Last128 updates average108.98ms. [Evidence](docs/results/2026-10-07-cdp-12m-preflight.json).
- **12M capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-07-cdp-capacity.md).
  Three fresh500k Breakout seeds score48.17/15.88/50.58, mean38.21 versus
  retained1M5.14; paired+33.07 [12.38,44.17]. Every seed improves, none reaches
  the381.81 long-run reference.1.5M actions/374,817 updates/11h39m37s training,
  3.58x the retained1M wall time. Service finishes15:39:48 UTC in11h42m14s;
  all nine guards/144 natural frozen episodes/exact346 tensors and six whole
  replays/videos pass. Three allocation warnings,18 excess episodes and all
  tails remain;88,120 frozen actions. CPU review passes; no worker remains.
  Only whole-model capacity changes (16.33M actual parameters including
  CDP/exploration), not an isolated encoder/RSSM/ensemble test. Native6d38eea2
  unchanged. Reused development seeds and old3019's8->1 host-worker repair
  remain disclosed.12M is the next Atari quality candidate,1M the fast control;
  12M prior forecasts remain untested; the separately allocated Pong result
  below extends policy evidence to a second game. No automatic extension.
  All five quality targets and the budget question remain open.
- **Projection-reuse throughput trial rejected:** [two numerical failures](docs/results/2026-10-07-cdp-exploration-throughput.md).
  Tiny/1M B3 F64 checks pass;12M head0 exceeds the pointwise gate with identical
  expanded/factored outputs. A reviewed unstarted1M B128 check then differs by
  4.909e-4 relative L2 versus2e-5. No tolerance relaxation, timing, game actions
  or actor updates; candidate source/binaries/failures retained, original graph
  restored. Precision-path selection is a hypothesis, not proven causality.
  No optimization sweep. [Latest-backend control refresh](docs/results/2026-10-07-meganeura-control-refresh.md)
  qualifies f104f35 on unchanged graphs:20 guards/2,824 upstream comparisons,
  three train/frozen smokes,13 retained allocation warnings. Native83be73bf.
  Paired12M timing107.85->108.00ms;272 metric reports/first outputs/final346
  tensors match exactly in this fixture. No speedup or universal parity claim.
  These scoped checks do not erase the failed stronger F64 check.
  12M updates previously averaged110.03ms,55.4% imagination/29.9% world training;
  utilization remains unmeasured.
- **Four-game allocation complete:** [declaration](docs/experiments/2026-10-07-cdp-12m-four-game-capacity.md),
  Pong, Qbert, Boxing, Freeway; three fresh500k-action seeds each, same centered
  CDP/action-effects recipe.6M actions/1,499,268 updates total; service finishes
  October9 at16:08:01 UTC in47h1m32s. All final/actual-initial natural cohorts,
  exact tensors/full replay/videos pass. CPU completion review finishes16:12:35;
  no learner or review worker remains. No allocation restarts.
  Keep completed12M Breakout and all retained1M controls with their original
  backend pins; no exact same-binary capacity ablation claim. No automatic
  budget extension; all five quality targets and the budget question stay open.
- **12M Pong capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-08-cdp-pong-capacity.md).
  Three fresh500k seeds score7.21/12.96/17.04, mean12.40 versus retained1M4.54;
  paired+7.86 [1.04,18.25],69/72 wins. Every seed improves, but only3019 passes
  the historical mastery gate; long-run20.45 remains unmet.1.5M actions/
  374,817 updates/11h40m7s training,3.57x retained1M cost. Cohort finishes
  October8 at04:52 UTC; Qbert1009 starts next in the unchanged allocation.
  Nine guards/144 natural frozen episodes/exact346 tensors/model and all six
  replays/videos pass;359,456 frozen actions, zero updates, eight excess episodes
  and eight allocation warnings retained. Independent CPU review passes in15.21s.
  A supplemental reporter's retained-control key error and failed source remain;
  its correction changes no runtime, cohort or actions. Native83be73bf versus
  historical controls6d38eea2 is disclosed. No Pong extension or mastery claim.
- **12M Qbert capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-08-cdp-qbert-capacity.md).
  Frozen1,790.63/1,328.13/4,855.21, mean2,657.99 versus retained1M993.75;
  paired+1,664.24 [435.42,3,823.96]. All three reward means improve, but first
  pyramids2/0/0 ->0/0/24:1009 loses its two completions; only3019 completes
  reliably. No seed passes mean15,000 or long-run193,220.77.1.5M actions/
  374,817 updates/11h39m35s training,3.51x retained1M cost. Cohort finishes
  October8 at16:34 UTC; Boxing/Freeway continue in the unchanged allocation.
  Nine guards/144 natural frozen episodes/exact346 tensors and all six replays/
  videos pass;114,408 frozen actions, zero updates, two excess episodes and17
  allocation warnings retained. CPU review passes in15.09s. Task completion
  is selected from existing replay outcomes, not positive-return counts; the
  old Pong reviewer source remains. Capacity helps three games' rewards, not
  yet reliable suite competence. No new Qbert budget or representation sweep.
- **12M Boxing capacity comparison complete:** [results, curves and whole videos](docs/results/2026-10-09-cdp-boxing-capacity.md).
  Frozen73.17/92.33/90.29, mean85.26 versus retained1M77.69; paired+7.57
  [−1.29,20.71] includes zero.1009 declines slightly; both sizes win72/72 and
  pass historical gates in all seeds. None reaches99.6133.1.5M actions/
  374,817 updates/11h41m33s training,3.57x retained1M cost; this does not
  establish a consistent larger-model advantage. Cohort finishes October9
  at04:20 UTC; only the declared three Freeway pairs remain. Nine guards/144
  natural frozen episodes/exact346 tensors and six whole replays/videos pass;
  212,288 frozen actions, zero updates, seven excess episodes and11 allocation
  warnings retained. CPU review passes in15.17s. Keep both sizes; the completed
  Freeway comparison below closes this allocation, not the quality goal.
- **12M Freeway and five-game capacity comparison complete:**
  [results, combined comparison and whole videos](docs/results/2026-10-09-cdp-freeway-capacity.md).
  Frozen33.58/17.25/31.79, mean27.54 versus retained1M30.26; paired−2.72
  [−13.29,4.08]. Crossing gates72/72 ->48/72;2017 regresses, with slow learning
  rather than an observed late collapse. All initials are0. The cohort costs
  1.5M actions/374,817 updates/11h41m22s training,3.57x retained1M. Nine guards/144 natural
  frozen episodes/exact346 tensors and six whole replay/videos pass;294,912
  frozen actions, zero updates, no excess episodes, all tails and13 warnings.
  All15 larger learners cost7.5M actions/1,874,085 updates/58h22m14s training,
  3.56x retained1M16h24m7s. All45 guards/720 selected natural episodes/30 full
  videos pass;1,069,184 frozen actions,35 excess episodes and52 warnings remain.
  Capacity helps every paired reward mean in Breakout/Pong/Qbert, not every
  title or task milestone. Keep12M as quality candidate and1M as fast control;
  no universal promotion, capacity sweep or automatic extension. All five
  long-run quality targets,12M prior forecasts and the RGB budget claim remain open.
- **Budget claims:**200k actions are~800k emulator frames, more than Atari100k's
  400k but less than Atari57's200M. Compare published online curves and our
  frozen controls separately. Retain all released seeds and the declared
  last10% long-run target, not convenient bins. Protocol/model/averaging windows
  differ; no exact parity or same-hardware compute-saving claim without a
  matched RGB control. At~800k frames the current means are similar to published
  Atari57 on Boxing/Pong/Breakout, ahead on Freeway, somewhat behind on Qbert.
- CDP is the roadmap main path, while the CLI still defaults to learned RGB:
  main-path declarations must pass `--cdp`. Joint small CNN + deterministic
  latent prediction, Dreamer RSSM/replay/imagined policy, qualified split rates,
  cosine loss and ac_grads=false. No frozen Tiny/DINO/Large requirement or
  visualization decoder. RGB remains the reference/fallback; causal Tiny is
  an optional later video/3D hypothesis, not a2D obligation.
- Freeway exploration is unlocked across all three unassisted200k seeds;
  only3019 passes its original mastery gate. No Freeway-only gate tuning.
  [Result and limitations](docs/results/2026-10-05-freeway-effects-200k.md).
  Action-effects disagreement subtracts each predictor's all-action mean before
  measuring ensemble variance; no new action hint, coefficient, encoder or
  externally shaped reward. Retain earlier all-zero/weak32k screens and their
  additional compute. There is no matched200k extrinsic-only superiority claim.
- The strategy reset remains iteration speed -> useful representation ->
  exploration/reward -> video priors -> native deployment. Phases0–2 and the
  Seaquest CDP/RGB comparison are complete. The3x12M speed target is an unmet,
  user-accepted stretch target. The completed five-game screen's31.62ms
  updates spend51% in imagination,27% world training; about8x aggregate /1x
  per-stream realtime. The new-backend Pong comparison measures32.11ms raw /
  32.13ms centered updates, about28min per200k learner; no new speed claim.
  GPU utilization remains unmeasured. Optimize measured whole-agent cost.
- No old queue restarts. The stopped45-run small CDP/RGB/Tiny comparison retains
  seven completed Freeway pairs plus interrupted CDP3019 at126,408 actions.
  The historical12M matrix retains24 completed and21 cancelled unstarted entries.
  Their controls/protocols are not the current intrinsic study. Venture, Tiny
  comparisons and new representation matrices are deferred by the five-game goal.
- **October6 backend refresh qualified:** Meganeura mainb684ffd9 adds bounded
  shader reads and primitive/composite gradient/recognition fixes over592a2f5a;
  Bladee349cddf unchanged. Native683ccd22 passes18 guards,2,824 upstream CDP/RGB
  comparisons, independent cosine/action-effects checks and train/frozen
  smokes. [Qualification](docs/results/2026-10-06-meganeura-refresh.md).
  Completed learning retains nativef4b6a5c7/592a2f5a; diagnostics disclose their
  new runtime separately. No claim that these fixes caused the old failures.
  Upstreamc6376542 adds attention-backward batching/layout search, not an
  identified CDP fix. It is now adopted/qualified for CDP/RGB with the grouped
  throughput trial; completed probes retain their training native. Tiny's
  attention path is not newly qualified by these CDP/RGB checks. Neighboring
  user work stays untouched.
  Rechecked October7 at02:44 UTC: Meganeurafc3a2fb adds f32 cooperative acceleration
  and checked geometry; Blade49ec60a changes presentation-damage hints. No
  relevant fix for the current small-CDP learning weakness was identified.
  Keep active jobs/frozen diagnosis on the qualified runtime; review a refresh
  before the next learner change, without repinning completed evidence.
- **October9 next-step review:** at approximately2M frames the completed12M
  online means are ahead of released DreamerV3 bins on Pong/Breakout and close
  on Boxing/Qbert; the completed Freeway online mean is also ahead of that bin.
  This is descriptive, not protocol-matched parity or a change to the five
  long-run targets. Freeway and the CPU completion review are complete.
  The final upstream recheck adds Meganeura31026833's native16 split-reduction
  stability fix; Blade56f0565 is unchanged. The [refresh qualifies](docs/results/2026-10-09-meganeura-refresh.md):
  20 guards,2,824 upstream comparisons and all three train/frozen smokes pass.
  Native7311547d; three allocation warnings and the corrected CPU provenance
  failure remain. Paired12M updates107.97->108.23ms: no speedup. First outputs,
  all272 metric reports and346 final tensors match exactly in this fixture.
  Qualification/timing finish16:38:45 UTC; no worker remains. No identified
  cause of earlier learning failures or historical repin. The [B8/T64 versus
  T16 preflight](docs/results/2026-10-09-cdp-sequence-preflight.md) completes:
  four guards, identical initial346 tensors and both frozen restores pass,
  zero warnings. Same65,536 replay positions cost.8580/.8185ms each:4.6% lower,
  below the declared10% gate. T64 has only2.05GiB sampled headroom,56MiB above
  the floor. Keep T16; no sequence sweep or T64 learning campaign.8,192 training
  actions/1,156 updates plus2,048 frozen actions; final T64 debt.5 is reconciled.
  Longer BPTT changes learning/batch statistics/scheduling, not just speed;
  keep capacity/exploration fixed. No sweep or automatic budget extension.
  Neighboring user work stays untouched.
- **Imagination cost attributed:** [one fixed-checkpoint capture](docs/results/2026-10-09-cdp-imagination-cost.md)
  finds the four all-action ensemble first projections at20.50ms/38.7% of
  instrumented imagination dispatch time,91.23G MACs. One guard, exact346 tensors
  across profiling, zero warnings/game actions, one synthetic update; no worker
  remains. Timestamp intervals are not ordinary latency or SM utilization.
  No shader defect, weak-seed cause or new optimization is established; rejected
  factorization stays rejected. Do not make optimization an indefinite learning
  gate. Keep12M/T16; all five quality targets and the RGB budget question remain
  open.
- **Qbert 2M budget study complete:** [results, curves and whole videos](docs/results/2026-10-11-cdp-qbert-budget.md),
  three fresh seeds1009/2017/3019 on unchanged12M/T16 centered CDP/action-effects.
  Frozen5,172.92/5,812.50/11,115.63, mean7,367.01 versus retained500k2,657.99;
  paired+4,709.03 [3,382.29,6,260.42]. First pyramids0/0/24 ->20/22/20:
  both weak seeds improve, but3019 loses four successes. No seed passes the full
  historical gate or long-run193,220.77. Online mean6,545.50 versus published
  8M-frame14,607.50 is descriptive, not protocol-matched parity/inferiority.
  Six million actions/1,499,817 updates/46h40m1s training,4.002x retained500k;
  service finishes October11 at16:10:24 UTC in46h44m25s. All nine guards/144
  natural frozen episodes/exact346 tensors/model and six whole replays/videos
  pass;221,648 frozen actions, zero updates, nine excess episodes and ten warnings
  remain. Final CPU review passes in52.02s; no worker remains. Native7311547d
  versus retained83be73bf is disclosed; no exact same-binary budget-ablation
  claim, historical repin, lifetime resume or automatic extension.
  Next inspect action-conditioned reward/terminal forecasts along recorded
  Qbert policy trajectories in a separately declared, bounded frozen diagnostic,
  before another learning allocation. No capacity/representation sweep or new
  five-game budget. All five quality targets,12M prior forecasts and the RGB
  budget claim remain open.
  **1009's recorded500k prefix matches exactly across runtimes:** [CPU comparison](docs/results/2026-10-09-cdp-qbert-prefix.md),
  initial346 tensors,62,500 transition records,124,939 non-timing learner reports,
  671 episodes and976 common progress reports; online last50 remains1,809.
  Review completes21:30 UTC in6.62s, zero extra actions/updates/GPU work. This
  supports prefix agreement for one seed, not universal parity, pixel identity,
  final500k tensor equality or three-seed/final2M competence. The independent
  end-of-seed reviewer also reproduces all three historical pairs in3.84s.
- **Qbert frozen forecasts complete:** [result and controls](docs/results/2026-10-11-cdp-qbert-forecasts.md).
  Six guards/exact346 tensors and identical traces pass at17:16:46 UTC;
  386,496 diagnostic actions/zero updates/24m16.84s capture,12 warnings.
  Every2M seed beats zero/mean/persistence/unrelated actions on h15 reward
  MAE/RMSE; positive AUC.951/.967/.973, paired MAE gain over500k−1.981
  [−2.854,−1.318]. Later-stage forecasts improve strongly;3019's first-pyramid
  forecast improves neither uniformly nor enough to establish stable control.
  Continuation's discount is explicitly removed for terminal probabilities;
  original metrics remain. Only16/seed h1 and3/2/2 h15 terminals: survival
  remains unresolved. One sampled latent path, final-policy state distribution,
  not exact vector belief replay or universal world-model quality. The failed
  40,789-action lifecycle capture,128-action smoke and original sources remain;
  427,413 total diagnostic actions,15 warnings across all attempts. No worker
  remains. Next a bounded greedy-versus-sampled Qbert check using existing action
  mode and actual initial controls; no new learner budget or representation
  sweep. All five quality targets and the RGB budget question remain open.
- **Greedy Qbert check complete:** [result and whole videos](docs/results/2026-10-11-cdp-qbert-action-mode.md).
  Scores5,444.79/6,392.71/9,022.92, mean6,953.47 versus sampled7,367.01;
  paired−413.54 [−2,092.71,580.21]. First pyramids20/22/20→22/24/23, so all
  improve that fraction, but3019 loses2,092.71 points and none passes the full
  mastery/long-run gate. Keep sampled evaluation. Six guards/144 natural
  episodes/exact346 tensors and whole videos pass;218,936 actions/zero updates,
  14 excess episodes/nine warnings retained. Finishes17:33:10 UTC in3m32.63s;
  independent CPU review passes2.33s. No worker remains. Preparation's lazy
  module error/source remain, before GPU work. Next review longer-horizon
  value/credit assignment and measured learning cost; account for the critic's
  combined extrinsic/intrinsic target. No entropy/representation sweep or
  automatic learner extension; all five quality targets remain open.
- After Atari: video priors for dynamics/behavior, mind-games vkQuake2 then
  TMNF, a small GOG/Wine panel, held-out cross-game adaptation/retention, then
  swarms. One effective actor first; no concurrent learner service now.

- **Qbert live-value check complete:** [result and whole videos](docs/results/2026-10-11-cdp-qbert-values.md).
  All three2M critics beat zero/calibration mean on held-out return MAE/RMSE;
  R².435/.927/.531. Bias+236/−62/−152 is not shared collapse;1009's first-pyramid
  optimism and3019's later underprediction remain. Intrinsic contributes<.03%
  of measured on-policy returns, not proof it was unnecessary during learning.
  Nativee9ebbf5d's lazy batched probe uses exact live features/actions; unchanged
  learning graphs/backend31026833/56f0565, historical training native7311547d.
  Five guards/exact346 tensors/72 natural episodes/full traces and videos pass.
  193,248 diagnostic actions plus1,024 smoke, zero updates;88.87s capture, five
  warnings, nine excess episodes/all tails. CPU timestamp-comparison failure
  and source remain; no GPU retry. Final independent CPU review passes18:06:57
  in2.11s; no worker remains. Next one bounded upstream measured-build check
  of unchanged learning graphs and whole-update cost, not a representation/
  entropy/critic sweep or indefinite optimization gate. Rejected factorization
  stays rejected. No new learner budget; all five quality/RGB budget goals open.

- **October11 measured-build check complete:** [result](docs/results/2026-10-11-cdp-measured-build.md).
  Ordinary Meganeura2c42b448/Blade56f0565/native0f26ee63 qualifies:21 guards,
  2,824 upstream comparisons, three train/frozen smokes and exact first outputs/
  272 metric reports/346 final tensors.107.93→108.30ms: no speedup. Original
  4GiB logical-memory preflight fails; one declared20GiB correction stops on
  value/bonus numerical tolerances at the third callback. Both failed guards,
  source/binaries and ten total warnings remain; zero search optimizer updates/
  game actions, no GPU recovery. Test-only hook removed; no further search,
  gate relaxation or rejected-factorization retry. Next separately declare a
  three-seed Pong experience-budget study on unchanged12M/T16, targeting69/72
  wins and inconsistent mastery. Keep actual initials, natural cohorts and20.45
  long-run target; no automatic suite extension. All five quality/RGB goals open.

- **Pong2M study stopped by user for the pivot:** the
  [fixed declaration](docs/experiments/2026-10-11-cdp-pong-budget.md) remains intact;
  [stop receipt](docs/results/2026-10-11-cdp-pong-stop.json). Seed1009 runs
  October11 18:56:09–20:25:19 UTC, ending gracefully at189,576 actions /
  47,333 updates, zero debt,1h29m6s runner time. The partial and actual-initial
  checkpoints remain under `runs/cdp-pong-2m-20261011.lb5DyI6h`; saved partial
  hashes match the final log. Native0f26ee63/2c42b448/56f0565 unchanged. No frozen
  evaluation or completed learner;2017/3019 and all evaluation jobs are cancelled.
  Service/control-group and game workers are gone, Restart=no, no timer. Systemd's
  failed exit records the intentional controller interruption, not a GPU fault;
  the interrupted guard has no terminal verdict. Zero new allocation warnings;
  143 baseline warning records remain. Do not restore this as an uninterrupted
  lifetime or resume the queue. All five quality/RGB goals remain open.

## Architecture and invariants

- CDP keeps Dreamer's categorical RSSM, replay and imagined actor/critic; it
  predicts detached CNN embeddings from the deterministic prior, not from the
  posterior containing the target. The CNN is jointly trained, not frozen.
  Keep the [qualified recipe](docs/experiments/2026-10-04-cdp.md) reproducible.
- Eight small-recipe environments share batched perception/policy and one
  learner, not causal histories. Preserve per-stream recurrent state, RNG,
  replay and resets. Shared mutable actor/learner weights require serialized
  access. Historical speed comparisons use N6/12M and their original settings;
  a smaller model or lower replay ratio is not an unchanged-learning speedup.
- The target is game GPU -> capture/preprocessing -> encoder -> belief/policy ->
  action readback. Only selected actions leave the acting hot path; rewards,
  checkpoints, diagnostics and sampled learner data may cross the host.
  Scalar targets/slow-critic work still partly uses the host. Do not claim a
  completely GPU-resident learner or environment from a buffer entry point.
- Preserve native image detail. CDP's learned RGB 64 path performs one GPU resize;
  never downscale to RGB 64 and upscale for JEPA. Optional causal Tiny uses
  independent 16-arrival chunk histories; a chunk reset is not an environment/
  RSSM reset. Do not slice Large weights or silently substitute random features.
- Capture must validate ownership, producer completion, memory visibility and
  ring reuse. Blade's existing `Memory::External(Fd(Some(fd)))` ->
  `create_buffer` path borrows/duplicates the FD and uses matching resource/
  allocation recipes at binding offset zero. Device/driver compatibility is the
  caller's responsibility. Acquire/release are safe whole-buffer encoder methods,
  separate from import; first-use ownership belongs to the buffer, not a slot.
  No parallel import constructor or exporter-metadata framework.
- Dullahan GPU_SYNC v4 rejects older protocol tags and transfers the whole ring.
  Its exact-byte ring passed functionally but logged an allocation warning.
  The v4 real-producer test is unrun; earlier v3 vkQuake success is not v4
  qualification or game competence. Keep sparse reward/terminal adapters and
  full mind-games GameSession integration distinct from transport success.

## Evidence to preserve

- [CDP evaluation](docs/results/2026-10-04-cdp-learning.md): three fresh Seaquest
  pairs,200k actions/49,939 updates each. CDP 543.6 versus RGB 318.1 online mean;
  paired+225.5 [62.8,330.4],13.6% less wall time and 35.9% less world-training time.
  All three pairs favor CDP; all guards/seals/counters/finite checkpoints pass.
  No frozen competence or Atari-wide claim. Keep all episodes/tails and failures.
- Frozen CDP/RGB diagnostics:196,608 extra random actions,24 GPU readouts,
  all 1,626 saved tensors unchanged, zero actor updates. CDP's RSSM has better
  readable player state; h15 latent forecasts beat persistence/constant mean/
  unrelated actions. One-step persistence, zero-reward MAE and matched privileged
  position persistence still win. Sparse reward events limit the result.
- [CDP qualification](docs/results/2026-10-04-cdp-qualification.md): independent
  cosine values/1,024 derivatives,1,300 CDP/1,524 RGB upstream comparisons and
  production replay/restore/smokes pass. Raw gradients use exact pre-step weights;
  optimizer/EMA is independently checked, not bitwise stochastic trajectory
  parity. Earlier sequential-weight/configuration/stack failures and warning
  stops remain retained; no numerical tolerances were relaxed.
- [Phase 2](docs/results/2026-10-01-frontend-decision.md) completed in seven of
  ten replication attempts plus six separately declared Tiny runs. Frozen Tiny
  halves world training but takes 18% longer end to end; no clear pretraining
  benefit. Initial Tiny was also frozen. Later
  [joint-Tiny](docs/results/2026-10-03-joint-tiny-learning.md) and
  [direct-policy](docs/results/2026-10-03-policy-tiny-learning.md) screens found
  no clear early advantage. Preserve failed diagnostics and extra offline/
  interrupted compute in their reports, not as new active work.
- Historical non-sticky reliability is3/5 (Boxing, Pong, assisted Freeway).
  Breakout/Qbert fail their original gates; sticky Pong fails (2/24 wins,
  mean−7.1667). Freeway training used random-action probability .5/hold 64.
  Tiny checkpoint `7fe9b252` used 250k same-title random-play observations,
  45k train +5k validation/game. These are not fresh CDP or online-only results.
  See the roadmap's game/video table and [archive](docs/experiments/README.md).
- Phase 1 implementation is complete; the3x12M speed target is an unmet,
  user-accepted stretch target. MinAtar's three-seed screen takes 8m18s but weak
  scores do not establish competence. It is a separate CPU-environment/small
  public-observation recipe, not the CDP Atari control.
- Historical pre-performance production backend: Meganeura main `2c42b448`, Blade `56f0565`,
  native `0f26ee63`; the [October11 ordinary refresh](docs/results/2026-10-11-cdp-measured-build.md)
  qualifies unchanged graphs without speedup. Its measured-construction trial
  fails and is not adopted. Historical `31026833`/`7311547d` retains the Qbert2M
  learners and [October9 qualification](docs/results/2026-10-09-meganeura-refresh.md).
  Historical `f104f354`/
  `e349cddf`/`83be73bf` retains the completed four-game learners and its
  [October7 qualification](docs/results/2026-10-07-meganeura-control-refresh.md).
  Previous `c6376542`/`6d38eea2` remains the historical
  grouped-imagination and completed capacity-study runtime.
  The [October6 refresh](docs/results/2026-10-06-meganeura-refresh.md) qualifies
  CDP/exploration, RGB and frozen Tiny on its historical `b684ffd9`. The historical
  [October4 refresh](docs/results/2026-10-04-meganeura-main-qualification.md)
  passes 2,824 upstream CDP/RGB comparisons, causal Tiny streaming and all three
  production/frozen-restore smokes; 14 guards pass with no new warnings. No
  learning campaign resumes. The previous `13b19d33` already included the
  attention-value alias fix and passed all 148 independent F64 Tiny gradients;
  do not relabel that historical joint-Tiny result as new-backend qualification.
  Check upstream before diagnosing already-fixed issues. Historical Phase 2/
  capture results retain their original pins; no recursive repinning or migration.
  The [capture report](docs/results/2026-10-02-matching-external-allocations.md)
  is separate from ordinary learner qualification.

## Research and status

- Keep one decision-focused roadmap with one game-status table and direct
  rollout/world-report links. Detailed results go in compact JSON + Markdown
  under `docs/results/`; large raw artifacts stay in `runs/`.
- Use the active PR description for dated done/running/next status, results and
  limits. [PR31](https://github.com/kvark/kindle/pull/31) is the current dashboard;
  PR29 is merged history. No STATUS.md. Update at meaningful boundaries, not
  every poll. Preserve the user's untracked `TASK.md`.
- Compare one changed mechanism, at least three learner seeds, matched actual
  interactions and score-vs-actions/time curves with seed-bootstrap uncertainty
  or suite IQM. Episodes/streams are not independent learner replicates.
  Numerical smokes are not learning evidence. One matched timing plus numerical/
  learning check suffices for a speed change; no micro-campaign.
- Frozen evaluation never updates model/optimizer tensors. Predeclare cohorts,
  retain unfinished tails and compare untrained controls for competence claims.
  New controllers must select `cohort(..., natural_only=True)` explicitly; its
  completed-episode default is retained only for historical reports. Do not
  reuse an old controller's complete flag as proof of a natural cohort.
  Separately evaluate prior forecasts with persistence/constant/unrelated-action/
  reward controls and event counts. Posterior estimates are not forecasts,
  features are not imagined RGB, and privileged observers never enter policy.
- A restore without replay/RNG/live belief is not an uninterrupted lifetime.
  Natural deaths/respawns are allowed; cloning/rewinding a live game for learning
  is not. Disclose all aids, shaped rewards and pretraining experience.

## GPU operation

- Ordinary bounded GPU work is authorized on driver580.178.04. Normal JAX/CUDA
  initialization, including internal NVML, is allowed. Separate NVML polling/
  legacy health loggers stay off. No NVML-free backend or CPU learner workaround.
  Unmeasured utilization is not zero; timings are not SM-utilization readings.
- Serialize heavy GPU jobs under `python/examples/gpu_host_guard.py` in persistent
  systemd user services: `Restart=no`, `KillMode=control-group`, bounded deadline.
  Explicitly own/clean up game processes too. Review every failure before follow-up.
- Require the expected native device and >=2 GiB sampled Vulkan estimated budget
  headroom. Budget-minus-usage is not physical free or peak VRAM. Last observed
  boot: `3e89d55c-a9e5-472f-a18a-06508c5bafa7`.
- **October4 user permission: proceed with GPU use unless it is wedged.**
  New declarations set `record_allocation_warnings=true`: retain standalone
  allocation warnings without aborting or requesting approval for each one.
  This supersedes the earlier120-second and proposed startup-only restrictions.
  API/numerical failures, hard faults and deadlines still stop the affected job
  for review; do not confuse a failed job with a wedged GPU or an approval gate.
  No blind retries. A wedge needs recovery review. The known
  `VUID-StandaloneSpirv-None-10684` is non-blocking by explicit user direction;
  other validation errors remain job failures, not evidence of a wedge by themselves.
- No GPU reset, driver reload/change, reboot or power-cycle without new user
  approval. Historical Xid62/154 incidents remain unexplained; successful
  no-NVML jobs prove neither causality nor safety. Never retry quarantined
  75dfe, 0a98775/02b600a1 or 070f4b51/7db0d05c bundles.
  Follow [incident response](docs/gpu_incident_response.md), not old followers.

## Implementation discipline

Keep exercised production code small. Delete obsolete encoding/compatibility
code instead of extending it; do not preserve old checkpoints at the cost of the
new path. Preserve unrelated user edits, especially in Meganeura/Blade/mind-games.
Use proportionate tests, formatting and Clippy. Do not rebuild unchanged native
source for docs, or compile during matched timings.

Automate routine validation; inspect long training about every 30 minutes or on
completion, not counters on every poll. Heavy preparation uses one CPU,2 GiB and
zero swap. Apply CPU limits to preparation/analysis, not GPU workers in matched
runs: the native host-worker count follows the available CPU quota. Review CPU
test filters: Meganeura has unignored GPU tests.
Only the user merges; commits/pushes are allowed. Keep history linear.
