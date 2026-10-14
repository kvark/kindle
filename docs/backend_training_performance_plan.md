# Backend training performance plan

Updated October 14, 2026. Improve one RTX 5080 learner on Meganeura + Blade
before returning to Kindle learning campaigns. Keep CDP, model capacity,
replay ratio, image detail and exploration semantics unchanged. This is an
implementation plan under the [roadmap](kindle_single_life_dreamer_plan.md),
not a new learning allocation; PR31 remains the status dashboard.

## Current diagnosis and implementation order

**Checked matrix staging wastes instructions even when tensor work matches.**
The [matched matrix/counter study](results/2026-10-14-matrix-library.md) verifies
four actual native contracts and2,672,640 exact F64/JAX values. Grouped forward
costs10.357µs native/10.324µs in a checked CUDA translation versus3.930µs for
cuBLAS kernels. Counters show the same24,576 BF16 tensor instructions and
near-identical occupancy, but3.40× warp instructions/5.22× integer work:
scalar16-bit staging versus128-bit vector transfers. This is not missing
tensor support or a case for more occupancy. Counters describe the translation,
not native Vulkan; whole JAX changes addends/transposes/output contracts, so
the miniature is not exact full-executable algorithm parity. Five guards pass,
two failed fixtures/control retained, zero allocation warnings/actions/updates.
No new whole-update speed claim; current24.340/12.836ms stays1.90×.

Next implement one alignment-proven vector-staging path within the shared typed
checked generator, retaining safe tails/transposes/groups and explicit storage/
multiply/output types. CPU contracts, independent GPU numerics, then one complete
update pair; no BF16-only fork or geometry/occupancy sweep. The existing aligned
engine already stages vectors; small/irregular shapes must not lose that benefit
merely because they need bounds checks. No promise this closes the full gap.

Precision continuity remains separate: native matrix outputs247 BF16/469 F32
versus JAX456/96, not a one-to-one mapping. Native backward intentionally widens
intermediates; latest upstream retains that policy. Its independent time cost
is unmeasured. The extra BF16-output small-row control fails exact single
rounding and stays failed; all four actual native contracts keep exact checks.

**Matrix inefficiency is concentrated in the checked small-product path.**
The [current-candidate attribution](results/2026-10-14-matrix-cost.md) assigns
2.903ms/63.54% of matrix cost to521 checked products doing only12.01% of nominal
matrix arithmetic;195 aligned products cost1.666ms. All already multiply BF16.
World recurrence supplies2.568ms of the checked cost.302 products have eight
rows,160 additional products reduce only eight elements. Plans/outputs/reports/
292 tensors remain exact; one original update,1,468 optimizer-free replays,
one guard/zero warnings. These are interacting fixed-graph allocations, not a
new whole-update speed claim. The matched small-row/short-reduction library
study above now motivates general checked staging rather than a new algorithm
or cross-use gradient aggregation. Retained full
HLO has96 reduction-eight cuBLAS calls with BF16 inputs/F32 results and `beta=1`;
those counterparts do not avoid F32 accumulation. No tile sweep.
The separate standard Vulkan executable query returns statistics but zero
internal representations; close that inspection route without a driver change.
Its logged compile-only guard passes; no native instructions were obtained.

**Precise Vulkan synchronization does not recover CUDA graph execution cost.**
The [four-path real-dependency check](results/2026-10-14-dependency-paths.md)
finds existing/synchronization2-global/exact-buffer barriers all~2.05µs per
dependent kernel; immediate event set/wait costs~4.10µs. All3,204 counters/two
guards pass, zero warnings/actions/updates. No arm reaches the20% gate; close
the API-flag detour without full-learner integration. This is not proof of a
universal Vulkan limit, nor a new whole-update measurement. CUDA graphs' lower
device cost remains distinct from our successful host command reuse. Next
isolate remaining kernel staging/reuse and work per execution unit on the
current candidate, not another barrier/row/iterator tuning sweep.

**Repeated convolution indexing is now a demonstrated native bottleneck.**
The [incremental iterator intervention](results/2026-10-14-convolution-iterators.md)
cuts complete cached RGB25.475→24.340ms (4.45%), exact272 reports/first outputs/
plans/initial-first-final292 tensors. Decoder dW improves37.15%, forward28.43%;
the five boundary kernels total3.747→2.501ms. Shared typed cursors retain all
arithmetic, precision, tiles, partitions and barriers. Numerical/address/alias
checks and seven guards pass, zero warnings. Keep it isolated; no production/
CDP/learning qualification. No further iterator/geometry sweep.

The preceding counter evidence explains why this was worth changing.
The [RGB-boundary study](results/2026-10-14-boundary-algorithms.md) matches all
five native kernels and isolates decoder forward/dW at1.310/1.742ms versus
cuDNN.261/.338ms. Encoder forward/dX are competitive. The dW CUDA bridge
executes1.785B warp instructions versus100.06M;73.1% are integer work.
cuDNN issues12.5% *more* identical BF16 tensor instructions, yet is much faster.
Its occupancy is lower, tensor-pipeline activity much higher. This is an
iterator/staging/reuse problem, not a reason to chase occupancy or remove
compensation. Bridge counters are not native Vulkan counters; its2.374ms is
36.2% slower than native. Two primitive JAX algorithms differ from the full
executable; dominant dW matches exactly. All outputs/F64 controls pass.

**The remaining gap includes a device-side execution-path difference, not
merely CPU launch overhead.** The [new causal controls](results/2026-10-13-execution-boundaries.md)
measure about 2.05µs per tiny dependent kernel on Vulkan **and ordinary
prequeued CUDA**, versus 0.64–0.66µs on CUDA graphs. Same CUDA kernel/data,
all work queued before GPU timing; 3,168 counters pass across the completed
studies. The first large prequeue check fails its unchanged 100ms host-gate
limit and remains failed; the reviewed shorter control passes.

CUDA graph execution therefore does more than avoid recording commands again.
Ordinary Vulkan command-buffer reuse has not been shown to deliver its lower
device dependency cost. Large-kernel CUDA translations previously stayed
within roughly 7% of Vulkan: this is a small-dependent-work finding, not a
blanket API penalty. No hardware mechanism or equivalent Vulkan extension
has been established.

| Current whole-update boundary | Time |
| --- | ---: |
| Retained isolated RGB12M/B8/BF16, row selection + latest compiler | 28.213ms |
| Isolated command-cache candidate, fresh ordinary full pair | 25.530ms |
| Isolated incremental-address candidate, fresh ordinary full pair | 24.340ms |
| Retained JAX core, same model/batch size | 12.420ms |
| JAX including caller seed/metric handling | 12.836ms |
| Retained isolated CDP12M/B32/mixed, row selection + latest compiler | 93.084ms |

The retained RGB control is2.20× complete JAX; the current iterator candidate is
**1.90×**. CDP is a different
workload and is not compared directly to that JAX number. Production stays
on its earlier qualified ~53ms RGB runtime; isolated diagnostic improvements
have not changed installed pins/defaults or restarted Atari.

The [actual-grid correction](results/2026-10-14-dense-launch-grid.md) saves
only1.00% RGB/.92% CDP against these fresh controls. Numerical/plan checks pass,
but trajectories differ; additional partitions also add reductions, stores
and dependencies. Preserve the isolated candidate, not a promotion or another
grid test. The table deliberately retainsb3a086f, not the grid candidate.
The [fresh weighted profile](results/2026-10-14-residual-reassessment.md)
now covers every update, including GPU RNG/EMA and asynchronous tails.
Native GPU-pass union22.144ms versus retained JAX10.295ms dominates the
captured gap; RNG+copies+EMA total only.430ms. Fixed-session masks allocate
6.888ms convolution/4.475ms matrices/5.545ms other plus4.272ms empty command
stream. These are interacting cost allocations, not additive promised gains.
Single-chunk ordinary/coarse30.066/30.115ms preserves all reports/states with
.16% instrumentation overhead; normal four-chunk28.213ms remains separate.

The [latest upstream compiler check](results/2026-10-14-storage-propagation.md)
finds no material5080 gain: fresh RGB28.197→28.314ms/CDP92.965→93.049ms,
all reports/outputs/plans/tensors exact. Blade96c1dc7/Naga a97559b removes
ordinary storage-access propagation; real barriers remain. Keep the refresh
in isolated experiments. The inherited row-gradient F64 failure is reproduced
on both compilers and diagnosed as three BF16 bin crossings; scoped old/new
whole-learner parity is not an unconditional primitive pass. No production
promotion, tolerance relaxation or5080 hardware-fence claim.

The [row-selection intervention](results/2026-10-14-row-selection.md) limits
the dependency-count hypothesis: groups fall2,271→2,058 RGB /2,483→2,253 CDP,
but fresh complete updates improve only28.777→28.479ms /93.938→93.001ms, about1%.
Every272-report/first-output/saved-tensor comparison is exact. Preserve its
reviewed uniform-binding failure and one matrix-addend fusion difference.
Do not continue a row-operation tuning campaign or equate fewer groups with
proportional throughput. The source audit instead exposes a concrete selector
inconsistency: dense64×64 tiles still choose K splits from a32×32 grid estimate;
grouped products already use their actual grid. The completed check above
finds no missing large speedup. End this line of tuning.

The [device RNG intervention](results/2026-10-14-device-rng.md) now saves
6.80% RGB /8.44% CDP against fresh32.246→30.053ms /104.048→95.269ms controls.
It generates the exact original ChaCha12 streams directly into GPU inputs:
2,107,118 exact primitive values, both24-update smokes, all272 reports and
292/346 initial/first/final tensors pass. Existing compiled plans are unchanged;
46 small RNG dispatches/two submissions replace host generation/uploads.
Seven guards/three retained warnings,1,136 synthetic updates, no game actions.
Keep this isolated candidate; JAX is the retained comparison, not a fresh run.

The subsequent [device EMA intervention](results/2026-10-14-device-ema.md)
removes parameter readback/blending/upload: fresh RGB30.260→29.169ms (3.61%),
CDP95.473→94.059ms (1.48%). Behavior/sync accounts for most of the difference.
5,267,639 primitive values and853,503 first production blend values/workload
pass the independent F64 bound; initial and first non-EMA state are exact.
Later trajectories differ (82/272 RGB and1/272 CDP reports exact), with
finite final states; no learning equivalence or proven LaProp causality.
Seven guards/six warnings,1,136 synthetic plus one small SGD update, zero
actions. Keep it isolated; installed runtime and training remain unchanged.

The [multi-output scalar follow-up](results/2026-10-14-pointwise-regions.md)
does not materially close this gap: fresh RGB32.384→32.312ms (0.22%),
CDP104.135→103.054ms (1.04%). It removes178 RGB dispatches but only32/2,271
dependency groups. Nine guards/912 independent F64 comparisons/148,274 exact
RNE values pass; all RGB reports/states are exact, CDP trajectories differ.
Keep the source/evidence, not a production promotion or another scalar sweep.
The stricter physical-descriptor audit catches horizontal packing3+1→2+2 in
CDP; its reviewed per-product audit passes. All failures and three warnings
remain. No worker or game actions; broader goal stays active.

What is now causally established:

- **Kernel algorithms matter.** Missing training precision coverage, excessive
  convolution indexing/staging and weak data reuse were identified and improved
  in [shared typed training kernels](results/2026-10-13-conv-training-layout.md).
  Full BF16 coverage does not imply library-level execution efficiency.
- **Duplicate work mattered.** [Staged posterior reuse](results/2026-10-13-staged-world.md)
  removes 10.319G nominal MACs/895 dispatches and improves complete RGB
  39.392→33.681ms. Mixed CDP's different forward/world precision intentionally
  stays on its existing path.
- **Fine-grained dependencies remain expensive.** The pre-selection5,500-dispatch
  graph has 2,267 inter-group barriers. Its empty stream takes 4.709ms with
  full, partial or no memory-access masks, keeping compute execution ordering.
  Access-mask weakening is not the fix; real dependencies must remain correct.
- **Scalar fusion alone is insufficient.** General typed row fusion helped;
  widening the pointwise input cap then reduces RGB dispatches 5,846→5,500 but
  groups only 2,333→2,271, saving just 2.32%. Shared/saved values and opaque
  backward operators block larger fusion. Independent primitives qualify,
  but the wider candidate changes trajectories and exposes near-zero LaProp
  sensitivity; no learning-equivalence claim or production promotion.
  The subsequent multi-output region trial preserves shared values safely yet
  removes only1.4% of RGB's groups. Further fusion should target row operations
  and layout boundaries, not merely additional scalar edges.
- **Host boundaries also matter.** Nsight finds native main-thread
  9.094ms on CPU/23.160ms off CPU per captured update; most poll time is
  sleeping, not CPU computation. Moving identical RNG streams onto the GPU
  saves2.194ms at B8 and8.779ms at B32, almost proportional to draw count.
  Device EMA then saves1.091ms RGB /1.413ms CDP. Target construction still
  crosses the host. These interventions measure complete replacements, not
  separately additive CPU/transfer budgets. JAX pipelines metric readback inside its wrapper; the excluded
  caller checking is only 0.177ms. Timing scope cannot explain the gap.

The [target-boundary attribution](results/2026-10-14-target-boundary.md)
changes the immediate order: selected host target/preparation/upload regions
total2.027ms RGB /7.520ms CDP; decode/bonus alone1.217/3.774ms. The unchanged
instrumented updates29.084/94.001ms preserve every numerical report/tensor.
Readback5.019/23.909ms includes unfinished GPU work, not pure transfer overhead.
Host targets cannot explain the remaining~16.3ms RGB gap.

The same plans have839 imagination groups and a longest logical chain of839
kernels; the full compiled world graph, including the staged posterior prefix,
has1,387 groups versus1,383. Existing ordering is already near the dependency
depth lower bound. These unweighted counts are not GPU critical-path timings.
First-max sampling, recursive feature concatenation and opaque backward/cast
boundaries expose concrete row/indexing operations for a coherent compiler
change. Preserve deliberate rounding/shared outputs; do not drop dependencies.

**Command recording is now addressed; continue with the GPU algorithms.**

The [completed integration](results/2026-10-14-connected-learner.md) automatically
selects127 connected forward/backward regions and removes378 dispatches, but
complete RGB28.280->27.999ms saves only.99%, below its3% gate. Numerical and
staged checks pass, later trajectories differ; no CDP follow-up, row sweep or
production promotion. This bounds what that promising block can contribute.

The [heavy-path diagnosis](results/2026-10-14-heavy-issued-work.md) finds a
more concentrated problem: five checked RGB-boundary convolutions consume
3.785ms versus2.906ms for18 packed convolutions, despite only11.1% of nominal
convolution MACs. All already use BF16. Actual source/grid accounting finds
152.574G cooperative product slots for84.756G useful MACs; two three-output
kernels waste10.67x slots. This is not hardware-counter or JAX-padding evidence,
and checked indexing/staging also costs time. The [typed scalar alternative](results/2026-10-14-narrow-products.md)
qualifies90 numerical cases but slows full RGB28.488->30.492ms (7.03%). It
removes padded tensor products but also loses tensor hardware; reject without
a geometry/CDP sweep. The expensive path remains unresolved.

The [command-cache intervention](results/2026-10-14-command-cache.md) reduces
whole updates28.416->25.530ms (10.16%). Exact272 reports/first outputs/plans/
292 tensors qualify unchanged arithmetic. Nsight shows5351->65 descriptor
updates and5297->11 pipeline binds per update;15 submissions and all5286 GPU
graph dispatches remain. Cache immutable graph chunks, keep optimizer state
live, retain resources/barriers/staged lifetimes. No cache sweep. Selected
API durations1.562->.116ms are incomplete host accounting and the capture
perturbs timing; use the ordinary pair. The16 lifecycle cases pass again
with repaired validation logging;1152 total synthetic updates/zero actions.

Keep the isolated improvement, not a portable production/CDP qualification.
The remaining~2x gap still requires actual GPU algorithm/dependency work.
That matching and the incremental-address intervention are now complete above.
Next distinguish the remaining staging/reuse cost from device dependency cost
on the current candidate, rather than repeat an iterator/tile sweep. JAX's
F32-output controls widen operands and do not isolate gradient storage precision.
The five kernels' remaining~2.50ms footprint cannot explain the whole11.50ms
residual; matrix/other materialized work and CUDA graphs' lower device issue
cost stay in scope. No whole-gap or Atari-learning closure.

An API/literature check does not supply a shortcut: Vulkan's NV generated-command
spec allows individual dispatches to execute in any order; a generated list is
not a replacement for our dependent compute chain. Preserve real dependencies.
[Khronos specification](https://docs.vulkan.org/spec/latest/chapters/device_generated_commands/generatedcommands.html#vkCmdExecuteGeneratedCommandsNV).
A recent A40/CUDA11.8→13.0 command-stream study attributes lower **host launch**
cost to a smaller submission footprint. It explicitly avoids interpreting
undocumented commands; it does not explain our already-prequeued5080 device
timing or establish Vulkan equivalence. No driver modification/reload is proposed.
[Primary study, §6.3](https://arxiv.org/html/2604.26889v1#S6.SS3).

The [general reduction prototype](results/2026-10-14-connected-row-region.md)
passes its20% block gate: complete8x256 norm/SiLU forward+both derivatives
12.841->8.280us (35.5%). Additional row/element outputs retain forward values;
column prologues combine backward work. All281,052 GPU values and70,263 CPU
values pass, normal aliasing/no-alias with poison included. Two guards/two
warnings; zero optimizer updates/actions. This was explicit plan construction;
the completed automatic BF16 learner integration above is its limited outcome.
No block-size sweep or extrapolation to all135 regions.
Retained full28.213/12.836ms remains2.20x; this alone cannot close the gap.

The [CPU design control](results/2026-10-14-training-residuals.md) is complete
on latest Meganeura7574ad1, including the upstream shared-widening fix.
Actual HLO provenance proves saved forward norm/SiLU tapes. Native primitive
expansion is numerically correct but worsens14 dispatches/seven groups to
28/16 or32/17; recomposition and materialized broadcasts prevent the simple
approach. Do not enable it in production. A general region must express row,
column and elementwise inputs, preserve required shared outputs alongside a
reduction, and keep deliberate F32/BF16 rounding and forward/backward lifetimes.
The retained learner has141 conservative pure duplicate dispatches, not evidence
for a large CSE-only gain. Prototype and measure one connected region before
generalizing. This is not a substitute for heavy-kernel/device-execution work:
the entire row/reduction allocation is only2.329ms of the remaining gap.

1. The [accumulation-cost check](results/2026-10-14-accumulation-cost.md)
   closes a specific hypothesis: removing compensated dW reduces fixed-world
   GPU time only16.169->16.025ms (.89%), while reproducing14 pointwise F64
   failures/layout. Keep compensation; no conditional Nsight or precision sweep.
   All292 tensors and restored outputs/gradients are exact; no alternative
   optimizer writes. This is not new complete-update throughput.
   The [separate prefetch check](results/2026-10-14-conv-prefetch.md)
   also closes: full RGB28.436->28.714ms (.98% slower), exact reports/states,
   independently qualified derivatives. No conditional CDP/counter follow-up.
   The [reachable-work audit](results/2026-10-14-executable-work.md) finds
   only1.40% extra native nominal heavy arithmetic despite2.20× latency.
   Convolution and grouped shapes/counts match. JAX's actual graph combines
   four head projections into one wide GEMM; native typed products and sessions
   do not. This is a concrete compiler difference, not a measured large gain.
   The [non-heavy attribution](results/2026-10-14-materialization-cost.md)
   now allocates1.432ms layout/casts,2.329ms row/reductions and1.631ms other.
   Full world row/backward/scalar work is the larger non-heavy region; copies
   alone are not the explanation. All plans/pipelines/outputs/292 tensors match.
   Design connected row/backward lowering with explicit layouts, rounding and
   multiple live outputs. Account for packing, views and all consumers in one
   representative measured/numerical control before generalizing. More nominally
   fused operations are not enough; no automatic optimization allocation.
   Continue general shared typed algorithms/coarser compiled execution, not
   another scalar/row/binding-cap/tile sweep or bespoke Dreamer kernel.
   The time-batchable prior head remains an unweighted source observation,
   not an established large opportunity.
2. GPU RNG and EMA qualify in the isolated candidate, with EMA's later
   trajectory differences retained. Target attribution is complete; move
   construction onto device. Posterior readback also refreshes CPU replay and
   initializes truncated/microbatched training; it is not merely diagnostic.
   Preserve that required work in timing and any future device transition.
   Preserve stream/RNG/update-order
   contracts. Existing posterior/imagination handoffs already stay on device.
   Qualify one connected device-side stage at a time; no CPU learner workaround.
   A large bespoke target/sort pipeline is not the main remaining gap-closing
   bet. Keep the CPU arithmetic design check separate from GPU qualification.
3. Investigate a lower-overhead backend execution path only with a causal
   control. Cached CPU recording and CUDA-style device graph execution are
   distinct; do not promise that ordinary Vulkan command replay closes this gap.
4. Qualify one bounded complete RGB/CDP comparison with independent numerics
   and saved-state checks, then a separately declared learning/time-to-score
   comparison before changing training defaults. No additive savings forecast,
   unsafe barrier removal, CUDA learner or new Atari allocation.

Staged reuse must retain forward activations across posterior → imagination/
targets → world backward; optimizer writes follow consumers of old weights.
The world loss needs replay-value targets produced after imagination. Merely
moving `train_world` earlier is not equivalent. Preserve microbatch/BPTT and
mixed-precision fallbacks; keep useful submission overlap until a measured
replacement exists.

The [current residual accounting](results/2026-10-14-residual-reassessment.md)
and retained JAX capture use different schedules/instrumentation. Keep the
1.048ms ordinary-versus-captured gap difference explicit. Native command-
building/submission spans4.548ms and decode/targets1.513ms in the single-chunk
capture; do not count GPU waits again as CPU arithmetic or claim SM utilization.
Four current guards pass/one warning,569 synthetic updates and2,380
optimizer-free session replays, zero actions/workers. All source/CPU coverage
failures remain. The full root-cause/performance goal stays active.

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

## Current implementation state

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
