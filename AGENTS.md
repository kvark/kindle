# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **Active goal: understand the remaining JAX gap and establish how to reach
  similar Meganeura performance.** It remains open. Latest isolated complete
  RGB12M/B32/T16 is48.068ms versus inclusive JAX27.223ms,1.77x; B8 retains22.780/12.737ms,
  1.79x. Production is unchanged; these are not current CDP learning results.
  Atari remains stopped. No automatic promotion or old queue restart.
- Follow the [backend performance plan](docs/backend_training_performance_plan.md)
  and [roadmap](docs/kindle_single_life_dreamer_plan.md). Keep one effective
  native actor before swarms. CDP remains the learning main path, while the
  CLI defaults to RGB: declarations must explicitly pass `--cdp --cdp-centered`.
  RGB is the reference/fallback; causal Tiny is an optional later video/3D
  hypothesis, not a frozen Large/DINO requirement.
- **The gap is structural, not simply a BF16 switch.** Reachable heavy arithmetic
  differs by only1.40%. Precision coverage, duplicate posterior execution,
  command recording and some indexing/layout costs have measured fixes.
  Remaining library convolution data feeding, device execution/materialization
  and host decode/targets are distinct costs. The [actual full-JAX capture](docs/results/2026-10-15-library-convolution-feeding.md)
  has10.33x/8.27x fewer warp instructions than same-algorithm CUDA dX/dW,
  despite more tensor instructions. CUDA/native timing bands fail: these are
  not native instruction counts or an8x speed promise. High occupancy is not
  the target; useful tensor throughput is.
- **Earlier shared-workgroup convolution family rejected:** [result](docs/results/2026-10-15-convolution-family.md).
  One shared typed fixed-address workgroup implementation covers all13 distinct
  forward/dX shapes,15 actual occurrences. Complete operations total
  12.833→18.131ms,41.29% slower; all transforms/scratch included. No cherry-picked
  selector, geometry sweep, Naga/Blade integration or full-update trial follows.
  All503,808 F64 values/416 independent probes/257,949,696 exact output pairs
  pass;27 guards/five allocation warnings, zero actions/learner updates.
  Native CNN1 dX's timing shifts~6.13→2.15ms within the retained128 samples;
  cause unknown, no samples discarded or new steady-state claim. Two CPU
  publication OOMs in the preceding qualification and the family shader-template
  compile failure remain. Streaming audit and corrected command-recording
  accounting pass without GPU repeats or raised limits.
- The preceding [single-derivative qualification](docs/results/2026-10-15-convolution-qualification.md)
  remains valid:2.14890→1.29280ms,39.84% less GPU time,196,608 F64 outputs.
  It occurs once in the full graph; its.856ms saving is only1.43% of the retained
  update before integration effects. The family rejection supersedes its
  proposed compiler-integration next step, not its numerical evidence.
  Workgroup scope, larger blocks, alignment metadata and precision alone do
  not reproduce the library's asynchronous multistage implementation.
- **GPU categorical means qualify:** [result](docs/results/2026-10-15-categorical-gpu.md).
  Full B32 updates59.963→54.224ms,9.57%; three distribution readbacks25.1MB→98KB.
  All28,860 independent F64 comparisons/873 symmetric zeros pass; exact initial/
  first292 tensors and first targets, later F32-vs-F64 trajectories differ
  (2/272 reports exact, final relativeL2.05875). Four guards/three warnings,
  545 synthetic updates, zero actions/workers. CPU plan-map serialization
  assertion and missing standalone SPIR-V tool failures remain; exact semantic
  plan/constant/pipeline audit and offline capability check pass without GPU
  retries. Targets/normalizer remain on CPU; no fully resident learner claim.
- **Whole-family contract audit complete:** [result](docs/results/2026-10-15-convolution-contract.md).
  All23 actual products/layer parameter paths and eight captured JAX updates
  reconcile. Library uses pass-specific reuse/three-to-five-stage schedules;
  current14 packed occurrences express8-byte BF16 transfers. Six CPU Naga
  cases reproduce a missing contract: cooperative loads infer scalar type from
  the pointer, discarding requested F32/F16/BF16 when storage is `vec4<u32>`.
  All three equivalent Vulkan SPIR-V modules validate offline. No GPU run,
  compiler edit or speedup claim. Retain original packed48-channel audit error
  and reserved-keyword probe failure; reviewed versions pass. Native tiled
  convolution work369.099G versus226.492G nominal is not a JAX instruction ratio.
- **Transfer observation complete, qualification failed:** [result](docs/results/2026-10-15-convolution-transfer.md).
  All14 packed occurrences/11 contracts cost4.933→4.170ms,15.48% less; generic
  control calibrates+3.66% GPU/+4.00% queued wall. The.763ms saving is only1.41%
  of the retained full update as a scale comparison, not measured whole benefit.
  Native/typed/wide outputs match exactly;40 forward and1,277 dX cancellation
  values fail original absolute envelopes in all three. Exact integer checks
  confirm every failed reference. Both stopped allocations stay failed; a
  separately reviewed cost-only observation repeats no cohort and cannot promote
  the candidate.132 guards/six warnings,18.819M F64 comparisons,5,115 operation
  calls, zero learner updates/actions/workers. No Naga work or timing sweep.
- **A concrete resource-interface constraint:** the separate properties-only
  query finds48KiB Vulkan explicit workgroup shared memory, while19/23 captured
  cuDNN convolution products use more; profiled dX/dW use76/80KiB. One passing
  guard/no warnings, no logical device or dispatch. A literal explicit-shared
  port does not fit, but this is not a causal slowdown or parity-impossibility
  proof. Keep compiler-managed storage distinct. CPU reviewer membership and
  compact/raw-schema failures remain; corrected audits pass without GPU repeats.
  The [fixed library-side check rejects](docs/results/2026-10-15-library-pipeline.md):
  original engines40/33 reconstruct, but both choice1 configurations fail3000
  before execution-plan construction. Individual stage ranges do not establish
  supported combinations. Two metadata-query failures also remain; four failed
  guards, zero executions/new warnings, four truncated failure-log snapshots.
  Exact plans confirm CUDA99KiB opt-in versus Vulkan48KiB, not its latency cost.
  No stage/engine/tile sweep or timing follows. Installed9.24 lacks the latest
  memory-limit attribute; no invented enums or silent reference upgrade.
  Do not infer legacy execution from serialized frontend=false. Explicit async-copy/wait/event still
  requires Kernel, not Vulkan Shader. Source width does not prove async lowering.
  World training remains30.20ms; dependencies/host targets remain distinct.
- **Valid-value family attribution passes:** [result](docs/results/2026-10-15-convolution-value-bypass.md).
  Replacing all23 convolution products with exact saved-output copies reduces
  fixed-world GPU34.191→14.282ms,58.23%, with original allocations/consumers/
  dependencies. This resolves the invalid-intermediate problem of old masks,
  not ordinary full-update performance or pure kernel latency. Copy-only2.005ms
  includes all original barriers and is not a subtractable/additive saving.
  Initial/first/before/after292 tensors, targets/reports, live-boundary hashes,
  plan/pipelines/groups and full outputs pass. Two guards/zero new warnings;
  original binding-contract smoke remains failed, with separately declared
  source repair. One initializer update,324 full-world/80 copy-only replays;
  CPU whole-object audit OOM and corrected streaming audit stay. No worker.
- **Encoder/decoder attribution passes:** [result](docs/results/2026-10-15-convolution-role-cost.md).
  One valid-value four-condition comparison measures fixed-world34.216ms normal,
  27.285ms encoder substituted,21.142ms decoder substituted,14.280ms both.
  Symmetric conditional differences6.896/13.039ms, interaction.069ms; about65%
  of this intervention belongs to RGB reconstruction, absent from CDP. Not pure
  kernel costs, ordinary learning gains or measured CDP throughput. Fresh outer
  conditions remain within.071% of the predecessor; all512 samples retained.
  Four exact live/full-output conditions, three292-tensor pairs, first targets/
  reports, complete plan/pipeline/group semantics and1,157 source seals pass.
  One guard/zero warnings,647 fixed-world replays/one initializer/zero actions;
  CPU audit67.01s/77.2MiB, four selector/shader tests and Clippy pass. No worker.
  Source review confirms CDP already has exploration projection reuse/hidden
  centering; its old18-way first-projection cost is obsolete. Mixed's separate
  posterior predates BF16 backward/staging support.
  No decoder-only CDP claim, small tile/iterator sweep, new subset allocation
  or automatic promotion. RGB/JAX remains a separate unchanged comparison.
- **CDP policy cost recheck complete:** [result](docs/results/2026-10-15-cdp-precision-recheck.md).
  Same native99f8aab0/checkpoint/B32/T16 recipe: Mixed→BF16 full updates
  78.206→50.122ms,35.91%; posterior+world49.788→21.715ms, imagination23.94ms
  unchanged. Historical Mixed selection predates BF16 derivatives/staged reuse;
  both mechanisms change together, not an isolated shader attribution. No new
  kernel/rebuild. Four guards/one warning/592 synthetic updates/zero actions;
  exact346 initials, finite states/reports and each arm's own24-report/first-
  outputs/tensors smoke prefix pass. All454 source seals pass; service finishes
  05:14:52 UTC in92.70s, no worker. CPU moment-name mislabel is corrected without
  GPU repeat:130 actual parameters differ by.0001757 relativeL2 after one update,
  .05677 after272; moments separately reported. Not numerical or learning parity.
  The scoped qualification below now passes. No timing repeat, precision sweep
  or learner restart. RGB/JAX2.04x and the broader root-cause goal remain open.
- **BF16 CDP boundary/staging qualifies:** [result](docs/results/2026-10-15-cdp-bf16-qualification.md).
  Eight independent projection/cosine cases pass5,585,254 values, explicitBF16
  rounding and unchanged gates. Actual12M world whole/staged pairs match all86
  raw gradient tensors/14,688,256 elements, including changed late targets,
  NaN-poisoned prefix isolation and original restoration. Three346-tensor pairs,
  first targets/reports, unchanged state/plans and1,200 seals pass. Original
  in-memory-versus-parsed JSON assertion remains failed: all5,618,720 underlying
  F32 values were exact; separate observer repair/CPU regression passes with
  both files parsed identically. CPU extractor whitespace/Clippy failures stay.
  Two passing/one failed guard, one warning,13 raw backwards/two initializer
  updates including failure, zero game actions. Service finishes05:48:33 UTC;
  no worker. No full-model independent oracle, learning parity or promotion.
  Current CDP imagination/targets23.94ms exceeds posterior+world21.72ms.
  Its existing ensemble workload is attributed below, separately from RGB/JAX.
  Do not resurrect the old18-way projection or assume decoder cost applies to CDP.
- **Current CDP imagination attributed:** [result](docs/results/2026-10-15-cdp-imagination-work.md).
  Exact saved-bonus substitution changes fixed complete imagination20.771→
  8.705ms, conditional12.066ms/58.09%. All seven outputs, input/parameter state,
  three346-tensor pairs and plan/pipelines/groups match; independent versioned
  closure/source audit passes.90 extra ensemble dispatches/12 products add
  64.42G nominal MACs; not pure kernel cost, DRAM traffic or full-update gain.
  One guard/two warnings,325 fixed replays/one initializer/zero game actions;
  service finishes06:11:11 UTC, no worker. Three CPU tests/Clippy pass.
  The missed four repeat/transpose/typed-normalization chains are now fixed
  by the complete-update comparison below; original attribution stays retained.
- **Repeated-row composition qualifies:** [result](docs/results/2026-10-15-repeated-row-composition.md).
  Complete CDP50.151→47.728ms,4.83% less; imagination/targets23.873→21.490ms.
  RGB54.129→54.245ms is effectively unchanged, still2.04x JAX. Exact272 metric
  reports and initial/first/final346 CDP/292 RGB tensors pass. Eight copies and
  1.132GB logical writes removed;599 groups/170 BF16 matrices/205 typed scalar
  kernels and logical buffer inventory unchanged. No VRAM/utilization claim.
  Signed-probe qualification stays failed in both compilers:1.568M exact paired
  values, all forward/raw-gradient arrays pass, only near-zero scalar relative
  gates fail at1.013e−10 absolute error. Separately declared conditioned probe
  passes5.249M F64 values/arm and exact pairs with unchanged tolerances.
  Eight passing/two failed guards, five warnings,1,136 synthetic updates,
  58 raw backwards/58 forward-only passes, zero actions. Service finishes06:59:59
  UTC; no worker. Build-cache collision and unrelated test Clippy failures stay.
  Differing same-name source copies need distinct Cargo target caches and
  old/new compiled-plan assertions, not binary hashes alone.
  Preserve barriers/rounding/visible values/fused metadata reads/source versions.
  No exploration change, row/geometry sweep or production/learning promotion.
- **Explicit image-to-matrix feasibility rejects:** [result](docs/results/2026-10-15-convolution-matrix-feasibility.md).
  All23 products' patch generation alone costs34.348ms, above the declared15ms
  gate, even excluding NHWC preparation/GEMM/output scatter/reduction. No full
  lowering, shape/chunk sweep or integration. All6.647M exact small-case values,
  47,288 actual samples and two guards pass; zero new warnings/actions/updates.
  Service finishes07:40:37UTC in15.403s; no worker. Warning-schema audit repair
  retains the original source/report, no GPU repeat. Logical18.119GB writes
  are not DRAM traffic or a universal lower bound for other implementations.
  CPU proof of intra-kernel halo reuse covers all15 forward/dX occurrences,
  20.070M exact addresses; unchanged staging/output scratch fits47,104/49,152B.
  Keep original32/64 tiles, K order, rounding and scatter; segmented output
  scratch is unnecessary and not selected. Eight dW products/compensation stay.
  Logical image requests9.461->1.642GB are not measured traffic or latency.
  The complete-family trial below now rejects this particular reuse scheme.
- **Fixed-geometry halo cache rejected:** [result](docs/results/2026-10-15-convolution-halo.md).
  All15 forward/dX occurrences cost13.706->18.598ms,35.69% slower, including
  original transforms/output handling. Unchanged math/tiles/K order and exact
  503,808 F64 outputs/257,949,696 B512 paired outputs pass; no full-update
  trial, selector/geometry sweep or integration. Only one case improves3.2%.
  Six guards/two warnings,3,952 operation calls/8,720 operation dispatches,
  zero actions/updates. Timing finishes08:42:42UTC; no worker remains.
  Original B1 compile selects different kernels: its two passing GPU guards
  do not qualify the family. Failed CPU coverage review and two build-receipt
  path failures stay; separate declared exact-schedule repair passes.
  Distinct cache/source receipt and old/new pipeline assertions are required.
  Fewer logical image loads do not establish lower DRAM traffic or better
  feeding; added shared/address/preload/resource costs are not separately
  measured. Two kernel sample series shift sharply, without trimming any;
  latest complete RGB first/last32 means54.260/54.198ms are stable. No cause
  for the kernel shifts or explanation-away of the2.04x gap is established.
- **Latest complete-update attribution passes:** [result](docs/results/2026-10-15-whole-update-attribution.md).
  Fresh ordinary54.113ms; coarse55.150ms,+1.916% instrumentation. GPU-pass
  union47.699ms, uncovered7.453ms; cached programs46.484ms, means.059ms and
  optimizer.464ms. Helpers/host work alone cannot explain the residual2.03x.
  Pass coverage is not SM utilization; readback/submit spans contain GPU waits.
  All272 reports/first targets/initial-first-final292 tensors and paired plans/
  pipelines/groups pass exactly. Two guards/no new warnings/2,068 seals,
  544 synthetic updates/zero actions. GPU finishes09:10:01UTC; no worker.
  Correct the sealed declaration's chunk description in the result: two prefix
  plus three backward chunks, not one plus four. Timed chunking adds three
  intermediate waits; same setting does not preserve ordinary overlap. Original
  CPU receipt/module failures stay; no GPU retry. Instrumentationcc11418e is
  not promoted. Current world graph matches prior valid-value attribution;
  its19.909ms conditional convolution difference is not an additive saving.
- **Current B32 JAX execution control passes:** [result](docs/results/2026-10-15-jax-command-buffer-b32.md).
  Command buffers on/off: core26.550/29.042ms, inclusive27.223/29.505ms.
  Exact272 reports/288 initial-final tensors,940 static fusion computation
  bodies/112 custom-call configurations/physical shapes/reachable heavy work
  match, including retained control. Native54.113ms remains1.83x slower even
  without JAX graphs; graphs are not the dominant B32 residual. Historical
  JAX26.632ms was a core boundary, not caller-inclusive; no new native speedup.
  Two guards/two warnings/82 seals,544 synthetic updates/zero actions. GPU
  finishes09:42:04UTC; CPU review09:47:43UTC, no worker. Original serialized
  HLO failure remains: independent parameter order and JSON key order only;
  ten CPU regressions qualify semantic comparison without numerical relaxation.
  No GPU repeat. The following closed-region diagnosis narrows the image work.
  Do not revive invalid-value masks, failed sparse-timing attribution, or start
  another matrix/tile/precision/flag sweep. No automatic learning promotion.
- **Connected image attribution passes:** [all outputs](docs/results/2026-10-15-spatial-value-cost.md)
  and [closed boundaries](docs/results/2026-10-15-spatial-boundary-cost.md).
  Copying every intermediate hides materialization: non-heavy image substitution
  saves.530ms there,4.173ms with internal writes omitted. Replacing image+conv
  with exact escaping values reduces fixed-world34.195→6.564ms;46 outputs/7.014MB
  escape versus344/3.154GB total region outputs. Not DRAM traffic, a pure cost
  decomposition or attainable full-update saving. Interaction−3.552ms remains.
  Independent version proofs, poisoned discarded values, full/live outputs,
  exact292 tensors and full plans/pipelines/groups pass. Three guards/no warnings,
  two initializer updates/1460 fixed replays/zero actions; CPU/lint failures and
  controller-only snapshot repair remain, no GPU retry. Last GPU finishes10:35:20,
  CPU review10:38:05, no worker. Production/complete gap unchanged. The following
  audit identifies the corresponding existing JAX/native image interfaces.
- **Image-layout audit complete; packing-limit repair not promoted:**
  [result](docs/results/2026-10-15-image-layout-contract.md). All23 JAX convolution
  interfaces are BF16 NHWC/KRSC. Native returns NCHW/F32 derivatives, then casts/
  repacks;14 products already consume packed inputs. Eight version-proven
  duplicate packs write83.886MB logically, not measured traffic or savings.
  Four products fall back solely at65536 packing groups. Bounded2D dispatch
  qualifies403.575M structured F64 values/134.525M CPU rechecks, but complete
  RGB54.117->52.963ms saves2.13%, below the declared3% gate; CDP47.677->46.207ms
  saves3.08%. No retime, subset or promotion. Exact retained-control272 metric
  reports/initial-first-final292/346 tensors pass; candidate trajectories differ.
  Full timing plan/pipeline exports were disabled: that gate stays incomplete,
  not replaced by source/precision-summary checks. Original CPU time-field
  comparison failure and observer-only repair remain. Six guards/four warnings,
  1088 synthetic updates/48 standalone calls/zero actions; no worker. Keep076f3135.
  Next prove a connected producer/consumer lowering on CPU: retain required
  tapes/rounding/external layouts, write required converted/expanded packed
  views directly and share identical versions across forward/dX/dW. Count all
  conversions/scratch; no new GPU allocation, Naga project, standalone family/
  geometry sweep or learning restart implied. Layout alone is not cuDNN feeding;
  the complete parity goal and non-image costs stay open.
- **Connected image views qualify:** [complete result](docs/results/2026-10-15-image-views.md).
  All253,755,392 proposed addresses/7,352 surviving reads/892 protected buffers
  pass;13 modified image kernels remove31 dispatches/876.610MB logical writes.
  Logical bytes are not measured traffic. Isolated shared typed implementation is in
  `runs/image-views-20261015.EblKUBZN`; native67559ce5 finishes CPU build/Clippy,
  exact independent-plan selection, protected/multiple-writer checks and144
  existing/432 new typed shader variants. Numerical24-update pairs and full
  timings pass: RGB54.205->51.945ms,4.17%; CDP47.699->47.112ms,1.23%. All272
  reports/targets/initial-first-final292/346 tensors match exactly, including
  retained controls and qualification prefixes. Full programs independently
  reconcile13/31 RGB and5/11 CDP changes/removals; non-world programs unchanged.
  Eight guards/seven warnings,1184 synthetic updates/zero actions. Timing ends
  12:30:39UTC, CPU review12:31:48; no worker. Keep the isolated candidate, not
  a parity/learning claim. Preserve Clippy failure/source and explicit-reference
  repair. No convolution selection, precision, model or production change.
  The interface review below now identifies a separate compiler restriction;
  no layout/geometry sweep, new GPU allocation or Naga project implied.
- **Feeding-pipeline CPU feasibility passes:** [result](docs/results/2026-10-15-feeding-pipeline.md).
  Nine typed Vulkan scheduling modules validate; all current23 convolution
  selections reconcile. Two operand slots plus output scratch fit BF16/F16
  at12/36/34KiB; packed F32 needs52/50KiB and exceeds the48KiB explicit limit.
  Two loading/four computing subgroups, full192-invocation barriers and no spin
  loops pass1,645,920 schedule events/five negative tests. This is not actual
  convolution indexing/numerics or measured overlap. Naga a97559b rejects the
  legal subgroup-uniform consumer branch for all three types; whole-workgroup
  controls pass and divergent-lane negatives reject. New upstream memory-model
  branchc0b23d16 has the identical analyzer; no compiler edit/commit. Preserve
  the three CPU reader/probe failures and repairs. Zero GPU calls/actions/updates;
  finishes12:55:34UTC, no worker in that CPU allocation. The actual complete-family
  diagnostic below precedes compiler changes; no
  unsafe uniformity bypass, fixed-subgroup assumption without enforcement or
  favorable tile subset. Current complete gap stays1.91x and goal stays open.
- **Actual feeding family rejected:** [complete result](docs/results/2026-10-15-feeding-family.md).
  All23 products/20 contracts retain32/64 tiles,K32,B512 and original splits/
  compensation. Native/GLSL serial/pipeline complete GPU costs22.412/24.418/
  46.671ms; queued wall22.472/24.483/46.749ms. Aggregate calibration+8.95%
  passes10%, not per-kernel calibration. Every contract regresses versus both
  controls; pipeline+108.24% versus native,+91.13% versus GLSL. No Naga project,
  favorable selector, loader/tile sweep or full-update trial.60 numerical/60
  timing guards,4,035,648 F64 comparisons and1,033,149,184 exact value pairs pass;
  31 warnings,8,940 standalone calls, zero actions/learner updates. Numerical
  fixtures use two active images with full512 grids/splits; old cancellation
  failures remain.180 SPIR-V variants/4.165M addresses/113,664 ownership checks
  pass. Original15 guards/45 calls/25 warnings plus an interrupted sixth native
  cohort remain: local IDs did not guarantee subgroup mapping. Corrected subgroup
  IDs pass720 assignments; old formula fails288. Missing logging dependency and
  early CPU-audit receipt race remain; no GPU repeat for those CPU repairs.
  Timing ends13:50:25UTC; full independent audits pass, no worker remains.
  Current whole gap stays1.91x; this changes neither production nor Atari.
  The following CPU output audit addresses one prerequisite for larger reuse;
  it does not revive this rejected schedule. Keep67559ce5.
- **Output-storage CPU feasibility passes:** [result](docs/results/2026-10-15-output-contract.md).
  All23 occurrences/280.986M outputs/378.339M fragment positions reconcile.
  Nine direct-store typed SPIR-V modules validate with zero explicit output
  scratch; nine existing-Naga fragment-store IR variants use4KiB, two barriers
  per phase.1.442M bit-rounding checks/seven negatives pass; no GPU execution.
  Padded two-slot operands fit3/9/13 of22 known library tiles with full/fragment/
  direct output; unpadded payload with fragment/direct fits18/22 and22/22.
  These are storage envelopes, not conflict-free layouts or actual library-stage
  replication. Four larger dW tiles also imply384 logical F32 values/compute
  lane with retained compensation; not measured registers/spilling. The old
  compensation-removal.89% null/numerical failure stays closed. No Naga feature
  project: the fragment alternative uses existing operations. Original missing
  rustc PATH failure/source retained; absolute-path repair and independent audit
  pass. Zero actions/updates; no worker. The resulting complete policy below
  fails; no geometry/precision sweep, relaxed gates or automatic promotion.
- **Compact-channel/reuse policy rejected:** [full result](docs/results/2026-10-15-reuse-policy.md).
  All23 products cost22.327/24.313/27.177ms native/GLSL serial/candidate:
  21.72% slower than native;18/20 contracts regress. Aggregate calibration+8.89%
  passes10%, not per-kernel calibration. Keep all samples including CNN0 GLSL's
  large transient.120 guards/no warnings,4.036M F64 values/1.033B paired values;
  candidate numerical15/20 exact, five reordered cases pass unchanged gates.
  Timing ends16:10:46UTC; independent audits pass, no worker.8,940 standalone
  calls, zero actions/updates.120 offline modules/104.808M CPU addresses pass.
  CPU source accounting finds33.93% fewer staged positions but more output
  barriers and reintroduced in-loop coordinate work; not hardware counters or
  proven slowdown attribution. Preserve initial preflight and constant-branch
  observer defects/repairs. No favorable selector, sweep or integration.
  Whole gap stays1.91x; keep67559ce5. Production/Atari stay stopped.
- **Actual native counters now qualify:** [report](docs/results/2026-10-16-native-counters.md).
  User-supplied Perf SDK2026.3 works directly on5080/580.178.04 despite absent
  standard KHR counters. All20 contracts/23 occurrences pass; native CNN1 dX/dW
  execute15.94x/9.04x retained full-JAX warp instructions, with fewer tensor
  instructions. These are native counters, not the old CUDA proxy or a latency
  ratio. Nine scalar-staged occurrences execute10.214G instructions versus1.596G
  for14 packed occurrences. The two derivatives show<6% peak DRAM throughput
  and122M/158M shared-bank conflicts; no isolated removable-latency claim.
  21 guards/no warnings,680 replay passes/1680 standalone calls/zero actions or
  updates. All58 ranges/2378 metrics reevaluate offline;258.287M outputs, full
  plans/pipeline keys and1794+29 source seals pass. Captures finish04:39:04UTC,
  audit04:39:52; no worker. SDK's device API-version flag is false but all calls
  succeed; no official-version-support claim. First preflight bridge/decode-wait
  correction stays; no GPU failure/retry. Artifacts in
  `runs/native-perf-sdk-20261016.ZVAi5Ubl`, production/67559ce5 unchanged.
  The following word-staging result tests this contract with existing tiles/
  K order/compensation/rounding/interfaces. Padding alone is not a promised fix.
  No geometry sweep, rejected-policy revival, favorable selector, Naga project
  or learning restart.
  Root-cause/parity goal stays open; no missing-SDK or usage-approval blocker.
- **Typed word staging qualifies:** [result](docs/results/2026-10-16-word-staging.md).
  Complete RGB51.763→48.068ms,7.14%; CDP46.929→45.505ms,3.03%, with exact272
  reports/targets/initial-first-final292/346 tensors and retained24-report prefixes.
  All plans/constants/pipelines/groups and actual shader changes pass; four
  shared modules cover nine/four RGB/CDP occurrences. Complete23-product family
  22.493→19.112ms,15.03%, including one12.1% regression and unchanged14 packed
  occurrences. CNN1 dX instructions1.554→.841G, conflicts122→22M and latency
  2.155→1.119ms; dW1.208→1.092G,158→27M and2.025→1.841ms. Tensor counts stay.
  dW uniform-pipe315.42→.46M but ALU507.57→765.88M suggests lost warp-uniform
  address work; conditional source evidence, not SASS attribution. Both still
  exceed JAX warp instructions8.63x/8.17x. Preserve all samples, typed-zero,
  offline-emission observer and direct-Blade-path CPU failures/repairs.
  124 passing guards/five warnings,2.690M F64/774.862M exact pairs,116 ranges/
  4,756 offline metrics;1,266/1,775 seals.9,320 standalone calls/1,360 counter
  replays/1,088 synthetic updates, zero actions. GPU ends05:53:42UTC, CPU review
  05:54:22; no worker. Keep isolated native`834241b6` under
  `runs/convolution-word-staging-20261016.ssN5oC6N`, with
  `MEGANEURA_CONVOLUTION_WORDS=1`, image views/device means/cached recording.
  Next CPU-proof packet addressing that preserves shared calculations, including
  subgroup/operand-layout contracts; no implicit contiguous-lane assumption,
  geometry/selector sweep, Naga project or automatic allocation. Whole parity,
  non-convolution work and learning quality remain open. No production/Atari promotion.
- Preceding isolated candidate`67559ce5` with `MEGANEURA_IMAGE_VIEWS=1` and
  `KINDLE_DEVICE_MEANS=1` under `runs/image-views-20261015.EblKUBZN`;
  `076f3135` under `runs/repeated-row-composition-20261015.dDTCIIip` is its
  control. `99f8aab0` remains the preceding
  control under `runs/categorical-gpu-20261015.lkZAlUAS`. Native`28864210` is the
  [general-ordering control](docs/results/2026-10-14-convolution-scheduling.md).
  Current isolated backend sources are in the word-staging study; the ordering
  control remains `runs/convolution-scheduling-20261014.BJ5AVZFt`.
  Do not rebuild unchanged code for docs or confuse neighboring user work with
  this candidate. Latest October15 upstream: `f648a453`/`e7c3f7b`; Meganeura's
  two new search/attention exploration fixes do not change ordinary convolution.
  No runtime refresh or historical repin from this review.
  CI status belongs on PR31.
- **Closed diagnostics stay closed.** No native assembly-query retry (the
  retained query returned no representations), failed Nsight Graphics injection
  retry, unsafe barrier weakening, or revived CUDA/geometry/hint/row-only sweep.
  Compiler local-memory statistics also fail a scalar control; do not mask
  their bits and claim measured spilling. Timings, MAC counts, command counts
  and logical byte extents are not SM utilization, DRAM traffic or additive
  savings. Posterior/world plan labels alias one staged session; never double
  count them. Keep full failures, tails, sources and gates.
- Use persistent automated bounded queues, review failures before follow-up,
  and report meaningful boundaries. No default GPU approval gate while healthy.
  Check upstream before diagnosing already-fixed issues. Only the user merges.
  Naga's repository requires explicit commit permission; the previous BF16
  permission does not automatically cover new compiler features.
- **Five-game quality and budget goals remain open.** The completed centered
  CDP12M500k comparison improves Breakout/Pong/Qbert reward means over1M, not
  every game/milestone: Freeway2017 regresses and Boxing's advantage is not
  consistent. All15 larger learners cost58h22m14s,3.56x the1M control; none
  meets its long-run reference. [Combined results/videos](docs/results/2026-10-09-cdp-freeway-capacity.md).
  Keep12M as quality candidate and1M as fast control, not universal promotion.
- [Qbert2M](docs/results/2026-10-11-cdp-qbert-budget.md) improves frozen mean
  to7,367.01, but no seed passes the full historical/long-run gate. Frozen
  reward/terminal/value diagnostics and the rejected greedy-action alternative
  remain in the roadmap. Pong2M was stopped by the user at189,576 actions/
  47,333 updates of seed1009; no completed learner, other seeds/evaluations
  cancelled. The later Freeway performance screen also stopped at39,568 actions/
  9,831 updates. Preserve partials; neither is an uninterrupted-lifetime resume.
- Budget claims must retain every released seed, historical pins, extra offline/
  interrupted compute and the declared last10% long-run target.200k actions
  are~800k emulator frames, not Atari100k or200M-frame Atari57. Published online
  curves and our frozen controls are different protocols. No matched five-game
  CDP/RGB compute-saving claim exists. After Atari: video priors, mind-games
  vkQuake2 then TMNF, small GOG/Wine panel, held-out transfer/retention, then
  swarms. No concurrent learner service now.

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
