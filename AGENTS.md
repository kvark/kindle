# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **Active goal: understand the remaining JAX gap and establish how to reach
  similar Meganeura performance.** It remains open. Latest isolated complete
  RGB12M/B32/T16 is46.199ms versus inclusive JAX27.223ms,1.70x; B8 retains22.780/12.737ms,
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
  The following packet-axis trial tests shared calculations without assuming
  contiguous subgroup membership and rejects that implementation. No geometry/
  selector sweep, Naga project or automatic allocation. Whole parity,
  non-convolution work and learning quality remain open. No production/Atari promotion.
- **Column-major packet axis rejected:** [result](docs/results/2026-10-16-packet-axis.md).
  All23 products19.089→21.217ms,11.15% slower; seven of nine changed contracts
  regress. No full-update trial, selector or production change. dW uniform-pipe
  work returns.46→372.22M and ALU falls765.88→569.86M, but total instructions
  rise17.06% and time1.01%; shared conflicts stay26.9M. dX instructions rise
  12.48%/time27.08% despite42.65% less L2 traffic; conflicts remain21.5M.
  Tensor counts stay exact; uniformity/occupancy/traffic alone are not the fix.
  The eight-versus-two reduction cursors and orientation change together, not
  an isolated SASS attribution. CPU local-ID ownership is permutation-safe;
  no implicit subgroup mapping or compiler feature.304 modules/5.690M addresses/
  10.492M cursor transitions pass.120 guards/22 warnings,2.690M F64/774.862M exact
  pairs,116 ranges/4,756 offline metrics/1,274 seals;9,320 standalone calls,
  1,360 counter replays, zero updates/actions. GPU ends06:38:36UTC; CPU06:40:19,
  no worker. Retain the exploratory reader syntax failure/repair. Keep834241b6
  and1.77x complete gap. The following CPU audit reconciles remaining programs
  and host boundaries before selecting another intervention; no sums
  of incomparable timing regions, renewed layout/tile sweep or automatic GPU run.
  Convolution feeding and whole parity remain open, Atari stopped.
- **Non-convolution contract audit passes:** [result](docs/results/2026-10-16-nonconvolution-contract.md).
  Current non-world plans/pipelines match the coarse capture; all28,064 JAX
  kernels/eight updates and940 fusion bodies/112 library configurations reconcile.
  Training prior is unnecessarily unrolled:144 products versus9 batched JAX,
  identical1.107G nominal MACs. Versioned gradients and both posterior closures
  prove these products are outside the recurrent dependency. This explains135
  of159 extra matrix products, not their latency or the whole gap. Imagination
  recurrence has180 products in both stacks;44.866/44.890G MACs differ only by
  action padding. Native8.740ms pass includes heads; JAX2.852ms recurrence kernel
  sum does not, so no3x recurrence claim. Heads already batch; grouped matrices
  already use cooperative hardware. Target/replay work and tape reuse stay distinct.
  First observer's loop-scope leak stays invalid;18 regression checks/six original
  negatives and independent SQL/body audits pass.51 seals; CPU ends07:13:16UTC,
  zero GPU calls/builds/updates/actions. Next isolate training-prior/KL batching,
  preserving masks/free nats/rounding/gradients/staged late targets; qualify before
  one complete comparison. No automatic GPU allocation, convolution sweep,
  production or Atari promotion. Keep834241b6; the1.77x parity goal remains open.
- **Training-prior batching is implemented, not qualified:** [result](docs/results/2026-10-16-prior-batching.md).
  Standalone12M products144→9, dispatches1181→168; all eight paired output/raw-
  gradient comparisons pass. Independent27.565M-value review fails14 array
  checks in both arms; some pointwise gates fail despite relativeL2<.000350.
  CPU F32 normalization/activation diagnosis reduces748 failed logit envelopes
  to408, not zero. Rounding sensitivity is established, full cause is not.
  Keep every failure/gate/source; no relaxed rerun or replacement cohort.
  Actual whole/staged checks and full timings are unrun. One guard/no warnings,
  16 raw backwards/zero optimizer updates/actions;2,271 seals, no worker.
  Preserve the read-method/build and length-one-config failures/repairs.
  Isolatedaca80566 remains unqualified; keep834241b6 and1.77x complete gap.
  Resolve the finite-precision reference before progressing this candidate.
  New upstream Blade4845d652 supplies workgroup matrices/BF16 capability alignment.
  No production/Atari promotion or convolution sweep.
- **Upstream workgroup contract reviewed:** [result](docs/results/2026-10-16-workgroup-contract.md).
  One properties-only guard/no warnings; all seven optional features advertised,
  but Blade enables two and Naga lacks tensor-addressing IR. All23 current shapes/
  resource bounds and six negatives pass;72 seals. Historical-index reader failure
  stays; full membership repair passes without GPU repeat. No device/dispatch/update.
  The rejected October15 family already used tensor addressing/workgroup matrices;
  new upstream exposure does not revive it or justify a compiler project. Keep
  834241b6/1.77x, resolve prior-reference discrepancy; no worker/production change.
- **Prior forward discrepancy explained:** [tape result](docs/results/2026-10-16-prior-tape.md).
  Two unchanged forward prefixes match retained logits/plan/pipelines;2.097M local
  matrix/bias values are exact. All1.049M normalization/activation values fit
  checked Vulkan precision contracts; six SiLU values lie within.604F32ULP of
  BF16 midpoints and round differently from exact SiLU. Not an end-to-end F64
  pass; original748 logit failures remain. Gradient failures remain unexplained,
  batching/world checks/timing unqualified. One guard/no warnings,106 dispatches,
  zero raw backwards/updates/actions;2,477 seals, no worker. Preserve CPU reader/
  JSON failures; exact constant-bit repair and seven negatives pass without GPU
  repeat. Next diagnose raw-gradient rounding, not tune final-output thresholds.
- **Prior-gradient discrepancy diagnosed:** [result](docs/results/2026-10-16-prior-gradient.md).
  One original168-dispatch backward preserves every output/40 raw gradients.
  KL F32 differences<=1.932e-10 change50 BF16 derivatives, then115/403 in earlier
  layers. All36 local checks/7.686M comparisons pass unchanged gates; observed
  operands explain all69 first-variant failed weights. Second-variant source-F32
  hypothesis clears both arms; not bit-exact Vulkan or end-to-end F64 qualification.
  Original14 failed arrays stay.49 finite differences/seven negatives,2,669 seals,
  one guard/no warnings/one raw backward/zero updates/actions; no worker.
  Preserve serde-array compile and35-versus36 audit failures/repairs. No backend
  rebuild. Next separately qualify actual whole/staged world gradients before
  timing; keep834241b6/1.77x, not unqualifiedaca80566. No Atari promotion.
- **Actual-world prior batching fails:** [result](docs/results/2026-10-16-prior-world.md).
  Eight whole/staged pairs, poisoned late-input isolation, restoration and all
  initial/final292/346 tensors pass exactly. All56 outputs pass, but113/308
  gradient arrays/903,879 values fail pointwise gates; all relativeL2<=.007095.
  Independent52.326M-value audit reproduces failures. Four guards/no warnings,
  24 raw backwards/zero updates/actions;3,455 seals, no worker or timing.
  Disabled compiled control matches; each workload removes135 products/982
  dispatches, not barrier groups. First prior product changes8-way F32 partials/
  SumRows/BF16 to one tiled/BF16 product. All34 paths preserve final conversion;
  changed accumulation is a candidate cause, not full causal attribution.
  Next inspect saved real prior inputs/weights and both reduction contracts.
  No relaxed gates, qualification replacement, K/tile sweep or timed promotion.
  Keep834241b6 and1.77x; this native-pair failure differs from the F64 diagnosis.
- **Real-input forward divergence attributed:** [result](docs/results/2026-10-16-prior-real-inputs.md).
  First prior products differ135/163 BF16 values, yielding8,028/3,549 changed
  RGB/CDP logits. Exact serial-product substitution makes all3.146M subsequent
  path values, including524,288 logits, exact.102 actual-world contracts and
  six closed dependency paths pass; no KL/backward seed or posterior placeholder
  executes. All2.097M partial/F64 checks and1.573M normalization/activation values
  each pass their existing gates/contracts. CPU dot16 predicts only15/17 changes;
  it is not bit-exact GPU arithmetic or independent partition/tiling attribution.
  One guard/no warnings,568 forward dispatches/two copies/zero raw backwards,
  updates/actions;2,855 seals, GPU ends09:46:51UTC, no worker. Independent7.340M
  paired-value audit/four negatives pass. CPU import/schema/status failures stay;
  repaired readers match full JSON without GPU repeats or raised limits.
  The113 whole-world gradient failures remain. Next separately attribute them
  with the same first-product intervention, retaining ordinary controls and both
  late-target conditions; no timing, relaxed gates, sweep or promotion. Keep834241b6.
- **Whole-world first-product attribution complete:** [result](docs/results/2026-10-16-prior-world-intervention.md).
  Substitution reduces pointwise failures903,879→60,682, but78 arrays still fail.
  All56 outputs are bit-exact;32 prior-parameter comparisons pass, maxrelativeL2
  8.914e-7. Residual failures are posterior/recurrent/encoder, not those prior
  parameters. Ordinary/split controls match104.652M values; independent156.977M
  comparisons and four292/346-tensor checkpoint pairs pass. Two guards/no warnings,
  14 raw backwards/four copies/zero updates/actions;5,674 seals, GPU10:06:37UTC,
  no worker. Only the test observer changes; backend library bytes stay exact.
  CPU32-join audit finds serial's six-input adjoint expression becomes a stored
  F32 branch plus four-input join;4,096 strict-F32 fixtures/two negatives pass.
  This is a materialization difference, not observed contraction/rounding or equal
  dynamic operands. Preserve the Clippy failure/repair. Original qualification
  stays failed, no timings or promotion. Keep this numerical follow-up open;
  next prioritize matched native/JAX imagination boundaries before another GPU
  allocation. Native8.740ms includes heads, JAX2.852ms does not. Keep834241b6/1.77x.
- **Imagination boundaries audited:** [result](docs/results/2026-10-16-imagination-boundaries.md).
  Native802 dispatches split20 parameter conversions/722 recurrence+features/
  43 rollout assembly/17 heads;180 recurrent matrices match JAX arithmetic
  apart from action padding. Fifteen private RSSM input chains use105 dispatches
  and747.110MB logical writes versus125.829MB final packed inputs;62.915M exact
  address checks/182 protected buffers pass. JAX uses15 actual fusions. Not
  measured traffic, speedup or GPU qualification. Stacking already uses balanced
  trees, but separate15/16-state outputs duplicate assembly;43 copies write
  647.049MB logically. JAX recurrence630 kernels+45 copies gives2.907ms activity/
  3.147ms span, not comparable directly with native8.740ms including heads.
  All360 retained copies/63 seals/six mutations/three label regressions pass;
  preserve original SQL inline-versus-registered-label failure and repair.
  CPU only, no build/GPU/actions/updates; review before10:38:10UTC, no worker.
  Next separately declare valid-value native recurrence attribution, matrices
  versus surrounding execution, before choosing composition or matrix work.
  No numerical relaxation, prior-candidate timing, sweep or promotion. Keep834241b6.
- **Native imagination attribution passes:** [result](docs/results/2026-10-16-imagination-cost.md).
  Complete fixed imagination8.702ms; recurrent matrices substituted5.932ms,
  surrounding work6.639ms,both2.275ms. Conditional differences2.770/2.063ms
  interact by−1.594ms; not pure/additive kernel costs or an ordinary speedup.
  All six outputs/four conditions, three292-tensor pairs,176 protected hashes,
  first targets/reports and ten plan/pipeline identities pass.31 outputs/84.439MB
  escape the722-dispatch recurrence; original allocations/groups stay. Two
  guards/four warnings,3,906 copy-smoke words/35 executions,733 fixed replays/
  one initializer/zero actions; GPU ends10:57:19UTC, no worker.574 seals/eight
  CPU tests/Clippy and independent audit pass. Keep the observer BufferRef-Ord
  compile failure/repair. No GPU retry, new ordinary-update ratio or promotion.
  The following general copy/cast composition now qualifies across full RGB/CDP
  programs. Matrices/convolution feeding remain open. No prior timing or
  geometry/precision sweep follows from the attribution alone.
- **General private copy composition qualifies:** [result](docs/results/2026-10-16-copy-composition.md).
  Complete RGB48.133→46.199ms,4.02%; CDP45.669→43.939ms,3.79%. RGB/JAX1.70x.
  All272 reports/targets/initial-first-final292/346 tensors and24-update prefixes
  match exactly; controls drift+.1363%/+.3604%.47.082M raw-gradient values per
  comparison match original/changed late targets and NaN-poisoned preparation;
  18 checkpoint pairs stay exact.319.752M CPU addresses/11 Rust fixtures/615
  typed variants/Clippy pass.93/112 rewrites remove362/458 dispatches and
  1.190/1.207GB logical writes, not measured traffic.24/32 executed copy modules
  match actual CPU generation; all source versions, plans/constants and unchanged
  pipelines reconcile. World groups audited; non-world runtime groups unexported.
  Six helper-order permutations and three unexecuted extra modules explain the
  shader-catalog failures. Preserve first arity, lint, fixture, dependency, group-
  schema, shader-scope and occurrence-count failures; CPU repairs use no GPU repeats.
  Twelve guards/ten warnings,1,188 synthetic updates/16 raw backwards/zero actions;
  GPU11:51:32UTC, final CPU11:59:49UTC, no worker.3,044 timing seals pass.
  Keep isolated native`58aff452` under `runs/copy-composition-20261016.ABr4pCUH`,
  with `MEGANEURA_COPY_COMPOSITION=1` plus word staging/image views/device means/
  cached recording.834241b6 remains its preceding control. No F16 GPU, acting,
  learning or production promotion. The following audit/capture narrows the
  remaining convolution work; no copy/geometry/precision sweep or prior timing.
- **Local-memory/shared-write diagnosis passes:** [result](docs/results/2026-10-16-kernel-memory.md).
  All20 contracts/23 RGB occurrences have zero local traffic; a private-array
  positive control reads8.468MB/writes12.594MB while scalar/empty controls stay
  zero.99.711% of remaining shared-bank conflicts are writes:308.566M versus
  .894M reads. All nine word paths fit workgroups×(32×K-stages+96), including
  seven held-out cases; this is not source-site/latency attribution. No software
  reciprocal division occurs in those K-loop call closures; automatic bounds
  checks/core robust access are off already. All716/769 matrices and23/11
  RGB/CDP convolutions retain descriptors/keys; current shader bytes match captures.
  21 guards/13 warnings,777 counter passes,1,740 convolution calls/76 probe
  dispatches,258.287M exact retained values,61 ranges/3,050 offline metrics and
  159 seals pass. All instruction counts reproduce; zero updates/actions.
  GPU12:25:52UTC, no worker. Preserve first CPU SDK-enumerator compile failure/
  host-only repair. No backend rebuild, memory-stat masking or native assembly.
  Keep58aff452/1.70x. Next CPU-audit shared-store transaction/ownership contracts
  across existing typed generators, including staging/cooperative result handling;
  counters do not assign source sites or rule out compiler-managed shared storage.
  No padding/geometry sweep, rejected orientation revival, Naga project or
  automatic GPU allocation. Matrix/host/graph costs and parity remain open.
- **Shared-store contract audit passes:** [result](docs/results/2026-10-16-shared-store-contract.md).
  All1,519 actual heavy occurrences map to28 selected modules. Nine tiled matrix
  keys retain32 scalar16-bit shared stores despite vector global loads;342/716
  RGB and387/769 CDP matrix products carry91.04%/93.24% of nominal matrix MACs.
  This is not measured matrix instruction/time coverage.54 distinct contracts
  span both workloads.29 catalog modules emit SPIR-V, one explicitly unused;
  152 retained typed modules,6,144 ownership values and14 signature negatives pass.
  The conditional bank model matches the word kernels' slope, but legal subgroup
  permutations change it; simple scaling fails all11 packed contracts. All20
  measured payload/wavefront relations reconcile. The96 intercept stays
  unattributed; Naga preserves cooperative-store opcodes, not known driver lowering.
  Original CPU review substring failure and construction-specific repair remain.
  Zero GPU calls/updates/actions, no worker. Keep58aff452/1.70x. The following
  unchanged-kernel capture now covers this matrix path.
- **Tiled matrix counters observed:** [result](docs/results/2026-10-16-matrix-feeding-counters.md).
  All54 actual tiled contracts/729 occurrences plus11 checked controls have zero
  local traffic and exact expected tensor counts. Weighted RGB/CDP warp-to-tensor
  counts11.94/12.72; shared read/write conflict fractions32.52/64.17% and32.28/60.39%
  are transactions, not time shares or speed promises. Checked-vector reads are
  about12 conflicts/tensor versus tiled1; shapes differ, no alternative timing.
  65 passing/one failed guard, zero warnings,335.647M output/161.773M input values,
  6,500 offline metrics/1,249 seals pass;5,720 completed calls plus2–3 interrupted,
  zero updates/actions. GPU ends13:26:37UTC; no worker. Preserve the180s mapped-read
  observer timeout/partials, reviewed bulk-copy repair and CPU schema/dependency
  failures. Original exact shared total=read+write gate fails44/65, max.12737%;
  separate CPU review reports independent counters without a tolerance/GPU retry.
  The following CPU proof covers the proposed storage interface.
- **Typed tiled vector-storage CPU proof passes:** [result](docs/results/2026-10-16-matrix-vector-contract.md).
  All54 actual contracts/729 occurrences,468 typed requests and463 old/463 new
  module emissions pass;32 scalar stores become eight vector stores, not measured
  native instructions.3.834M producer/7.668M consumer/442,368 affine input checks,
  48 envelopes/seven negatives/1,885 seals pass. All selected cases fit9,728–27,648B;
  two larger transposed-B/scratch envelopes newly exceed48KiB. Integration must
  make the existing capacity accounting orientation-aware; no favorable selector.
  Non-staging source, barriers/K order/rounding and58aff452 stay unchanged.
  Original CPU reviewer96-versus48 count failure and count-only repair remain;
  zero GPU calls/builds/updates/actions, review13:57:45UTC, no worker. The complete
  comparison below rejects this implementation. No padding/geometry/precision
  sweep, prior timing, compiler feature or automatic promotion.
  Keep58aff452/1.70x and the broader goal open.
- **Complete matrix vector-staging family rejected:** [result](docs/results/2026-10-16-matrix-vector-family.md).
  All54 contracts/729 memberships: GPU RGB7.763→7.633ms,1.68%; CDP11.006→10.840ms,
  1.50%. Queued wall improves1.54%/worsens.16%; both5% time gates fail. Warp work
  rises7.18%/8.34% despite63.99%/50.92% fewer write conflicts. Tensor counts stay;
  all54 control warp counts reproduce. All20 NN contracts retain exact total LSU
  counts and unchanged scalar row geometry; ALU/FMA-pipe work increases. Static
  float opcodes stay;16 new constant4 divisions/module are not native source-site
  attribution or a new candidate. Nine-module/six-group CPU interpretation passes.
  108 guards/two warnings,665.846M integer/retained-F64-reference output checks,
  10,800 offline metrics/1,987 seals pass;87 nonadditive shared observations stay.
  9,504 standalone calls/3,996 passes, zero builds/updates/actions. GPU14:20:56UTC,
  interpretation14:39:01UTC; no worker. No integration, favorable selector, full-
  update trial or retime. Keep58aff452/1.70x. Next CPU-map actual retained JAX
  matrix interfaces/launch membership, especially the common imagined recurrence,
  before selecting counters. No shape-constant/layout/geometry sweep or prior
  timing. Production/Atari stay unchanged and the root-cause goal remains open.
- **Actual imagined matrix interfaces reconcile:** [CPU result](docs/results/2026-10-16-jax-matrix-contract.md).
  All12 roles/180 products match, apart from18→24 action padding; native/JAX
  44.866/44.890G MACs.20.934M physical addresses/seven negatives pass. Grouped
  layouts match physically, but JAX includes BF16 beta=1 additions while native
  adds bias outside the matrix; all JAX epilogues DEFAULT, not inferred ReLU.
  All12 graph nodes/1,440 actual launches reconcile2.105519ms kernel sum, not
  a native pass time. Individual HLO ownership is unresolved; no invented order-
  based layer timing. Native counters cover150/180 occurrences/seven contracts;
  the30 action products need their actual512-row shapes, not older checked cases.
  Six new Meganeura commits through12df95c add typed dense staging/tuning and a
  primitive accuracy policy. Upstream tiled blobcdb65862 is unchanged and lacks
  isolated typed/grouped branches; current checked staging is not identical to
  new upstream. No silent repin or retrospective graph-gate relaxation.
  Original CPU absent-file assertion stays failed; source-membership repair/
  19 seals pass, CPU14:51:56UTC, zero GPU/builds/updates/actions, no worker.
  The bounded full-JAX capture below covers the two missing native contracts.
  Retain aggregate-first padding/addend limits. No HLO selector invented
  from names/order, new kernel/geometry/layout sweep or learning. Keep58aff452/
  1.70x; the root-cause/parity goal remains open.
- **Actual recurrence counters reconcile:** [result](docs/results/2026-10-16-jax-matrix-counters.md).
  All180 products: native277.979M/JAX300.272M warp instructions,7.42% fewer native;
  tensor21.934/21.965M, zero local traffic in both. Convolution's large instruction
  overhead does not generalize to these matrices. Native read conflicts24.36x,
  total read wavefronts only1.0174x; neither is a latency attribution. Actual JAX
  captures all15 steps, native uses synthetic checked fixtures; action padding/
  grouped beta=1 boundaries remain. Exact272 reports/288 tensors/full HLO,
  all memberships and561,152 new F64/integer outputs pass. Three guards/no
  warnings,3,240 JAX passes/74 native passes/176 native calls/272 synthetic
  updates, zero builds/actions. GPU ends15:15:27UTC, CPU review15:18:11UTC;
  no worker. Successful transient-unit collection trips the completion observer;
  retain failure/source and separate unchanged CPU review, no GPU repeat.
  Current ordinary RGB caller intervals7.037/10.208/24.403/4.552ms sum46.199ms;
  they contain waits, not pure GPU work or equivalent JAX boundaries. The current
  complete trace below reconciles those boundaries. No generic
  matrix staging/geometry sweep, prior timing or promotion. Keep58aff452/1.70x;
  the root-cause/parity goal remains open.
- **Current complete attribution reconciles:** [result](docs/results/2026-10-16-current-update-attribution.md).
  Fresh ordinary46.108ms/coarse47.176ms,+2.315% overhead; GPU-pass union39.650ms,
  uncovered7.528ms. Host elimination alone cannot reach inclusive JAX27.223ms;
  coverage is not SM utilization. Cached world27.935/imagination6.976/slow.505/
  behavior3.014ms; helpers1.232ms. Decoder-containing chunk15.505ms and encoder-
  derivative chunk6.720ms are not pure convolution times. All23 products/272
  reports/three292-tensor states/full plans/pipelines/world groups/cache pass.
  Two guards/no warnings/544 synthetic updates/zero actions; GPU15:38:52UTC,
  publication15:49:07UTC, no worker. Backend rlib unchanged; instrumentation
  7324a048 is not promoted. Retain inherited fmt failure, zero-test preflight
  (correct test passes only after capture), and global-stack trace failure.
  One2107ns crossing retains every timestamp; ten CPU tests and independent
  6775-interval/1088-stage/3249-seal audit pass without GPU repeat. Original
  preflight sequencing remains failed. Timed chunking adds three waits.
  Current JAX inputs are resident; native input preparation is measured. Next
  CPU-reconcile all23 convolution products/helpers against actual JAX launches
  before any additional collection. Two encoder derivative captures are not
  a whole-family instruction/speedup claim. No geometry/staging sweep, prior
  timing or promotion; keep58aff452/1.70x and the full goal open.
- **Complete convolution counters reconcile:** [result](docs/results/2026-10-16-jax-convolution-counters.md).
  All23 native/JAX mains execute10.296/2.420B warp instructions,4.255x; ALU7.287x,
  tensor.792x. Forward/dX/dW ratios3.942/3.531/4.922x; not latency ratios. Nine
  typed occurrences contribute84.50% of native instructions,14 packed still3.20x.
  All18 JAX helpers add61.033M; native composed helpers remain outside this scope.
  JAX has78.643/1.589MB local reads/writes versus native zero; async/shared paths
  differ, so LSU wavefronts are not all transfers. Original36/41 guard stays
  failed; a41-launch metadata census and seven-profile supplement complete
  membership, with two retained controls. Exact reports/tensors/HLO and240 seals
  pass. Two passing/one failed guard, no warnings,816 synthetic updates/zero
  native calls/actions. GPU16:25:10UTC, CPU16:28:01UTC; no worker. Retain section,
  receipt-race, singular-pass, CSV-unit and metric-membership reader failures;
  offline repairs do not repeat GPU work. The following source audit narrows
  feeding work;58aff452/1.70x and the root-cause goal remain unchanged.
- **Convolution load contract passes on CPU:** [result](docs/results/2026-10-16-convolution-load-contract.md).
  All20 contracts/23 occurrences: nine typed paths retain16 scalar global loads
  per static K-loop and four vector shared stores; no hot BF16 conversion or
  injected array checks. Existing cursors/divisors/compensation stay.196.947M
  positions and independent per-case counts prove aligned-four composition:
  modeled source reads9.566→6.885B,28.03%, not hardware traffic/instructions or
  speed. A58.59%/B11.52% reduction is uneven; repeated masks/addresses remain.
  All152 CPU re-emissions match/20 actual modules validate. Original WGSL seals
  pass; retained runtime-option SPIR-V was not historically sealed. Restoring
  the accidentally appended prior declaration repairs the initial audit;
  edited version/failure remain, runtime unchanged. Public import offset is zero,
  ordinary/donor alignment is preserved; raw alias helper is not a size guard.
  CPU17:13:51UTC, zero GPU/builds/updates/actions, no worker. The following fixed
  read-span trial rejects that implementation. No width/geometry sweep, new Naga
  feature, prior timing or promotion. Keep58aff452/1.70x and the goal open.
- **Shared read-span composition rejected:** [result](docs/results/2026-10-16-convolution-read-spans.md).
  All23 complete products cost19.025/18.809/28.187ms native/control/candidate;
  candidate49.86% slower, queued wall49.61% slower. Calibration−1.13%/−1.04%
  passes5%; all nine changed contracts regress42.02–102.72%. No integration,
  selector, retime or full-update trial. Warp instructions rise18.01%, ALU12.14%,
  CBU13.68x, LSU15.27%, despite10.86% less L2 traffic; tensor counts stay exact.
  Control/packed instructions reproduce. Not source-line, divergence or spill
  attribution; the41-metric capture does not measure candidate local traffic.
  All304 modules/196.947M actual CPU addresses/27 small cases,2.690M F64 values
  and1.033B bit-exact output pairs pass.140 guards/133 warnings,1,360 counter
  passes/4,756 offline metrics/10,960 calls, zero learner updates/actions.
  All7,680 timings stay, including one native33.14% first/last32 shift; no cause
  asserted. Last guard18:23:15UTC, CPU18:23:50UTC, no worker.1,331 seals pass.
  Preserve unused-field build, per-image-alignment audit, binding-index compile
  and historical-library-membership failures/repairs. Five libraries have old
  seals; all137 are sealed now, not retroactively. No backend rebuild.
  The unchanged-kernel observation below diagnoses the control lowering;
  no new width/tile/layout sweep, compiler feature or promotion follows.
- **Read-span control lowering diagnosed:** [result](docs/results/2026-10-16-control-lowering.md).
  All20 contracts/23 occurrences: active thread work falls23.08% but warp work
  rises18.01%; active threads per warp instruction31.997→20.857. Both arms have
  zero local traffic with a passing positive control. The qualified baseline
  already keeps its lanes active: do not generalize the rejected candidate's
  problem to the remaining JAX gap. No direct divergence/call/inlining attribution.
  All old instruction anchors,516.575M retained values/258.287M paired values,
  full plans/pipelines and361 collection seals pass.41 guards/40 warnings,
  2,050 passes/6,902 offline metrics/2,822 calls, zero updates/actions/timings.
  GPU18:55:17UTC; no worker. CPU proof covers1.311M unreachable dW fallback
  positions and65,564 interpreted constant-one remainder values; subgroup mapping
  is not inferred. Signed-helper and terminal-newline audit failures/repairs stay.
  The CPU common-base/carry proof below preserves qualified scalar loads.
  No alignment fast/slow path, vector-width/geometry sweep or promotion.
  Keep58aff452/1.70x;
  library feeding/reuse, non-convolution costs and the full goal remain open.
- **Shared derivative address CPU proof passes:** [result](docs/results/2026-10-16-convolution-carry.md).
  All20 contracts/23 occurrences: six derivatives change,17 forward/packed
  occurrences stay exact.532 modules validate;152 old controls reproduce and
  222 typed variants change. Actual emitted-expression interpretation matches
  184.591M A slots/115.384M valid reads,5.244M cursor transitions,20 small cases,
  99,585 boundary carries; ten mutations reject. Exact structural scope preserves
  B-side code/load masks/types/K order/compensation/layouts/barriers. Static dX
  divisions32→25/calls26→18; dW branches75→72. Added select/min/max work is not
  measured native work or a speedup. No new alignment branch, helper or selector.
  CPU audits finish19:20:31UTC without retries; zero GPU/backend rebuilds/updates/actions,
  no worker. The complete comparison below rejects integration. No automatic
  allocation, prior timing, geometry/precision sweep or promotion.
- **Shared-address complete family rejected:** [result](docs/results/2026-10-16-convolution-carry-family.md).
  All23 products18.954→18.349ms,3.19%; queued wall3.26% improves but both miss5%.
  Aggregate native/control calibration+.81%/+.85% passes, not per-kernel proof.
  Four dW contracts improve9.31–10.89%; two dX regress, no favorable selector.
  Warp work+1.60%/ALU+3.03%, active threads/warp remain31.997 and local traffic
  remains zero. Source simplification is not an instruction/latency model.
  All2.690M F64 comparisons/1.033B bit-exact pairs, full plan/pipeline/binding/
  segment checks and2,031 seals pass.140 guards/46 warnings,2,000 passes/6,728
  offline metrics/11,600 standalone calls, zero updates/actions/backend rebuilds.
  All7,680 timings stay, including the dX32→3 control's.445→.739ms first/last32
  shift; cause unknown, no steady-state claim or retime. GPU19:49:53UTC,
  aggregate19:50:20UTC; no worker. Initial tool syntax failure preceded writes/
  execution; corrected without a cohort retry. No integration/full-update trial.
  Next CPU-audit reusable geometry-only operand addresses/predicates across the
  complete family, preserving K order/values and counting all table/preparation/
  lookup/lifetime costs. This is a feeding hypothesis, not a promise or claim
  about cuDNN internals. No local cursor/division tweak, width/tile/layout sweep,
  compiler feature or automatic GPU allocation. Keep58aff452/1.70x and goal open.
- **Reusable operand metadata CPU feasibility passes:** [result](docs/results/2026-10-16-convolution-metadata.md).
  All20 contracts/23 current world occurrences share eight maps/272,000 u32
  entries/1,088,000 bytes. Forward/dW share spatial addresses; dX uses inverse
  maps. Exact original K order/layouts/padded reads,30.428M explicit endpoint
  address checks with bounded affine interior proof,36 small cases/177,993
  checks and eight negatives pass. Independent source/map review reconciles
  128 B-read sites/37,888 staging assignments/1,375 split ranges and38 seals.
  Logical7.314B requests/29.255GB before masks are not GPU loads/DRAM traffic.
  One-time GPU initialization would require eight dispatches/1,064 groups;
  existing Session constants are host-visible, not the required device-local
  immutable lifetime. No actual consumer/initializer or cross-Session cache
  qualified. CPU review20:04:56UTC, no failures/retries/workers, zero GPU/builds/
  actions/updates. Next implement one shared typed observer consumer and GPU
  initialization, prove emitted indices/bindings/readiness and unchanged other
  semantics offline, then separately declare complete-family qualification.
  No automatic allocation, selector/layout/tile sweep, compiler feature or
  promotion. Keep58aff452/1.70x and goal open; the local carry rejection stands.
- **Spatial-metadata complete family rejected:** [result](docs/results/2026-10-16-convolution-metadata-kernels.md).
  All23 products19.112→36.871ms,92.92% slower; queued wall92.65% slower and19/20
  contracts regress. Aggregate control/native calibration+.0559%/+.0388% passes5%.
  ALU work−9.25% but warp work+16.26%, LSU+28.08%, L2 bytes3.422x and CBU22.57x;
  tensor counts exact, DRAM+.012%, local traffic zero. Packed regressions also
  occur with fewer L2 bytes: no single traffic/source-site/latency explanation.
  All320 paired modules plus initializer, actual-source addresses,2.690M F64/
  1.033B exact GPU values, plans/pipelines/bindings and2,415 seals pass.140 guards/
  ten warnings,2,000 counter passes/6,728 offline metrics/11,600 calls; zero
  updates/actions/backend rebuilds.60 map initializations/readbacks verify2.126M
  entries; setup wall376.987ms total, not shared full-world startup or per-call
  cost. All7,680 samples stay; max first/last32 shift4.21%. GPU20:52:28UTC,
  independent review20:53:01UTC; no worker. Preserve type-ID/whitespace/tagged-path
  CPU failures and repairs; retain first pair, no GPU repeat/recovery. No table/
  layout/geometry sweep, selector, integration or full-update trial. Keep58aff452/
  1.70x. Next CPU-reconcile existing native host-target/readback dependencies and
  resident JAX boundaries before choosing a device implementation or measurement.
  Uncovered7.528ms is not pure CPU cost or additive saving; preserve precision/
  normalizer/target state and existing device means. No new GPU allocation,
  prior timing, numerical relaxation or Atari restart; full goal remains open.
- Preceding isolated candidate`67559ce5` with `MEGANEURA_IMAGE_VIEWS=1` and
  `KINDLE_DEVICE_MEANS=1` under `runs/image-views-20261015.EblKUBZN`;
  `076f3135` under `runs/repeated-row-composition-20261015.dDTCIIip` is its
  control. `99f8aab0` remains the preceding
  control under `runs/categorical-gpu-20261015.lkZAlUAS`. Native`28864210` is the
  [general-ordering control](docs/results/2026-10-14-convolution-scheduling.md).
  Current isolated backend sources are in the copy-composition study; the ordering
  control remains `runs/convolution-scheduling-20261014.BJ5AVZFt`.
  Do not rebuild unchanged code for docs or confuse neighboring user work with
  this candidate. Latest October16 upstream: `12df95c`/`4845d652`; six new typed-
  matrix/staging/tuning/accuracy commits are reviewed above, not runtime-qualified.
  Preceding `f648a453` search/attention fixes do not change ordinary convolution.
  Blade's preceding entry-point/override resolution takes its unchanged fast
  path for all152 tested single-entry/override-free variants. New workgroup/
  BF16 capabilities are not yet qualified here. No runtime refresh or historical
  repin from this review.
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
