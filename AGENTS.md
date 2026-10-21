# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **Raw-word shared-view hypothesis rejected by counters:**
  [result](docs/results/2026-10-21-shared-word-view.md). All9 actual typed
  occurrences pass paired/F64/changed/restored outputs and35 model anchors
  across7 regions. Warp8.761B/ALU5.520B/LSU1.018B/tensor104.858M unchanged;
  wavefronts-minus-conflicts634,981,888 exact in each arm. No instruction fix,
  SASS identity, timing or driver-only claim.18 passing guards/zero warnings,
  450 counter passes/1,008 primitive dispatches/zero updates/actions;3.373M F64,
  742.034M empty zeros/372.703M paired values include capture rechecks.281 receipts
  pass11:29:31UTC, GPU ends11:27:20, no worker. Preserve pre-capture SDK-reader
  name correction and original offline failures; no GPU repeat. No Naga edit,
  timing, representation/geometry sweep or promotion. Keep79bc7cec/1.642x.
  Next CPU design separates cooperative-load from accumulator/shared-output
  handling with actual matrix contracts and known values before any GPU test;
  cuDNN's larger reuse/multistage feeding remains distinct. Atari/full goal open.
- **Residual shared-read relation and offline word view pass:**
  [result](docs/results/2026-10-21-shared-read-residual.md). All7 typed regions/
  9 occurrences have historical→current shared-read wavefronts minus conflicts
  increase209,715,200=2x tensor instructions;10 packed regions/14 occurrences
  unchanged. Predicate fix restores31.99952 active threads/warp but not this
  relation. Post-hoc counter evidence, not driver-only/SASS/source-site/latency
  attribution. CPU reversible vec4<bf16>→vec2<u32> shared view validates18 modules/
  262,144 payloads across9 occurrences, same logical packet/index/stride/matrix
  ops. Implicit workgroup layout: no physical allocation identity or GPU claim.
  No Naga edit; next separately declared actual-binding numerical/counter probe,
  not new geometry/read-span/pipeline work or old retiming. Preserve receipt-name,
  absent-ArrayStride and duplicate-type failures. Zero GPU/builds/updates/actions;
  last CPU view11:11:57UTC,no worker. PR276/e545a9f passes all6 CI1050 jobs,
  unmerged. Keep79bc7cec/1.642x; production/Atari unchanged; full goal open.
- **Narrow upstream predicate port qualifies:**
  [result](docs/results/2026-10-21-upstream-predicates.md), MeganeuraPR276/e545a9f,
  unmerged. Three guards/two shared templates, one CPU regression; no retained
  extension import. All1,860 offline modules/51 type combinations/two shapes
  pass; original-template regression fails as intended. GPU148 cases/arm,
  180 paired arrays/395,304 values,148 plans/pipeline lists exact. All72 runtime
  sources reconcile:18 dense/eight conv changed,46 unchanged. Two guards/zero
  warnings,264 forwards/32 raw backwards/zero updates/actions; review10:56:21UTC,
  no local worker. Current upstream derivatives differ: no transfer of retained
  speed claims, F32 hardware or full-model claim. Preserve workspace/log-receipt/
  linker/test-only-API preparation failures, no GPU repeats. Keep79bc7cec/1.642x
  and72005c8b reference; production/Atari unchanged. Next CI/review and remaining
  typed operand-load/feeding audit, not retiming or rejected read-span/geometry/pipeline
  sweeps. Same native geometry is not native/JAX equivalence; full goal open.
- **Predicate hardware attribution passes:**
  [result](docs/results/2026-10-21-predicate-counters.md). Nine typed convolutions
  warp9.945→8.761B(−11.91%), control456.762→17.359M(−96.20%), tensor exact.
  Packed/non-convolution total/control/LSU/tensor counts exact. LSU rises.26%:
  measured control-flow improvement, not a feeding/load fix or additive time
  attribution. Complete35 grouped RGB regions/3,751 dispatches,2,088 offline
  metrics,813 protected buffers/1.311M outputs and292-tensor states pass.
  Ordinary observer24-prefix exact; Naga/Blade and marker-only recording unchanged.
  Two guards/zero warnings/25 synthetic updates/50 counter replays/zero actions;
  2,717 receipts10:35:57UTC, no worker. No new full timing or CDP capture.
  Keep isolated79bc7cec/1.642x and reference72005c8b; production/Atari unchanged.
  Next clean upstream pure-predicate port CPU scope: a75c38e has shared checked
  templates, not the retained two files; no direct cherry-pick or port speed claim.
  Preserve conditional reads/types, qualify separately, leave dirty neighboring
  tree alone. No retained-extension import, tile sweep or automatic promotion.
- **Predicate primitive/staged-gradient qualification passes:**
  [result](docs/results/2026-10-21-predicate-qualification.md).48 fixed BF16 cases/
  arm,178.617M F64 comparisons each (170.077M empty-partition zeros;8.540M
  remaining),576 exact output hashes and all9 changed actual sources covered.
  No arbitrary-input/F16/F32 or SASS claim. Actual RGB68/CDP86 raw gradient
  tensors match whole/staged and old/new with original/changed reward labels;
  NaN prefix isolation and five292/346 checkpoint pairs pass. Raw state unchanged;
  final restoration is weights only after deliberate optimizer clear. Six guards,
  zero warnings/288 primitive forwards/16 raw backwards/four initializer updates/
  zero actions.900 receipts pass10:25:24UTC; no worker. No timing repeat; retain
  11.65%/3.25% RGB/CDP saving and isolated79bc7cec, control72005c8b. Next qualified
  grouped actual-model counter observer on candidate, ordinary transfer first.
  No profiling-only barriers, retime, sweep, automatic promotion or Atari restart.
- **Pure convolution predicates pass the complete-update cost gate:**
  [result](docs/results/2026-10-21-convolution-predicates.md). RGB48.452→42.808ms
  saves11.65%; CDP41.927→40.563ms saves3.25%, calibrated fresh controls. Exact
  24/272 reports/targets/initial-first-final292/346 tensors and complete programs/
  runtime sources pass. Nine typed occurrences change;14 packed remain exact.
  Pure eager bounds checks preserve conditional loads/nonnegative remainder
  guards, geometry/order/precision/barriers.130 offline modules pass; fewer
  static branches are not measured hardware instructions. Six passing guards,
  zero warnings/1,136 synthetic updates/zero actions. Candidate79bc7cec remains
  isolated; keep production/control72005c8b. Candidate RGB1.642x retained current
  JAX, not parity or learning evidence. Next independent GPU primitive/reference
  qualification and current-model grouped counters, not retiming/tile sweeps.
  Preserve rustfmt/WGSL parse/pre-child schema failures and final-step controller
  failure. Only the unstarted CDP candidate ran in continuation; no cohort repeat.
  Publication's compensated-sum mismatch is repaired with Rust-order explicit
  folding, no tolerance change.883 receipts pass10:14:56UTC; no worker remains.
  Reclaimed18.26GiB via247 byte-verified immutable-readout hardlinks; all273 paths/
  bytes retained. Copy before mutating those archives. No Atari restart.
- **Current-driver grouped model counters qualify:**
  [result](docs/results/2026-10-21-current-model-counters.md). RGB35/CDP20 regions
  cover3,751/763 dispatches with exact outputs/state/commands and3,306 offline
  metrics. Original RGB per-kernel capture stays failed: four adjacent CTA
  migrations, missing post-capture gates; no CDP start. Marker-only repair uses
  existing barriers, no new GPU synchronization. Nine typed convolutions account
  for the historical-to-current instruction increase; all23 reconstructed
  actual-binding SPIR-V pairs match exactly across linked compiler stacks.
  Not captured driver payloads, SASS identity or driver-only causality. Current
  conv11.487B instructions/180.224M tensor; no fresh JAX counter ratio. CDP19
  mixed regions cannot isolate exploration. Four passing/one failed guards,
  zero warnings/51 synthetic updates/150 counter replays/zero actions; final
  2,135-receipt CPU audit09:41:35UTC, no worker. Every existing CDP sync interval
  containing bonus work is mixed: six coalesced regions narrow mixed work to130
  dispatches but cannot isolate82 bonus dispatches. No new capture/barrier.
  Preserve missing-import/stale CTA annotation repairs,
  original reports and truncated failure-log snapshot; no CPU-repair GPU repeats.
  Removed1.33GiB rebuildable copied cache only; all selected/sealed evidence stays.
  Next typed-convolution control/load audit, not
  another tile sweep or native assembly query. Keep72005c8b/RGB1.858x; no
  production/Atari restart. Full performance/learning goals remain open.
- **Current counter-observer CPU contract passes:**
  [follow-up](docs/results/2026-10-21-driver615-attribution.md#follow-up-current-counter-observer-cpu-contract-passes).
  Complete41 RGB/52 CDP ranges cover3,751/763 dispatches; independent reverse
  closure proves82 bonus-exclusive dispatches/12 matrices/64.425G nominal MACs.
  Ten malformed partitions reject. Marker erasure matches current recording
  functions; only isolated Blade secondary-handle accessor needs restoring.
  Host-only SDK sizing53 ranges27,923,748B fits128MiB; zero device/GPU calls.
  Original whole-JSON CPU2GiB OOM remains; bounded metadata reader passes same
  limit. Review08:56:30UTC, no builds/updates/actions/worker. Not compiled or
  current-driver counter qualification. Next marker-only implementation/build,
  exact ordinary transfer, then separately declared bounded current615 captures.
  Keep72005c8b/1.858x; no coarse retime, tile sweep, production or Atari restart.
- **Current-driver complete attribution qualifies:**
  [result](docs/results/2026-10-21-driver615-attribution.md). Ordinary RGB48.442ms/
  CDP41.893ms; coarse overhead1.926%/1.947%, exact272 reports/targets and292/346
  initial-first-final tensors/full programs/shaders. GPU-pass union45.148/38.596ms,
  uncovered4.230/4.116ms is not SM utilization or pure CPU time. Means+six target
  passes only.281/.278ms; cached imagination6.253/15.705ms, world29.309/12.384ms
  excludes posterior. Keep72005c8b, RGB/JAX1.858x; no production/Atari change.
  Four guards/zero warnings/1,088 synthetic updates/zero actions;1,458 source/
  63 publication receipts, final08:48:28UTC, no worker. Missing shader-test output
  env and first-tail reader failures remain; same-binary/CPU repairs repeat no GPU.
  Five positive/nine negative reader tests pass; preserve tails and three extra
  coarse waits. Upstreama75c38e unchanged. Next CPU port/prove existing actual-model
  SDK observer on current runtime: RGB world and CDP imagination, with complete
  command/source/range coverage before GPU declaration. Historical580 counters
  are not615; no helper/tile sweep, automatic promotion or learning restart.
- **Final optimizer interface qualifies; PR275 ready for review:**
  [result](docs/results/2026-10-21-optimizer-final.md). 5d916851 passes all six
  CI1048 jobs, including unchanged Metal regressions. Isolated72005c8b passes
  160,998,750 F64 state checks, exact cached/ordinary and prior24-update transfer.
  Complete RGB48.524→48.448ms/CDP41.792→41.826ms,−.156%/+.082%; all272 reports
  and292/346 initial-first-final tensors exact. Full programs and exactly one
  optimizer-source replacement reconcile. Eight guards/zero warnings/1,184
  synthetic updates/zero actions;1,341/1,346 receipts; final CPU08:28:34UTC,
  no worker. Preserve failed whole-file equivalence assertion: upstream lacks
  retained prepared-step guard; exact-context two-file delta preserves it.
  No GPU repeat. PR275 is ready, unmerged; production unchanged. Use qualified
  isolated72005c8b for next declared current-driver attribution, not automatic
  Atari restart. Include categorical-mean and resident-target timestamps missing
  from helper waits; preserve numerics/caching and disclose extra waits. Historical
  580 counters are not615 captures. JAX gap~1.86x and five-game goal remain open.
- **Optimizer complete-model qualification passes; Metal CI follow-up active:**
  [result](docs/results/2026-10-21-optimizer-model.md). Shader-only4bd/6b621e5c
  passes160,998,750 F64 state checks and exact24-update observer transfer.
  No model subnormal variances; component tests cover those. Complete RGB
  48.403→48.446ms/CDP41.817→41.886ms,.090%/.164% overhead; all272 reports and
  292/346 initial-first-final tensors exact. Full programs/sources reconcile
  exactly one optimizer-body replacement. Eight guards/zero warnings/1,184
  synthetic updates/zero actions; final CPU audit08:12:48UTC, no GPU worker.
  Preserve two pre-build assertions and original all-shader-equality audit;
  separate exact-body CPU repair passes without GPU repeat. PR275 CI1047 passes
  five jobs but Metal fails normal variance/Inf regressions. F32 fused
  beta*(old-g*g)+g*g reproduces its exact bits and NaN signature; not SASS proof.
  Separate worktree/runs/optimizer-coefficients-20261021.pcZGh5zI supplies independent
  1-beta uniforms,56-byte interface; tests/tolerances unchanged. Follow-up5d916851
  pushed to PR275: three GPU/two layout tests/Clippy pass,53 Session updates,
  192 exact checkpoint values,one guard/zero warnings/338 receipts. CI1048 is
  confirmed in progress; actual Metal CI and final-model interface checks remain.
  No local worker remains. No production
  promotion/Atari restart; original complete evidence does not qualify this follow-up.
- **Optimizer poison and cached-runtime qualification pass:**
  [result](docs/results/2026-10-21-optimizer-cached.md). Finite-history guard
  preserves NaN/Inf;192 new classifications/value checks and prior finite/raw-
  gradient/checkpoint tests pass on latesta75c38e/Bladeeae0c6f7/Nagaa489554c.
  Portable regression has no workstation headroom gate; local launcher retains it.
  Draft MeganeuraPR275/4bdcb8d is pushed, not merged. Same shader-only change on
  retainedd3ff4948 backend passes three tests/eight cache recording+replay checks;
  192 checkpoint values/metadata exact. Built Kindle6b621e5c in
  runs/optimizer-cached-20261021.2sOmnmzz, no full-model execution yet.
  Two guards/zero warnings/106 synthetic Session updates/zero Dreamer updates/
  actions;342/451 receipts pass, no worker. Missing-wrapper-seal failure is
  pre-child. Original checkpoint byte gate remains failed3/4 due metadata key
  order; separate strict semantic audit/eight negatives pass, no GPU repeat.
  Next complete RGB/CDP numerical qualification then repair overhead, preserving
  six flags/config/precision. Keepd3ff4948 control; no reproduction of known-bad
  underflow, automatic promotion or Atari restart. Full parity/learning goal open.
- **JAX tiny-gradient control and first GPU repair pass:**
  [result](docs/results/2026-10-21-optimizer-denormals.md). Pinned JAX passes96
  F64 checks and preserves tiny variances; queried Vulkan FP32 DenormPreserve
  is unsupported, so latest Blade's F16 fix is insufficient. Upstreama75c38e
  still has the failing update. Isolated scaled-variance shader preserves state
  format/epsilon and passes96 checks over16 gradients/two reversed-sign steps;
  all48 original control values and30 unaffected candidate values match exactly.
  For3e-18, momentum30→.0996678. Not a full learner or performance qualification.
  Two passing guards/zero warnings/six standalone optimizer steps/zero learner
  updates/actions;625 candidate receipts and119 JAX receipts pass. CPU memory-
  field/offline-binding failures remain. Review07:15:35UTC; no worker.
  Next state-boundary/rounding/long-sequence tests, then isolated actual Session/
  checkpoint/cache integration. Keep ordinary state semantics and epsilon; no
  CPU workaround, Naga feature project, automatic promotion or Atari restart.
- **Current-driver JAX control passes; optimizer underflow reproduces:**
  [result](docs/results/2026-10-21-driver615-baseline.md). JAX25.606ms core/
  26.066ms inclusive versus native48.281ms,1.85x on615.71.09. Exact272 reports/
  288 initial-final tensors,940 fusion bodies/112 library calls/heavy work pass.
  Native observer transfer fails numerical equality despite exact programs and
  shader sources. Historical580 has four oversized first LaProp moments, max29.24,
  with zero variance; current model fixture has none, but a separate current-driver
  probe reproduces six signed tiny-gradient failures:3e-18 stores variance0 and
  momentum30 instead ofabout.1. Raw gradients exact; original failure plus one
  declared diagnostic export remain failed. No tolerance relaxation or driver-bug
  claim. One passing/two failed guards,zero warnings,272 JAX updates/two standalone
  optimizer steps/zero actions; no worker. Next pinned-JAX tiny-gradient comparison
  and F32 float-control qualification before more profiling; do not assume latest
  Blade's F16-only denormal fix covers it. No CPU workaround, epsilon change,
  automatic promotion/Atari restart or assertion that this explains game scores.
- **Slice views miss the cost gate:** [result](docs/results/2026-10-21-slice-views.md).
  Fresh driver615.71.09 RGB48.281→47.776ms/CDP41.771→41.016ms improve1.05%/1.81%;
  RGB fails3%, no promotion/retime. All24/272 reports/targets and four292/346
  tensor pairs match exactly;329/359 dispatches removed. CPU121.508M addresses/
  59 modules and GPU48 cases/12,800 F64 values pass. Nine guards/zero warnings,
  1,184 synthetic updates/384 primitive forwards/zero actions; no worker.
  Preserve initial CPU type-inference failure; staged raw-gradient and non-world
  runtime-group qualification remain unrun after cost rejection. Keepd3ff4948.
  Host rebooted into615.71.09: historical580 timing42.983/27.223ms/1.58x and
  counters are not current-driver measurements. Next fresh unchanged JAX control
  against the new native baseline, not another small-kernel sweep or Atari restart.
  Upstreama75c38e/eae0c6f7 checked; no silent source refresh.
- **Guarded reduction-tail implementation rejected:** [result](docs/results/2026-10-20-reduction-tail.md).
  Complete RGB42.936→43.275ms/CDP40.673→42.023ms regress.79%/3.32%;
  both declared cost gates fail. All24/272 reports/targets and four292/346-tensor
  checkpoint pairs match exactly; full plans/groups/pipelines and27/30 changed
  runtime shader bodies reconcile.1,972 offline modules/2,323 ownership cases
  pass. Shader-level subgroup count/width guard preserves the old tree on partial
  membership; no Blade/Naga change or fixed local-ID assumption. Primitive GPU
  oracle, runtime branch observation and staged raw gradients remain unrun:
  cost rejection ends this candidate without those allocations.
  Eight guards/six warnings/1,184 synthetic updates/zero actions; final audit
  06:08:32UTC, no worker. Preserve the CDP initial-step observer failure and
  missing five runtime-source bodies per workload; repairs repeat no GPU cohort.
  No retime, specialization/geometry sweep or promotion. Keepd3ff4948/1.58x.
  Next version-safe slice-view implementation from the existing CPU proof;
  its limited opportunity is not convolution's larger deficit or parity.
- **CPU reduction-tail contract passes:** [result](docs/results/2026-10-20-reduction-tail-contract.md).
  All1,534 generated reductions/eight actual RGB/CDP programs inventoried;
  231 symbolic configurations/20,763 ordered nodes/6,625 independently constructed
  row trees pass. Common RMS barriers could fall9→5 at full subgroup32; no
  compiled candidate, numerical or speed claim. Ordinary Blade emission does
  not enforce full subgroups: establish that contract, use actual subgroup/lane
  IDs and preserve fallback, row geometry, ordered arithmetic and cross-subgroup/
  final-publication barriers before GPU qualification. No local-ID assumption.
  Latest upstreamf1f4395c/eae0c6f7 reviewed in isolation; matrix extraction/
  rectangular staging/FP16 denorm changes do not implement this mechanism.
  No runtime refresh. CPU review05:49:50UTC October20,13 receipts, zero GPU/
  builds/updates/actions; no worker. Next isolated typed implementation and
  complete offline compiler checks, not a sweep or Atari restart.
  Keepd3ff4948/1.58x; convolution's larger deficit and the full goal stay open.
- **Actual-model dense selection qualifies but fails performance acceptance:**
  [result](docs/results/2026-10-19-model-selector.md). RGB43.010→42.890ms,.28%,
  below3%; CDP40.817→40.375ms,1.08%. Keepd3ff4948/1.58x, no promotion/retime.
  Full-race bounds111.25/228.625MiB fit512MiB; actual timed180/206 dispatches and
  pipeline keys change. All527.640M paired outputs/raw gradients, staged/whole
  late-target checks,272 reports and five292/346-tensor checkpoint pairs pass
  exactly. Search3.536/5.352s; private timing differences are not additive graph
  savings. RGB posterior+world essentially unchanged. Six guards/eight warnings,
  1,094 synthetic updates/84 raw backwards/24 forwards/zero actions; GPU ends
  14:14:54UTC, final tensor audit14:19:05, no worker.678 seals/CPU tests/Clippy
  pass. First CPU audit397s; upcoming-audit BLAS threading fixed, no numerical
  relaxation/repeat. Isolated study`runs/model-selector-20261019.0xWDbwn1`.
  Missing dense choices are not the big remaining lever. Resume CPU subgroup-tail/
  slice-view contracts from the recurrence audit; convolution feeding remains
  the larger distinct deficit. No new geometry sweep, automatic allocation or
  Atari restart; root-cause/parity goal remains open.
- **Typed selector integration qualifies, not a new learner speedup:**
  [result](docs/results/2026-10-19-typed-selector.md). All587/643 RGB/CDP typed
  dense incumbents preserve their captured shaders;1,182 typed variants validate
  offline. Twelve GPU fixtures pass327,800 exact F64 comparisons, unchanged
  parameters and command-cache re-recording; eight plans change. Grouped129/126
  products remain excluded. Local tuning/swap cache invalidation is repaired;
  swap GPU coverage remains unrun. Default64MiB scratch excludes eight/sixteen
  actual dense occurrences; largest incumbent lower bounds86.25/203.625MiB.
  Two guards/one pass/one wrong-adapter initialization failure, zero new warnings/
  actions/updates; original CPU API/SPIR-V observer failures stay. Repaired source
  is `runs/typed-selector-20261019.MhXVhJTs/device-fix`; not production or a new
  Kindle binary. Next actual-model qualification and complete cached/staged
  search with a properly accounted scratch bound. Keepd3ff4948/1.58x; no fixture
  winner promotion, manual sweep or Atari restart. GPU ends07:06:03UTC; final
  CPU publication07:12:44UTC checks743 seals, library Clippy passes. Empty-race
  serialization/baseline-warning reader repairs retain both originals, no GPU
  repeat. No worker remains.
- **October19 current upstream qualifies without a net speedup:**
  [result](docs/results/2026-10-19-upstream-refresh.md). Meganeura`c19cb1ca`,
  Blade`4845d652`, Naga`13f582dd`, plus retained isolated extensions. New ordinary
  dense policy regresses RGB42.912→57.108ms/CDP40.701→65.495ms. Restoring the whole
  qualified dense policy removes18.76%/37.50% versus fresh new-default controls,
  but RGB46.380ms fails its original5% recovery gate. Full source audit then
  finds one convolution route bypassing vector/incremental staging despite
  unchanged plan labels. Restoring that existing generator removes3.367ms RGB:
  latest42.983ms/CDP40.741ms, within.17%/.10% of the original controls. Not a new
  JAX-gap improvement. Exact272 reports/targets/initial-first-final292/346 tensors,
  eight programs/constants/pipelines and205/225 shader-source multisets pass.
  Keep current-upstream isolated`d3ff4948` in
  `runs/upstream-refresh-20261019.KWEh8dEH/conv-route`, all six retained flags,
  with`e45b9c43` as historical anchor.26 guards/21 pass/5 fail/21 warnings,
  3,456 complete synthetic updates/one partial world update/19 dense probes,
  zero actions. Final2,464-seal review06:09:49UTC; no worker. Preserve integration
  storage bug, observers, failed recovery gate, CPU filename/source assertion,
  copied-manifest/flag failures and repairs; no tolerance relaxation or GPU retry
  for CPU defects. Fixed-binding offline251-module compiler comparison has250
  byte-exact pairs; RMS-gradient difference stays, not a driver-code identity claim.
  Next expose the retained typed larger-reuse family to the common selector
  before measured whole-program search; the current typed domain cannot choose
  it, and ordinary`build` does not run search. This supersedes the unimplemented
  recurrence/subgroup proposal, not the measured convolution-feeding deficit.
  No closed manual sweep, production promotion or Atari restart follows.
- **Active goal: understand the remaining JAX gap and establish how to reach
  similar Meganeura performance.** It remains open. Latest isolated complete
  RGB12M/B32/T16 is42.983ms versus inclusive JAX27.223ms,1.58x; B8 retains22.780/12.737ms,
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
- **Host-target dependency contract passes:** [result](docs/results/2026-10-16-host-target-contract.md).
  RGB/CDP each write22.1472MB of eight behavior inputs on CPU, including16.6464MB dense
  two-hot targets expanded from65,280 scalar bytes;80 world target inputs add
  1.5708MB of logical writes, not unique data or measured traffic. Means/features/
  slow EMA already have device paths. JAX also constructs dense one-hot targets,
  inside its compiled update; no compact-loss/fusion claim. All18 behavior/160
  world interfaces,36 source-F32 lambda cases/6,696 values, six negatives/seven
  retained CPU tests and33 seals pass. Raw272-update trace/2,448 intersections
  reconcile: targets1.437ms, behavior inputs1.012ms;8.086ms readback overlaps
  7.913ms GPU work, not removable transfer overhead. Preserve initial[32,512]
  stochastic-layout audit failure; exact[1024,16] repair passes on CPU. Production
  replay-context refresh remains required and excluded from the fixed benchmark.
  No GPU/builds/actions/updates. Next implement resident F32 returns, percentile/
  EMA, weighted action/two-hot targets and staged-world writes; reuse existing
  DeviceCopies::copy_regions, means and graph losses. Preserve masks/discount/
  pre-update ordering, normalizer checkpoints/diagnostics and CDP bonuses.
  Offline/binding/lifetime proof precedes separately declared GPU qualification
  and complete RGB/CDP comparison; no CPU-normalizer workaround, prior timing,
  convolution revival, automatic promotion or Atari restart. Keep58aff452/1.70x;
  target work is a bounded portion and the full root-cause goal remains open.
- **Resident target implementation exists; qualification fails:** [result](docs/results/2026-10-16-resident-targets.md).
  Isolated27b8befc implements F32 returns/weights, percentile/EMA, weighted action/
  two-hot targets, GPU replay features and staged-world writes. Keeps losses,
  masks/CDP arrival bonuses/checkpoints and CPU replay-context refresh. Actual
  RGB/CDP16 behavior/160 world targets, prefix/scratch exclusion, eight negatives,
  ten offline Vulkan modules and Clippy pass. New buffers598,892B;81 device-copy
  regions/6.486MB logical, not traffic/savings. Supports<=8192 returns and staged
  full-BPTT/full-microbatch only.12 component generations/zero updates/actions,
  one guard/no new warnings; identity consumers use48-wide features, not actual
  12M training.759 output-array/301 restored-file checks pass.19,652,262 numerical
  comparisons fail one replay coefficient plus its world copy: error1.6867e-5
  versus unchanged1.6019e-5 gate. Exact rational reference confirms the failure.
  Original Rust helpers reproduce7,839,540 source-F32 values exactly but fail
  four occurrences elsewhere under this same envelope: CPU equality alone is
  not a solution. Same-scalar encoding agrees within one F32 ULP; no native
  instruction-level contraction claim. A CPU-only affine/FMA rewrite fails78
  occurrences and is rejected without GPU execution. Preserve mutable-caller,
  reserved-WGSL-name and missing-CPU-helper failures/repairs; no widened gates,
  GPU repeats, compiler feature, timing or promotion. Last guard21:48:36UTC;
  CPU control22:05:39UTC, no worker. The following separately declared repair
  supersedes its next step, not the failed evidence or original gates.
- **Resident targets qualify under the revised encoding contract:** [result](docs/results/2026-10-16-resident-targets-qualified.md).
  Complete RGB45.852->43.197ms,5.79%; CDP43.657->40.997ms,6.09%. RGB/JAX1.59x;
  fresh controls calibrate within.76%. Keep isolatede45b9c43 with
  `KINDLE_DEVICE_TARGETS=1` under `runs/resident-target-model-20261016.tEVYDi5C`,
  alongside device means/copy composition/word staging/image views/cached recording.
  Leave the presence-based target flag unset for control58aff452. Backend unchanged.
  One local compensated-F32 recurrence passes the original19.652M component
  comparisons; ordinary F32 buffers/masks/losses stay. Actual12M first targets/
  metrics, all programs/pipelines/groups, retained272-report/initial-first-final
  controls and each arm's24-report prefix pass. Eight checkpoint roundtrips
  preserve all292/346 tensors/normalizer/counters exactly, not replay/live lifetime.
  The original exact-slow-target smoke stays failed: CDP5,219/310 coefficients
  differ by<=2/1 ULPs. Explicitly revised smoke applies the original dense-target
  tolerance to all encodings; no component tolerance or GPU cohort changes.
  Fifteen CPU regression checks pass; provisional RGB/one-ULP labels are corrected.
  Learner parameters are not trajectory-exact: final relativeL2.000256 RGB/.05434
  CDP after272 updates; moments/slow critic separately reported. No learning parity.
  Nine new guards/six warnings,12 component generations/1,184 synthetic updates/
  zero actions. GPU ends22:45:15UTC, independent audit22:49:19UTC; no GPU worker.
  Retain CPU descriptor exhaustion and pair-lifetime repair, plus stale common
  observer96 metadata and independently verified1,088 timing updates. No raised
  limits or GPU repeats. Supports<=8192 returns/full staged BPTT/microbatch;
  production replay-context readback/refresh remains. No production/Atari promotion.
  About2.66ms saved does not explain the remaining15.97ms over JAX. Continue from
  existing convolution/nonconvolution counters and dependencies; require a distinct
  mechanism and CPU proof before another GPU allocation. No closed-family revival,
  geometry/precision sweep, prior-batching timing or new compiler project implied.
  Full root-cause and learning-quality goals remain open.
- **Direct F32 output family rejected:** [result](docs/results/2026-10-16-direct-output-family.md).
  All23 complete products19.120->18.432ms GPU,3.60%; queued wall3.65% better,
  both miss5% gates. Native/control calibration−1.77%/−1.74% passes in aggregate;
  ten of20 contracts regress. All2.690M F64/774.862M bit-exact comparisons and
  2,020 source receipts pass.100 guards/15 warnings,8,880 standalone calls,
  zero updates/actions. GPU ends23:28:37UTC, independent audit23:29:00; no worker.
  Native dX32→3 first/last32 shifts2.035→.457ms; retain all samples/cause unknown.
  K loops and shared allocation stay; seven occurrences bypass output scatter.
  Preserve CPU syntax-reader failure/repair. No retime, selector, integration,
  full-update trial or compiler project. Artifacts
  `runs/convolution-direct-output-20261016.93l7AixM`. Keepe45b9c43/1.59x.
  Convolution feeding and recurrent work distribution remain distinct questions;
  fewer matrix instructions do not prove time parity. No closed tile/split-K/
  feeding sweep, automatic GPU allocation, production or Atari promotion.
  Source review confirms October15's rejected tensor family already loads global
  image/weight storage directly, not shader-managed shared operands. Do not
  relabel that API as untried. Next prove complete current native model counter
  membership and a command/dependency-preserving observer: existing native
  counters use standalone synthetic inputs, while JAX's use its full update.
  Historical captured-group value controls remain valid but are not universal.
- **Current-model counter CPU contract passes:** [result](docs/results/2026-10-16-model-counter-contract.md).
  All3,751 world dispatches/23 products and versioned owners reconcile. All17
  convolution-containing groups are mixed.1,138 non-nested in-pass marker ranges
  preserve1,131 original barriers in the CPU command model; six negatives pass.
  Naive per-group diagnostic recording adds1,111 compute boundaries. Allocate
  for3,751 launches, not1,139 ranges including empty; invalidate marked caches
  at profiler lifecycle changes. The subsequent capture below now qualifies a
  coalesced partition; retain this original CPU evidence.
  Initial label-owner assertion fails; separate versioned-path repair passes.
  CPU ends23:45:00UTC, zero builds/GPU calls/updates/actions, no worker. The
  later implementation/capture is not part of this CPU allocation.
  Fixed-world counters omit optimizer/other programs; replay time is not ordinary
  latency. No kernel/geometry sweep, closed-family revival or automatic promotion.
- **Current-model counters qualify:** [result](docs/results/2026-10-17-model-counters.md).
  All3,751 world dispatches retain original1,112 groups/1,131 barriers. All23
  convolutions match92 standalone CTA/warp/tensor/ALU anchors; actual inputs
  do not explain the instruction gap. They execute85.83% of world warp work,
  4.255x JAX convolution instructions/7.287x ALU, not time fractions or speedups.
  Whole/detailed integer sums, full outputs/targets/three292-tensor pairs and813
  protected hashes pass; local traffic is zero. First detailed allocation fails
  the128MiB host-profiler cap before dispatch; seven target exports are missing.
  Both failures remain. Separate41-range+empty coalescing fits22.13MB without
  raising limits or repeating the completed whole capture; new target exports pass.
  Three guards/two pass/one fail, zero warnings,26 synthetic updates/100 counter
  replays/35 ordinary fixed replays/zero actions;2,552 offline metrics and2,095
  repaired-capture seals pass. GPU ends00:19:53UTC, CPU publication00:22:19;
  no worker. Original dependency/lint/premature-preflight/target/metadata reader
  failures and failed controller status remain; separate terminal review passes.
  Isolated observer47dfc6f7 is not promoted; keepe45b9c43/1.59x complete gap.
  Next CPU-map retained convolution dependency counters and18 non-convolution
  spans to actual producers/consumers. No new GPU cohort needed for that review,
  closed-family/assembly retry, compiler project, precision sweep or Atari restart.
- **Current-model dependency map complete:** [result](docs/results/2026-10-17-model-dependencies.md).
  All3,751 dispatches/7,063 versioned reads/813 roots/41 ranges reconcile. A1,109-
  dispatch chain crosses1,109 of1,112 groups: regrouping alone cannot remove
  hundreds of dependencies. This is not critical-path latency. Other spans are
  mixed, not encoder/decoder labels; long-scoreboard warp-cycle fractions are
  44.21% other/18.79% convolution, not wall-time shares or SM utilization.
  Blade already disables synthetic loop/division guards. One fixed non-algebraic
  SPIR-V normalization validates40 modules for20 contracts/23 products,
  complete interfaces/layouts/cooperative operations/barriers and three negatives.
  Call-expanded K-loop IR falls11.10–19.01%; all branch/global/shared-access sites
  stay. No native instruction/speed claim: the driver may already remove this IR.
  Original output-validation claim was incomplete: optimizer validates inputs.
  Separate unchanged-output validation passes,60 tool calls total/no normalization
  repeat. Six graph negatives/171 combined receipts pass; CPU ends00:53:58UTC, zero GPU
  calls/builds/updates/actions, no worker. Next CPU-review one exact-module
  replacement observer, then separately declare one complete-family numerical/
  counter test of this fixed mechanism. No pass/geometry search, Naga feature,
  closed-family revival or automatic integration. Keepe45b9c43/1.59x; goal open.
- **Uniform-interface control reconciled:** [result](docs/results/2026-10-17-uniform-interface.md).
  Fixed normalization passes all20 contracts/3.366M F64/275.972M paired values,
  then its first counter control fails native anchors;19 captures stay unstarted.
  Keep that controller failed. A separate fixed four-condition CNN0-forward
  diagnosis identifies inline-uniform versus UBO parameter storage: UBO removes
  14.05% warp/22.77% ALU instructions, identical outputs/CTA/tensor work. Explicit
  full32/default subgroup settings have no effect here. Both native/direct
  anchors reproduce exactly. All20 functions match across library generations
  and actual Blade SPIR-V1.3 emission; packed descriptor permutations are known.
  This is not ordinary latency, a whole-family result or source-line attribution.
  Across both allocations23 guards/15 warnings/450 standalone dispatches/100
  counter passes, zero learner updates/actions. Four-condition checks pass2.097M
  F64/402.653M pairs;450 source seals pass, CPU review01:39:15UTC, no worker.
  Missing names/descriptor-order/negative-extent/output-name/SPIR-V-version CPU
  failures and repairs remain, as does the original normalization-anchor failure.
  Next CPU-audit existing Blade UBO scratch/descriptors/refresh against actual
  RGB/CDP cached programs before a separately declared same-native complete
  numerical/ordinary comparison. Count all uploads/lifetimes. No normalization
  restart, new descriptor backend, favorable selector, pass/subgroup/geometry
  search or automatic promotion. Keepe45b9c43/1.59x; root-cause goal remains open.
- **Complete UBO policy rejected:** [result](docs/results/2026-10-17-uniform-model.md).
  Same-native complete RGB43.164→42.540ms,1.45% less, misses3%; CDP41.044→
  41.820ms,1.89% slower. All24/272 reports/first targets/initial-first-final292/
  346 tensors, full plans/shaders and eight checkpoint roundtrips match exactly.
  Controls calibrate within.116%; all samples/tails stay. CPU audit covers ten
  program views/430 modules/19 lifetime checks; existing1MiB per-command scratch
  fits, cached graphs remain immutable and optimizer bindings dynamic. Actual
  alignment64; command cache required. No shader/Naga/precision change.
  Eight guards/four warnings/1,184 synthetic updates/zero actions; GPU ends
  02:06:04UTC, independent3,196-seal review02:07:51UTC, no worker. Preserve the
  constant-shader reader, Rust edition and dual-Blade-path build failures/repairs.
  Isolatedb01fe8f3 is not adopted; keepe45b9c43/1.59x. Single-kernel instruction
  savings are not full-model latency. No selector, retime, normalization restart
  or production change. Next CPU-match current convolution/normalization/
  activation/adjoint interfaces to actual JAX fusion bodies, including versions,
  tapes and rounding. Existing captures only; no new GPU allocation/compiler
  feature or geometry sweep. Feeding/non-image costs and the full goal stay open.
- **Image-fusion contract complete:** [result](docs/results/2026-10-17-image-fusion-contract.md).
  All16 RGB/eight CDP image regions already fuse bias/pooling/normalization/
  SiLU and adjoints. Eight JAX forward libraries use activation=None;26 connected
  forward fusion nodes include extra layout/mask work, not a timing/dispatch-
  advantage ratio. All1,215 top-level HLO nodes/3,198 reads/17 call contexts and
  23 products reconcile through actual tuple paths. Eight JAX BF16 activation
  boundaries differ from native F32 image arithmetic; no precision change.
  All15/eight native parameter-gradient chains and required tapes reconcile;
  logical313.852/208.306MB tapes are not traffic, peak VRAM or removable savings.
  Four shape fixtures/four malformed-shape/eight trace-mutation rejections and
  29 seals pass. Preserve nested-tuple/comment reader failures, local inspect
  shadowing and original negative-check label clarification. CPU02:28:37UTC,
  zero GPU/builds/updates/actions, no worker. Keepe45b9c43/1.59x.
  The following CPU timing contract now passes; no fusion/precision/geometry
  search follows. Convolution feeding/recurrence and the full goal remain open.
- **Cached boundary-timing CPU contract passes:** [result](docs/results/2026-10-17-boundary-timing-contract.md).
  All23/11 RGB/CDP convolutions occupy17/eight original intervals; ten cached
  world command projections preserve groups/barriers/submissions.1,137/1,130
  proposed query slots,2,176 modeled executions/22 negatives/29 seals pass.
  Retained272-update/6,775-interval coarse trace still adds three waits; not
  current ordinary timing. Original18-versus20 enum-census failure remains.
  CPU02:43:42UTC, zero GPU/builds/updates/actions, no worker. Keepe45b9c43/1.59x.
  The isolated implementation below now qualifies with checked existing waits,
  per-secondary ownership/generations and actual non-world groups. The original
  CPU evidence stays; query availability alone remains insufficient.
- **Cached model boundary timing qualifies:** [result](docs/results/2026-10-17-boundary-timing.md).
  Complete RGB43.192→44.775ms marked,+3.6647% overhead, passes5%; control drift
  −.0100%. All24/272 reports/first targets/initial-first-final292 tensors, four
  checkpoint roundtrips/full plans/shaders/groups and submit/wait counts match.
  Source command projection retains original passes/barriers/waits; no complete
  driver event trace. All2,176 cached replays/462,128 intervals/1,677 seals pass.
  World29.069ms includes18.060ms convolution-only intervals; total cached39.909ms,
  nonconvolution21.850ms. These are completion-frontier intervals, not pure
  kernels, global GPU coverage or an additive gap budget. All23 products match
  retained JAX11.255ms kernel activity including helpers; different boundaries.
  Native encoder/decoder6.162/11.897ms versus JAX4.958/6.297ms is not a CDP gain.
  Four guards/two warnings/592 synthetic updates/zero actions; controller
  03:05:30UTC, independent attribution03:08:47UTC, no GPU worker. Initial patch
  parser refusal/repair stays; no GPU failure/repeat. Three CPU tests/build/Clippy
  pass. Observer786a3010 is not promoted; keepe45b9c43/1.59x, production unchanged.
  The complete CPU membership review below now passes; no timing-density search
  or automatic promotion follows. Goal stays open.
- **Nonconvolution timing membership reconciles:** [result](docs/results/2026-10-17-nonconvolution-timing.md).
  All4,692 dispatches/8,948 versioned reads/716 matrix products/1,699 intervals
  and28,064 JAX kernels reconcile. Native imagined recurrence/features has180
  products in631 dispatches; JAX180 in630 kernels plus45 copies. This is not a
  large launch-count mismatch. Native wholly-recurrent intervals5.758ms contain
  2.472ms matrix-only/2.458ms no-matrix/.827ms mixed;148 products are matrix-only.
  Boundary-mixed.438ms stays separate. JAX matrix2.106/other.747/copy.055ms,
  3.147ms span uses different interfaces/boundaries; no additive saving or ratio.
  All669 grouped fusion-symbol rows/1,120 exact body pairs pass;54 library groups
  retain candidate sets and one.0112ms posterior symbol remains unresolved, not
  omitted.20 negative/six positive sort checks pass; both failed sort readers stay.
  CPU review03:27:19UTC,1,712-seal closure03:30:48UTC; zero GPU/builds/updates/
  actions, no worker. Keepe45b9c43/1.59x.
  Fresh upstream5d6d9a36 adds a workgroup matrix family, absent from our runtime;
  opt-in/search selected, not an automatic speedup. The CPU interface/compiler
  audit below now passes before any separate GPU comparison.
  No geometry sweep, favorable subset, closed-family revival, compiler feature
  project, automatic refresh/promotion or Atari restart. Nonmatrix/conv costs and
  the full goal remain open.
- **Upstream dense workgroup CPU compatibility passes:** [result](docs/results/2026-10-17-workgroup-matrix-contract.md).
  One32x64/K32/64-thread policy fits504/716 RGB and552/769 CDP products;
  grouped/unaligned fallbacks stay. All61 contracts/six unchanged upstream
  shaders validate with Naga13f582dd/SPIR-V1.6. All106 split products retain
  partitions/F32 partials/SumRows.4.937M addresses/2,048 ownership values,
  27 negative checks and71 receipts pass. No GPU numerics/performance claim.
  Four failed CPU readers (semantic hashes, Default tag, stripped names,
  uniform wrapper) and repairs remain; no GPU repeat/compiler edit. CPU review
  03:47:15UTC, zero GPU calls/updates/actions, no worker. Retained properties
  support the policy, but current logical-device enablement is still absent.
  The complete bounded comparison below now qualifies features/bindings/numerics
  but rejects this policy's performance. No geometry search, favorable subset,
  new compiler feature, silent refresh or promotion.
  Keepe45b9c43/1.59x; convolution/nonmatrix costs and the full goal remain open.
- **Fixed upstream workgroup-matrix policy rejected:** [result](docs/results/2026-10-17-workgroup-matrix-family.md).
  All96 operations/716 RGB/769 CDP memberships include grouped/unaligned
  fallbacks and142 reductions/130 casts. Current/upstream-original/candidate
  weighted GPU costs11.932/11.939/12.797ms RGB and15.605/15.614/18.345ms CDP:
  candidate regresses7.25%/17.56% versus current; queued wall also regresses.
  Both compiler controls calibrate within.312%; both5% performance gates fail.
  These are complete standalone policy costs, not full-update timings or a
  rejection of every upstream geometry/search. All190.859M F64/763.437M exact
  paired values and288 guards pass;20 warnings,23,904 operations/26,643 dispatches,
  zero updates/actions/counter passes. GPU ends04:30:34UTC, independent3,308-seal
  review04:30:53UTC, no worker/retry. Structured fixtures do not qualify actual
  gradients/learning. No full-model trial, favorable subset, retime or sweep.
  Keep e45b9c43/1.59x. The CPU recurrence comparison below now passes. No closed
  precision/geometry search, failed prior restart, compiler project or promotion.
- **Nonmatrix recurrence contracts pass:** [result](docs/results/2026-10-17-recurrence-contract.md).
  Native135 RMS calls versus225 JAX kernels implement the same135 logical
  normalizations: basic normalization fusion is not missing. Three retained
  native WGSL forms all use256 invocations/one row/nine workgroup barriers;
  this is not hardware stall attribution or evidence of JAX subgroup scheduling.
  GRU slices/update take90 native dispatches versus15 JAX fusions, excluding
  bias. F32/BF16 state/rounding and external/in-loop RNG differences remain.
  All631 native/630 JAX launches and45 JAX copies stay accounted for.
  Complete RGB/CDP private F32 split-view proof passes121.508M addresses,
  source versions/protection/readers and36 small cases/560 bit patterns. Could
  remove329/359 copy dispatches and404.488/421.265MB logical writes; no codegen,
  DRAM or speedup claim. Copy-only current RGB intervals total.239ms world/
  .308ms imagination, with.572/.153ms mixed; not attainable/additive savings.
  Separate sigmoid duplicate bindings remain, not included in removals.
  One failed metadata reader/repair retained. Independent104-seal review
  04:46:46UTC, zero GPU/builds/updates/actions, no worker. Next CPU-prove a
  subgroup reduction tail preserving the exact tree, complete typed family,
  enforced size/lane ownership, required cross-subgroup barriers and rounding.
  No local-ID assumption, unsafe barrier weakening, row/geometry sweep or new
  Naga project. Slice views remain unimplemented. Keepe45b9c43/1.59x.
- Preceding isolated candidate`67559ce5` with `MEGANEURA_IMAGE_VIEWS=1` and
  `KINDLE_DEVICE_MEANS=1` under `runs/image-views-20261015.EblKUBZN`;
  `076f3135` under `runs/repeated-row-composition-20261015.dDTCIIip` is its
  control. `99f8aab0` remains the preceding
  control under `runs/categorical-gpu-20261015.lkZAlUAS`. Native`28864210` is the
  [general-ordering control](docs/results/2026-10-14-convolution-scheduling.md).
  Current isolated backend sources are in the copy-composition study; the ordering
  control remains `runs/convolution-scheduling-20261014.BJ5AVZFt`.
  Do not rebuild unchanged code for docs or confuse neighboring user work with
  this candidate. Latest October17 upstream is `f4153c6d`/`4845d652`. The two
  commits since5d6d9a36 add repeated-row folding and intermediate timestamp
  harvesting; both mechanisms are already represented in our isolated stack.
  No runtime refresh. Preceding October16 `41a15d31` added seven
  tuning/calibration commits preserve codegen/shader/compiler trees and ordinary
  unbounded pipeline construction. Isolated fetch/audit, no refresh or search.
  The preceding six typed-matrix/staging/tuning/accuracy commits through12df95c
  are reviewed above, not runtime-qualified.
  Preceding `f648a453` search/attention fixes do not change ordinary convolution.
  Blade's preceding entry-point/override resolution takes its unchanged fast
  path for all152 tested single-entry/override-free variants. New workgroup/
  BF16 capabilities qualify only in the isolated observer. No runtime refresh or historical
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
