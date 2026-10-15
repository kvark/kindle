# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **Active goal: understand the remaining JAX gap and establish how to reach
  similar Meganeura performance.** It remains open. Latest isolated complete
  RGB12M/B32/T16 is54.224ms versus JAX26.632ms,2.04x; B8 retains22.780/12.737ms,
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
- **Latest convolution family rejected:** [result](docs/results/2026-10-15-convolution-family.md).
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
  Reconcile its existing ensemble workload separately from RGB/JAX; do not
  resurrect the old18-way projection or assume decoder cost applies to CDP.
  Remaining structural feeding/execution investigation stays active.
- Keep isolated candidate`99f8aab0` with `KINDLE_DEVICE_MEANS=1` under
  `runs/categorical-gpu-20261015.lkZAlUAS`; native`28864210` remains its
  [general-ordering control](docs/results/2026-10-14-convolution-scheduling.md).
  Backend sources remain `runs/convolution-scheduling-20261014.BJ5AVZFt`.
  Do not rebuild unchanged code for docs or confuse neighboring user work with
  this candidate. Upstream rechecked October15: `99e3de27`/`e7c3f7b`, unchanged.
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
