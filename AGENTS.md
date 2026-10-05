# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **October5: exploration diagnosis complete; test lower-noise targets next.**
  All196,608 logged actions replay with matching accounting; coverage barely
  changes. Nine frozen GPU diagnostics pass: action-dependent bonus variation
  is small, and mean UP-minus-DOWN bonus is negative in all three seeds.
  The tight F64 discrepancy is isolated to cooperative reduced-input math;
  native-F32 GPU readout agrees within2.65e-8. All768 UP/DOWN preferences agree
  between paths; production precision stays unchanged. Retain failed checks.
  [Evidence](docs/results/2026-10-05-freeway-disagreement-diagnosis.md).
  The [soft-target test](docs/experiments/2026-10-05-freeway-soft-disagreement.md)
  is qualified:114 Rust/1,137 Python tests, four GPU checks and production/frozen
  smokes pass; all346 tensors stay unchanged during frozen restore. All three
  fresh32,768-action seeds complete: still zero rewards, almost unchanged bonus,
  near-uniform policy and coverage. Guards/counters/checkpoints and CPU replays
  pass; no frozen evaluation or automatic extension. [Result](docs/results/2026-10-05-freeway-soft-learning.md).
  Removing target sampling noise is insufficient. Next test predicts the
  existing detached CNN embedding: position is more readable there than in the
  categorical target. No coefficient/budget change, scripted aid, new encoder or
  broad matrix. Qualify and declare the changed target width/scale before launch.
  Freeway remains unsolved.
- **October 4 user decision: focus exploration on unlocking Freeway for CDP.**
  The research-order question is resolved. Implemented and qualified the roadmap's
  small GPU-native action-conditioned latent-disagreement bonus against fresh
  extrinsic-only CDP, seeds1009/2017/3019. Start with32,768 aggregate actions per
  arm/seed; all six runs finished October4 at23:00 UTC. All rewards are zero;
  the exploration channel is active but policy entropy remains near uniform.
  Six guards/counter audits/finite checkpoints and CI295 pass. No frozen
  evaluation or new GPU job follows the predeclared all-zero branch. Diagnose
  action-dependent bonus contrast and coverage before changing the budget or
  mechanism. See `docs/results/2026-10-05-cdp-freeway-exploration.md`.
  Actual updates are8,131/run; earlier8,135 prose was a planning error.
  Alignment, detached gradients,
  zero-scale equivalence, restore and short overhead checks pass; see
  `docs/results/2026-10-04-cdp-exploration-qualification.md`. Diagnose the result
  before declaring any larger budget. Freeway remains unsolved by this recipe.
  No scripted UP, persistent action overrides, new pixel encoder or CPU feature
  readback. The five-game representation comparison is deferred, not completed
  or automatically restarted. Keep PR31 as the dashboard.
- **October 4, 20:42 UTC: investigate zero-reward learning before more runs.**
  The user redirects the active comparison to diagnosis. The serial study is
  stopped: seven Freeway pairs are audited; CDP seed3019 is interrupted at
  126,408 actions/31,541 updates and retained. No GPU worker remains. Do not
  restart the matrix automatically. Trace actions, raw/stored rewards, policy
  advantages and optimizer activity; distinguish sparse exploration failure
  from adapter/learner bugs before choosing the next learning experiment.
  [Diagnosis](docs/results/2026-10-04-freeway-zero-reward-diagnosis.md): all
  1,526,408 collected training actions yield zero rewards/advantages. Uniform
  random ALE controls score0, scripted UP21–23; rewards/state match raw ALE.
  One excluded1024-action/195-update GPU reward-pulse probe passes: real rewards
  reach replay and produce nonzero advantages. Same-code Seaquest learns, and
  published DreamerV3 Atari-100k Freeway scores are also zero across five seeds.
  No learner fix or larger comparison is justified by this floor result alone.
  The user now selects the exploration test above; do not repeat the pulse or
  silently restart/extend the matrix. All five-game work remains incomplete.
- **Retained user goal: train CDP on the selected Atari games and compare with
  Dreamer RGB and Tiny JEPA.** Diagnose reward discovery before deciding how
  this comparison and the exploration experiment below proceed. The
  [declaration](docs/experiments/2026-10-04-cdp-atari-comparison.md)
  uses the qualified small recipe, three learner seeds and fresh matched
  controls; Tiny is pretrained/frozen, with its extra experience disclosed.
  The user confirms Boxing, Pong, Freeway, Breakout and Qbert: 45 fresh small
  runs (five games x three methods x three seeds), with frozen evaluation.
  Execute Freeway first; both excluded Tiny qualification smokes pass.
  The first CDP launch stopped before training on an allocation warning; retain
  that failure. The user subsequently authorizes proceeding unless the GPU is
  wedged: standalone allocation warnings are logged, not approval gates. The
  proposed two-warning/first10-second restriction is superseded, not required.
  Do not reopen the cancelled historical queue. The newly authorized CDP
  exploration study is separate from this stopped extrinsic-only comparison.
- **October 4: the user adopts CDP on the main path.** The authoritative roadmap
  is [docs/kindle_single_life_dreamer_plan.md](docs/kindle_single_life_dreamer_plan.md).
  Fast iteration, a faithful RGB control and the CDP evaluation are complete.
  Develop the small jointly learned CNN + deterministic latent-prediction
  Dreamer agent. No frozen Tiny/DINO/Large requirement or RGB visualization
  decoder. RGB remains the reference/fallback; LeVJEPA is a later video/3D
  hypothesis, not an obligation for 2D.
- **Current: exploration/reward with CDP (Phase 3).** Test unassisted
  Freeway: extrinsic-only CDP versus one GPU-compatible intrinsic mechanism,
  three learner seeds 1009/2017/3019. Start at32,768 actual aggregate
  actions/seed on Size1M/N8/B8/T16/H15/R32/microbatch 8/replay 100000, sticky 0.25,
  full actions. Select the mechanism and declare exact finite budgets/settings
  before launching. No random-action aid, new reward shaping or pretraining.
  Then confirm on one predeclared second exploration game, planned as Venture.
  Do not repeat the completed Seaquest comparison unchanged.
- **CDP is the roadmap baseline; the CLI default is still learned RGB.**
  New main-path declarations must pass `--cdp`. A documentation edit does not
  change runtime defaults or launch a campaign. Keep qualified split rates,
  cosine loss and `ac_grads=false` until a separate comparison supports changes.
  No old CPU feature-readback visitation workaround, additional representation matrix,
  unchanged mastery queue or automatic budget extension.
- After Atari: video priors for dynamics/behavior, mind-games vkQuake2 then TMNF,
  a small GOG/Wine panel, held-out cross-game adaptation/retention, then swarms.
  One effective actor comes first. An asynchronous actor/learner is a later
  measured response to real-time latency/debt, not the next engineering project.
- The unstarted posterior-Tiny ablation is deferred. The historical 45-run12M
  matrix remains cancelled: 24 completed/audited and 21 unstarted cancelled.
  Never restart its queue/drain services or workers. Preserve
  `runs/representation-learning-20260928.kjidlR/queue-cancellation.json`.
  The ten-attempt cap applied to RGB replication, not every subsequent study.

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
- Current qualified runtime backend: Meganeura main `592a2f5a`, Blade `e349cddf`.
  The [October4 refresh](docs/results/2026-10-04-meganeura-main-qualification.md)
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
zero swap. Review CPU test filters: Meganeura has unignored GPU tests.
Only the user merges; commits/pushes are allowed. Keep history linear.
