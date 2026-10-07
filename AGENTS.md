# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **October6 goal: reach DreamerV3 quality on Boxing, Pong, Freeway,
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
- **Next reviewed learning allocation:** [three fresh500k seeds on each of the
  five games](docs/experiments/2026-10-06-cdp-centered-five-game-budget.md),
  centered CDP on native6d38eea2/c6376542. Fifteen learners,7.5M actions /
  1,874,085 updates; seed-major Pong/Boxing/Freeway/Breakout/Qbert. Actual saved
  initial and frozen final controls use new held-out base4,000,000,000, first3
  natural episodes/stream,600k-action ceiling, full audits/videos. Expected
  16–18h,24h service bound,100min training processes. The controller's smoke
  audit rehearsal and all45 job declarations pass. No checkpoint lifetime
  resume, game-specific recipe, raw-cosine retry or automatic extension.
  This is longer centered-package learning, not a new five-game loss ablation
  or a completed quality goal. Review every failure or capped cohort.
  The first two completed Breakout seeds remain weak. The declaration now
  prepares a bounded initial/final frozen-state/forecast check for all three
  seeds after the full campaign audit. It is not launched; do not interrupt
  this allocation or add training budget before diagnosis. Breakout probes
  are CPU-tested in7e5df42; their GPU path is still unrun.
  **October7 independent audit caught a natural-cohort bug:** Breakout3019's
  selected24 completed episodes include one timeout; its8.0833 score and
  three-seed5.1944 summary are not valid complete natural-cohort results.
  Finish active Qbert unchanged, then repair stopping/selection and rerun only
  that frozen final on the same checkpoint/seeds/caps, requiring exact agreement
  with the original207,288-action prefix. Retain all old evidence and extra
  compute. Vector v5 uses natural quotas; v4 artifacts retain completed quotas.
  No learner retraining or budget extension. Reconcile before final campaign
  acceptance and the declared Breakout diagnosis; see the allocation's review.
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
  Rechecked19:50 UTC: new Meganeurafc3a2fb adds f32 cooperative acceleration
  and checked geometry; Blade49ec60a changes presentation-damage hints. No
  relevant fix for the current small-CDP learning weakness was identified.
  Keep active jobs/frozen diagnosis on the qualified runtime; review a refresh
  before the next learner change, without repinning completed evidence.
- After Atari: video priors for dynamics/behavior, mind-games vkQuake2 then
  TMNF, a small GOG/Wine panel, held-out cross-game adaptation/retention, then
  swarms. One effective actor first; no concurrent learner service now.

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
- Current qualified CDP/RGB backend: Meganeura main `c6376542`, Blade `e349cddf`,
  native `6d38eea2`; the grouped imagination trial qualifies the new build.
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
