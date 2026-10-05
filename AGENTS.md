# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- **October5 goal: unlock Boxing, Pong, Freeway, Breakout and Qbert with CDP,
  and test whether DreamerV3 scores are reachable with less budget.** All five
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
- **Next reviewed allocation:** [three fresh500k-action Pong seeds](docs/experiments/2026-10-05-cdp-pong-budget.md),
  exact same model/recipe; only the finite learning budget changes.124,939
  updates/run,1.5M new actions total. Frozen final/actual-initial controls use
  new held-out base3,000,000,000, first3 episodes/stream,200k-action cap and full
  audits/videos. Expected3.5–4h;6h service bound,100min training processes.
  The published Atari57 Pong mean is−20.50 at800k frames,−7.16 at2M. Test an
  early-learning floor before changing capacity/loss/exploration. Two current
  Pong seeds have weak reward separation and KL~.57; posterior diagnostics are
  not prior forecasts or proof of feature collapse. No automatic extension;
  persistent weak prediction calls for state/forecast diagnosis before more
  training. Current native is unchanged; no rebuild or duplicate qualification.
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
  user-accepted stretch target. Current mean31.62ms updates spend51% in
  imagination,27% world training; about8x aggregate /1x per-stream realtime.
  GPU utilization remains unmeasured. Optimize measured whole-agent cost.
- No old queue restarts. The stopped45-run small CDP/RGB/Tiny comparison retains
  seven completed Freeway pairs plus interrupted CDP3019 at126,408 actions.
  The historical12M matrix retains24 completed and21 cancelled unstarted entries.
  Their controls/protocols are not the current intrinsic study. Venture, Tiny
  comparisons and new representation matrices are deferred by the five-game goal.
- Backend rechecked October5 after the screen: Meganeura upstream6288f885 has
  only docs/paper/artifact changes over qualified592a2f5a; Bladee349cddf is
  unchanged. No missing runtime fix or reason to rebuild unchanged native.
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
