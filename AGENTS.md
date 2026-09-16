# Kindle working direction

Kindle is a Rust agent that learns while acting. Games are the first testbed;
explicit game rewards and human guidance are allowed. Intrinsic motivation and
experience sharing between independent Kindles remain longer-term goals.

## Priorities

- Favor minimalism, expressiveness, safety and speed. Keep native learning and
  inference on Meganeura + Blade; Python is for adapters, controls and analysis.
- Maintain one authoritative roadmap at
  `docs/kindle_single_life_dreamer_plan.md`: decisions, one current game-status
  table, and direct video/world-report links. Protocols and chronology belong in
  `docs/experiments/` and pinned `runs/` artifacts, not repeated in this file.
  Editing documentation never changes a pinned declaration or acceptance gate.
- Prioritize one learning actor: reliable Atari breadth, accelerated playing
  with training, video/world pretraining, mind-games vkQuake2/TMNF, GOG/Wine,
  then held-out cross-game adaptation and retention. Strong single-actor GOG
  and transfer results must precede swarm work. No concurrent learner service.
- Vector collection shares one learner/policy while retaining each environment's
  visual cache, recurrent belief, RNG and causal replay history. Count actual
  interactions, not vector ticks; preserve replay-ratio credit and report both
  aggregate and per-stream simulated/wall time.
- Follow `/mnt/data/GUIDELINES.md`: compute is cheap, attention is expensive;
  act autonomously within scope. Only the user merges PRs. Commit/push are
  permitted; do not contact other people without authorization.

## Current runtime and next work — September 16

The user explicitly resumed ordinary bounded GPU work on driver **580.178.04**:
use the GPU, temporarily disable **all NVML calls**, and avoid CPU learning
fallbacks. This supersedes old blanket GPU-approval stops, not historical
failures, numerical gates or the prohibition on host recovery.

The [current backend](docs/experiments/2026-09-16-current-upstream-timing.md) and
[same-backend block-matmul](docs/experiments/2026-09-16-current-block-matmul.md)
are qualified. The latter passes full native correctness, complete state and
all ten N6 pixel/restore/override/budget windows. Its two fixed-recipe orders
show **27.3% higher throughput**, about **0.682× aggregate real time**, not
real-time training. All 241 tensors / 146 moments and non-timing reports/traces
match exactly. This is native graph optimization, not reduced training.

- Qualified source `dee38b2`, native `886bae68`, Meganeura `589d73ab`, shared
  Blade `2accfeee`; checked upstream tips are `986f49a` / `92553493`.
  Preserve the completed Cx3Xei, zc0Aig, NaQ1zs, M7whE0 and oVOgnU groups.
  The original t8SQkG depfile-reader failure remains failed; its unchanged
  build was independently verified, not rerun.
- Episode-budget support is carried onto that unchanged native in `8dc0b98`.
  The separate v5vKcE completion verifies 597 Python tests and actual imports.
  Preserve GvkNxZ's failed original import checker and prepared package.
  All six individually reviewed CXHzRj GPU integration phases now pass: default
  training/frozen parity, exact full frozen trajectories/state, first-eligible
  episode stopping and the negative action cap. All 84 initializations / 888,372
  trace records and 25,687 native memory samples verify; all children exit zero
  and are reaped with no fault or NVML. Preserve this terminal group. See
  `runs/current-block-episode-runtime-20260916.CXHzRj/declaration.md`.
- Main now adopts the matching source after the separate 4Smhmw audit: 81
  identical source/build files, 702 Python tests, both formatting checks,
  twelve command records and 5,354 pins. Reuse exact unchanged native checks;
  no new binary was built. Preserve that writer; audit only. The default
  editable extension remains historical; explicitly select the qualified
  source-matched package. See
  `docs/experiments/2026-09-16-block-runtime-adoption.md`.
- Matched Pong is now declared in `runs/pong-block-confirmation-20260916.rBwdGF`
  with 5,329 pins and nine reader checks. Root 2017 training starts at 22:07 UTC,
  direct native child 131029; do not duplicate it. Run order is 2017/3019/1009,
  each fresh on one bundle with 400,008 training actions and the unchanged frozen/
  untrained controls and gates. Each phase is individually invoked/reviewed;
  host polling is one second, native memory coverage remains every GPU stage.
  No automatic successor or recovery. Do not mix historical successful root
  1009 into this new three-root claim. See
  `docs/experiments/2026-09-16-pong-block-confirmation.md`.
- The original xPz5ud queue and its reserved
  `pong/seed2017-train.stdout` remain terminal. Never remove the hold or restart
  that queue or any retired follower. Throughput qualification precedes Pong;
  it is now complete, not a reason for another open-ended qualification loop.
- Breakout's action-width checks are carried onto this runtime in `8092790`:
  742 Python tests, 32 identical native/build inputs and the unchanged qualified
  package. The separate `3c4f87f` fixtures pass 109 CPU tests, fmt and release
  Clippy, with production library code unchanged and all 27 GPU tests ignored.
  Preserve KA1kRO's depfile-name reader failure; the pzWbk8 completion independently
  verifies the unchanged binaries and compiler chains. No GPU result or learning follows.
  Preserve old sources/followers. Finish the declared Pong campaign before a
  separate action-width GPU declaration. See
  `docs/experiments/2026-09-16-breakout-current-runtime.md`.

## GPU operation

- Serialize GPU-heavy jobs. Use the existing
  `python/examples/gpu_host_guard.py` around the **direct process hosting native
  GPU work**, never a scheduler, Cargo, controller, or process that delegates
  GPU work to descendants. Pin a fresh bounded declaration and review each
  result before individually starting the next invocation. No run-all or
  automatic successor unless separately and explicitly declared.
- No `nvidia-smi`, NVML bindings, legacy health logger, recovery-action polling
  or vendor diagnostic invocation. Legacy `gpu_guard.py` stays for retained
  readers/helpers, not its NVML launch/health paths. Host checks are not GPU
  health/utilization. Preserve the no-NVML profiler.
- Retain actual native RTX 5080/device/driver assertions, complete initialization
  observations and fail-fast waits. Current declarations use boot
  `4f5152d1-e5fd-46cf-a0c4-06534c430d26` and driver `580.178.04`; a changed boot
  or driver requires a distinct matched declaration, not editing old results.
- New memory gates explicitly use native Vulkan **estimated budget minus usage**,
  >=2 GiB after GPU stages. This is not physically free/reserved or peak memory;
  those and utilization/recovery action remain unmeasured. Do not replace old
  NVML gates with estimates retroactively or claim N8 qualified from N6 evidence.
- Stop on a new kernel fault, native failure, invalid evidence or incomplete
  budget. Preserve the direct-child result and logs; no blind retry. No reset,
  module reload, reboot, power cycle, driver/package change or other host recovery
  without new user approval. See `docs/gpu_incident_response.md`.
- The four matching September 13–15 Xid 62/154 incidents remain unexplained.
  Passing 580/no-NVML jobs does not establish a driver/NVML causal fix or
  hardware safety. Historical 75dfe, 0a98775/02b600a1 and interleaved
  070f4b51/7db0d05c failed bundles remain quarantined outside their consumed,
  explicitly declared historical diagnostics. Never rerun those attempts.
  For exact IDs/evidence see the September 13 forensics, September 14 pixel
  incident and September 15 interleaved-initialization incident reports.
- Every completed or failed writer, native invocation, source worktree, target
  and package is terminal/immutable. Only documented audit/verify modes may be
  reused. Preserve absent success results and original reader/build failures;
  corrections need separate readers, never rewritten history. CPU checks do not
  qualify GPU behavior, and initialization-only passes are not full qualification.

## Research and learning gates

- Current gameplay uses **native causal LeVJEPA**, not DINO. Its 16-arrival
  chunk prefixes never consume future frames; chunk boundaries reset perception
  only, while episode boundaries also reset belief. The frozen 303M frontend
  is distinct from the trainable action-conditioned RSSM. Keep the established
  DINO controls; no unqualified encoder fallback or representation switch.
- Predict before observing: the JEPA-style head reads the deterministic prior,
  not the posterior containing its target. Preserve reconstruction/future
  controls 0.25/0, 0.25/0.25 and 0/0.25, matched heads and initialization, full
  recurrence, F32 gradients and row-independent microbatching.
- Change one scientific variable per comparison. Record actual interactions,
  updates, wall time, seeds, encoder/data/backend identity and failures.
  Lower replay ratio, larger batch, new action vocabulary or longer exposure
  is a new learning experiment, not an identical-recipe throughput win.
- The five-game objective requires all three fresh roots 1009/2017/3019 to pass
  each game's unchanged final-policy gate and beat separately restored untrained
  controls. Boxing passes; Pong has initial learning/one newer successful pair,
  not new-bundle reliability. Freeway's confirmation and Breakout/Qbert pilots
  fail competence. Keep exact current results and thresholds in the roadmap.
- Pong remains N6/R256, 400,008 training actions, no exploration overrides.
  Frozen trained and untrained controls each use environment root 100000,
  sampled actions, four complete episodes per stream and hard cap 600,000.
  Require >=20 natural games, >=90% wins, mean >=+15 and no cutoffs; untrained
  must fail, trained mean must exceed it. Cap exhaustion is incomplete.
  Include faster-stream extras and report partial tails; never select episodes
  by score. Whole stream-zero movies must accompany complete ALE replay checks.
- Verify actual encoder bytes on restore and complete checkpoint state, including
  all moments/normalizers and seed provenance. Preserve historical executables:
  equal shapes are not identity and backend metadata must not be rewritten.
  Checkpoint restore does not resume replay or live belief equivalently.
- Do not replicate unchanged failed recipes just to accumulate seeds. Keep the
  staged Breakout minimal-action and Freeway/Qbert bounded-exposure hypotheses
  distinct, with matching controls, native/runtime gates and unchanged competence
  thresholds before confirmation. Prioritize experience/reward coverage and
  calibration over speculative perception expansion.
- Before new backend diagnosis/optimization, freshly read upstream main and inspect
  relevant runtime/build fixes and tests. Record the checked tip; carry fixes
  into a separate candidate, preserving active campaigns and packages. Do not
  churn identities for documentation-only upstream changes.
- Profile measured dominant stages before spending days of GPU time. Require
  production losses/all gradients, reset causality, complete state/moments from
  update one, action traces, declared memory headroom and untraced AB/BA timing.
  Readback waits include producer compute and transfers; they are not necessarily
  GPU idle. Submission timestamps are not calibrated kernel idle gaps.
  Grouped-GRU failed exact learning and is not adopted; world-sync fan-out remains
  unqualified. Do not substitute stale-source candidates for the qualified bundle.

## World models, pretraining and transfer

- Evaluate the world model separately from policy competence. Forecast before
  consuming targets; distinguish prior forecasts from posterior estimates.
  Include persistence, unrelated-action and zero-reward controls, positive/
  negative/terminal counts and visual-cache/reset strata. Report MAE and MSE;
  sparse zero-baseline MAE alone does not prove absent reward signal.
  Feature error is not imagined RGB, AUC is not magnitude calibration, and
  another policy's logged return is not an unbiased critic target.
- Common-recording controls are offline diagnostics, not that model's policy
  rollouts. Preserve completed strict/same-model and cross-model results.
  Preselect first-N complete stream-zero matches without score filtering; require
  separate source-matched serial/vector/strict/forced GPU forecast gates before
  using the staged multi-match probe. CPU extraction is not forecast validation.
- Distinguish video-encoder initialization, action-conditioned world pretraining
  and policy-skill transfer. Missing action/reward labels are not NOOP/zero.
  World-only pretraining is staged, not adopted dataset training. Require verified
  ingestion, native/adaptation gates and format-4 offline lineage; never silently
  reinterpret it as an ordinary format-3 checkpoint.
- Hold target titles out of source data/tuning. Measure adaptation and forgetting;
  declare policy/head/optimizer/normalizer/replay/belief resets. Reuse mind-games
  launch, time-control, capture and input, checking its current Kindle API.
  Privileged task observers stay outside policy inputs and training rewards.
- Natural deaths/respawns are allowed; cloning or rewinding live games for training
  is not. Independent initialized vector streams require their own declared
  protocol. Uncapped stepping, accelerated playing plus training and genuinely
  free-running play are different capabilities. Preserve arrival order, actual
  action durations, gaps and bounded training debt.

## Implementation discipline

Preserve unrelated work. Prefer small concrete modules over speculative frameworks;
delete genuinely superseded code and redundant docs. Keep frozen experiment
sources/artifacts intact. Use fresh private build/package directories, one CPU,
2 GiB and zero swap for heavy preparations; do not compile during matched timing.
Use relevant Rust/Python tests, formatting, Clippy and native numerical checks.
Meganeura includes an unignored GPU library test: CPU-only checks must use reviewed
module filters, not an unrestricted library test invocation. Reuse exact unchanged
qualified binaries rather than rebuilding for a new label, and never relabel a
new binary as an already-tested artifact.
