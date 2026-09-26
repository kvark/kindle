# Kindle working direction

Kindle is a Rust agent that learns while acting. Games are the first testbed;
explicit rewards and human guidance are allowed. Favor minimalism, expressiveness,
safety and speed. Keep learning/inference on Meganeura + Blade; Python is for
adapters, controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Product and research

- Keep one authoritative plan: `docs/kindle_single_life_dreamer_plan.md`, with
  one current game-status table and direct rollout/world-report links.
- Maintain the current PR description (currently `https://github.com/kvark/kindle/pull/29`)
  as the user-facing status dashboard: timestamp, completed work, active phase,
  next actions, results and known limitations. Do not maintain a separate STATUS.md.
  Update at meaningful phase boundaries, not every poll. Keep detailed gates in
  the plan and raw chronology in runs; link the PR prominently from README.
  Distinguish prepared, running and completed experiments; distinguish runtime
  correctness from gameplay success. State the next decision and keep measured
  bottlenecks separate from untested explanations.
- Prioritize a reliable single actor: Atari, accelerated playing plus learning,
  video/world pretraining, mind-games vkQuake2/TMNF, GOG/Wine, then held-out
  cross-game adaptation and retention. Strong GOG/transfer results precede swarms.
  Do not build a concurrent learner service.
- Current gameplay uses native causal LeVJEPA, not DINO. The frozen frontend's
  16-arrival chunks reset perception only; episode boundaries also reset belief.
  The prediction head reads the deterministic prior before observing its target.
- September 20 model-sizing decision: retain causal-video JEPA and train a
  ViT-Tiny/16 frontend (12 layers, width 192, three heads; 5,486,592 parameters).
  Keep the Dreamer12M RSSM and 7×7×64 observation contract initially. Do not
  substitute a plain RGB/DINO encoder or truncate Large weights. Tiny requires
  its own native pretraining, held-out validation, checkpoint identity and GPU
  numerical/timing checks; shape support is not a trained or qualified encoder.
  Native pretraining is staged at `exp/levjepa-tiny-pretrain-20260920` (1d9f0e96),
  on Meganeura cc5fea74 / Blade eaff5092, including upstream dbb43648's RMSNorm
  fix. Qualified trainer package 60f7060b stays unchanged; observation-only
  package cfa1749a adds a Vulkan budget getter. Readiness, source/package identity,
  full-gradient/AdamW/EMA/763-tensor restore and independent causal/N6 checks are
  linked from `runs/levjepa-tiny-cpu-20260920.e4QkQH/README.md` and the research plan.
  Preserve the earlier AMD-selection/strict-trig failures and explicit primitive
  accuracy-bound revision; no full-model/game gate changed. Use NVIDIA-only
  Vulkan loader selection for environment-independent test contexts.
  Fresh corpus oTjdon contains 250,000 observations /999,112 emulator frames with
  whole-recording splits. Include all offline experience in comparisons.
  Seed 743's 4,096-update pretraining in `runs/levjepa-tiny-pretrain-20260921.JaPZpW`
  completes in 84.18min; guard and read-only `audit_training_v2.py` verify all nine
  complete checkpoints and EMA exports. Preserve the first JSON-presentation
  reader failure; the writer is terminal. Mean data/native times are .78/.45s,
  not GPU utilization. Final encoder: 7fe9b252; own initialization: 7bc344f3.
  `runs/levjepa-tiny-quality-20260921.ojeZgt/results.md` retains the completed
  fixed frozen comparison: all five noncollapse screens pass; pooled Pong
  position R² improves .904→.938, but history-motion R² falls .508→.333 with severe
  paddle outliers. Keep all targets, RGB/constant controls and whole-seed splits;
  no test-driven retuning or general quality/adoption claim.
  Opt-in Tiny gameplay/restore selection is adopted byte-identically from
  c20915ba on `exp/levjepa-tiny-gameplay-20260921`, using unchanged gameplay
  Meganeura 589d73ab /Blade 2accfeee. YF4BKS preserves native 998078ca and its test;
  754 Python/93 Rust CPU checks pass. Its own dense-reference and exact N6/serial
  GPU checks now pass in `runs/levjepa-tiny-gameplay-gpu-20260921.X6TnAI`.
  `runs/levjepa-tiny-gameplay-pixels-20260921.NQLh0I` completes all four bounded
  acting/learning/save/restore phases: 3,840 actions /611 updates per arm, exact
  complete frozen state and full replays/videos. Large matches retained state
  and training/frozen trajectories exactly. All writers are terminal; only
  documented readers are reusable. All 754 main Python CPU checks pass; compiled
  Rust/Cargo inputs match the qualified source without a rebuild. Keep the main
  perception-probe helper unchanged because the completed quality run pins it;
  its standalone Tiny CLI option remains in staging. Large stays the default; all
  seven synthetic CI GPU canaries remain selected. No old declaration, learner,
  replay-ratio, vocabulary or game gate changes. Pretraining-only extensions do
  not automatically update the gameplay backend. Use the same qualified runtime
  for both sides of a later package comparison; different pretraining corpora are
  not a pure size ablation. Initial short-loop times are 194s Tiny /289s Large,
  including replay warmup, not matched-order or steady-state qualification.
  Matched-order cost completes in
  `runs/levjepa-tiny-throughput-20260921.cY1QjK`: Tiny A /Large A /Large B /Tiny B,
  10,008 actions /2,153 updates each, timing actions 6,000–10,008. Require exact
  same-arm complete state/non-timing reports/trajectories, >=10% total and >=50%
  observation time reduction in both orders, <=10% repeat drift. Review each
  phase before individually declaring the next; no heavy CPU work during timing.
  All four windows and the complete comparison pass: total time falls 26.0–26.9%,
  observation time 73.6–74.8%; all same-arm state/moments/reports/trajectories are
  exact. Tiny still runs at .897–.902× aggregate real time, ~87% learning. Keep
  these terminal writers; only unrecorded audit/compare readers are reusable.
  The separate full-budget package comparison is
  `runs/levjepa-tiny-breakout-20260921.ghJPWG`: fresh seed0 Tiny training starts
  at 05:14 UTC for 200,004 actions with the unchanged 12M/R256/full18 recipe. Seven
  CPU budget/refusal tests and all direct input pins pass; qualified native 998078ca
  is reused without rebuilding. All four phases now complete: 200,004 actions /
  49,652 updates, 4.108h training; frozen mean 10.9167 versus .9310 control, two-wall
  successes 0/24 versus 0/29. Zero frozen updates/cutoffs, exact complete state,
  common initial learner state against Large, all-stream replays/videos and
  all four guards pass. Preserve these terminal writers; use unrecorded readers.
  Large's matched-budget mean 30.7917 is better; speed alone does not justify
  adoption, and different pretraining corpora do not establish insufficient Tiny
  capacity. The separate single-variable pretraining ablation completes in
  `runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN`: own initial encoder
  7bc344f3 versus retained pretrained 7fe9b252, same causal 5.49M architecture,
  fresh seed0/200,004 actions/49,652 updates/12M/R256/full18. All four phases and
  pair audit pass data/state/control/replay checks: frozen mean12.4583 versus1.10,
  two-wall successes0/24 versus0/30, zero evaluation updates/cutoffs. Common initial
  learner state remains exact. Preserve all terminal writers; only unrecorded
  readers are reusable. Pretraining shows no downstream benefit in this one seed,
  not a reliable negative effect or capacity limit. See its results.md/videos.
  This is a diagnostic untrained-encoder control, not a random-encoder product
  direction. Keep the trained causal-video JEPA target, old four-action hold and
  two-wall gate. No default adoption or three-root proof follows from either pilot.
- Vector collection shares one learner/policy, not causal histories. Preserve
  independent visual caches, belief, RNG and replay streams. Count actual
  interactions and retain replay-ratio credit; report aggregate/per-stream time.
- Change one scientific variable per comparison. Declare budgets, seeds, controls
  and final-policy gates before running. Keep failures, all completed episodes and
  unfinished tails. Online wins and adapter fixtures are not frozen competence.
- Verify complete saved state, optimizer moments and actual encoder identity.
  Restore does not recover replay, RNG or live belief equivalently.
- Evaluate prior world forecasts separately from posterior estimates and policy
  competence, with persistence/unrelated-action/zero-reward controls and event
  counts. Feature prediction is not imagined RGB. Missing pretraining labels are
  not NOOP/zero. Privileged game observers are never policy inputs.
- Before backend diagnosis, inspect latest upstream fixes. Keep a qualified
  package fixed across an active campaign; updates need a separate comparison.
  September24 inspection finds relevant BCE/LogSoftmax, gradient-buffer and
  RMSNorm fusion fixes after gameplay589d73ab. Qualify current upstream before
  new learner roots; preserve Qbert seed0's completed old-runtime pair unchanged.
  The isolated9746c9ac/Bladefbb4f28c preparation and applicability audit are in
  `runs/meganeura-correctness-refresh-20260924.Be6kq9/README.md`.
  The source audit alone is not qualification or a diagnosed cause. Include
  independent value/gradient references: same-backend parity can share a bug.
  Profile dominant stages before optimizing. Readback waits are not GPU idle.
  Preserve gradients, complete state/moments and action traces; use untraced
  matched-order timing. A lower replay ratio is a learning ablation, not parity.

## Current work

- Historical qualified block runtime: native `886bae68`, Meganeura `589d73ab`, Blade
  `2accfeee`; immutable episode adapter `8dc0b98`. Block products improved fixed-
  recipe throughput by 27.3%. Source cleanup does not rebuild or relabel it.
- `runs/pong-block-confirmation-20260916.rBwdGF` is complete: fresh roots
  2017/3019/1009 pass with 24/24, 23/24 and 24/24 frozen wins, versus 0/76 combined
  untrained. Cross-root initialization, complete state, all replays and six whole-
  stream videos verify. All native phases, pair writers and `finish` are terminal;
  only documented readers may be rerun. Boxing and Pong now meet their three-root
  gates; five-game reliability is unfinished. Details: `docs/experiments/README.md`.
- The original xPz5ud queue and its reserved `pong/seed2017-train.stdout` stay
  terminal. Never remove the hold, restart old followers or count historical
  root 1009 as a root in the new matched campaign.
- Compact-encoder cost passes, but its first Breakout package regresses and the
  completed pretraining ablation shows no benefit in one seed. Current world
  checks complete in `runs/tiny-world-one-step-20260921.U7yHOa` and
  `runs/tiny-world-horizon15-20260921.bKmiUF`: first four pretrained Tiny matches,
  1,126 actions, exact strict/forced one-step replay and exact H1 overlap in the
  16,470-target horizon15/all-origin diagnostic. Complete241-entry/146-moment
  frozen state passes; scalar collection metadata6->1 is explicit. Feature/reward
  forecasts beat controls at every horizon, continuation is worse, with only22
  positive rewards/four terminals. Recorded future actions condition forecasts;
  this is not counterfactual validation or a cause of policy failure. Preserve
  all terminal writers; use only unrecorded readers. See both results.md files.
  The runner-only history port bf0c5c0 passes all three native windows in
  `runs/tiny-checkpoint-history-20260921.kVSoYg`: original/history3,840 actions and
  610 updates, exact final full state/trajectories, historical1,920-action restore
  96 actions/zero updates, all six streams replayed. Both staged and main CPU
  suites pass768 tests; source/tests are adopted byte-identically. Preserve the
  three terminal writers; use only unrecorded audit readers.
  `runs/tiny-freeway-exposure-20260921.ejKgSH` completes all five native phases:
  fresh seed0,400,008 uninterrupted actions/99,652 updates, pretrained Tiny7fe9b252,
  unchanged native998078ca/12M/R256/full18/N6, probability.5/hold64 assistance.
  Frozen midpoint/final/control each complete75,000 unassisted actions,36 natural
  rounds,zero updates/cutoffs: means30.50/33.0278/0 and successes36/36,36/36,0/36.
  Complete241-entry/146-moment states,common initial learner state,all six replay
  streams and decoded whole-stream videos pass; pair.json binds the full result.
  Training takes7.969h,.929x aggregate/.155x per-stream realtime,89.4% learning;
  these are wall times,not utilization. All guards pass; no child remains.
  Preserve all terminal writers; only unrecorded audit/pair readers are reusable.
  Both checkpoints pass,so extra exposure is not shown necessary. The separate
  confirmation `runs/tiny-freeway-confirmation-20260922.tij9QW/completed.json`
  now verifies all fresh roots1009/2017/3019: final36/36 each,means32.9167/31.6944/
  33.25,versus controls0/36 each,mean0. All midpoints also pass36/36. Each root
  completes400,008 actions/99,652 updates in about8h; all frozen arms have75,000
  unassisted actions,zero updates/cutoffs. Full241-entry/146-moment state,all
  replays,decoded videos and guards pass. The360-pin cross-root reader verifies
  the same complete recipe/runtime/encoder and31 distinct initial parameter
  tensors per pair,covering world/behavior/slow value,with zero initial moments.
  Its nine CPU fixtures pass. This is learner-seed reliability conditional on one
  pretrained encoder,not pretraining reliability or an isolated size advantage.
  Preserve the first seed2017 attempt,incomplete at45,714 actions/11,078 updates
  after loss of its host guard,and the ownership incident in
  `runs/freeway-guard-interruption-20260922.2xt6tnn_`; do not resume it or combine
  its experience with another run. The user explicitly approved one fresh
  replacement in `runs/tiny-freeway-seed2017-replacement-20260922.12z27y72`.
  The separate CPU training-replay completion also passes all400,008 actions;
  the prior CPU scope's absent result is not relabeled as a pass. All native
  phases,pair writers and `complete.py --record` are terminal; only unrecorded
  audit/pair/complete readers are reusable. All Freeway guards/learners are terminal.
  Boxing,Pong and Freeway satisfy their fixed gates: five-game status is3/5.
  The Tiny Qbert pair completes in `runs/tiny-qbert-exposure-20260924.3aGHUA`:
  fresh unassisted seed0,400,008 actions/99,652 updates in8.283h,same trained
  Tiny/native/runtime. Frozen midpoint/final/control score2/24,22/24,0/24 first
  pyramids with means1,156.25/4,811.4583/120.8333. Final passes the pyramid fraction
  but fails mean15,000; five-game status stays3/5. Full state/moments,all-stream
  replay,videos and all five guards pass,zero frozen updates/cutoffs. Preserve
  the original midpoint v2/v4 declaration-label audit failure. The separate
  twelve-test `protocol_completion.py` changes only that expected header field
  for its exact pinned declaration; no native work or gate is changed/repeated.
  Use its unrecorded `audit PHASE`/`pair` readers; all writers are terminal.
  Keep this old-runtime pair and trained encoder fixed for comparison.
  Current-upstream GPU evidence is in
  `runs/meganeura-correctness-gpu-20260924.y1vWOG`,separate from CPU preparation.
  Review each declared phase before follow-up;
  no automatic queue.
  Both Tiny frontend checks,eleven independent primitive regressions and both
  production world/behavior gradient fixtures now pass with unchanged bounds.
  The separate pixel/state/restore group in
  `runs/meganeura-correctness-pixels-20260924.u3ISw0` also completes3,840 actions/
  611 updates,exact241-entry/146-moment frozen restore and all-stream replays/video.
  All those writers are terminal; only unrecorded readers are reusable.
  Matched backend-only timing completes in
  `runs/meganeura-correctness-timing-20260924.mYGvjj`:control A,candidate A,
  candidate B,control B. Keep encoder7fe9b252,recipe,per-arm10,008 actions/2,153
  updates and all predeclared exact-state/trajectory/drift gates. No heavy CPU work
  during timing. All four windows pass: total time falls7.5–7.8%,with exact
  same-backend complete state/moments/non-timing reports/trajectories. Candidate
  aggregate realtime is.974–.975x,about91.8% learning by wall time,not utilization.
  Default compatibility also passes in
  `runs/meganeura-correctness-defaults-20260924.M6s2on`: actual LinearNorm/SiLU
  independent gradients,Large dense causal and N6/serial references,and102
  combined frozen actions with exact241-entry/146-moment state. Preserve the
  original CPU service-state refusal; separate CPUv2 passes94 tests,fmt and
  Clippy without production changes. All native writers are terminal.
  Adopt the exact six-file8be26783 production delta: nativea761ee5c,
  Meganeura9746c9ac/Bladefbb4f28c. No model/loss/encoder/recipe changes or native
  rebuild. Full learning improvement remains unproven. The fresh backend-only
  Qbert seed0 comparison completes in
  `runs/correctness-qbert-comparison-20260924.Q4NZyO`:400,008 actions/99,652 updates,
  7.536h,midpoint2/24 pyramids/mean1,144.79,final16/25/3,414,control0/24/120.83.
  All full states,replays,videos and guards pass; the game gate fails. Preserve
  the original old-backend schema-reader failure/absent pair.json. The separate
  eight-test `pair_completion.py` uses the already-qualified same-backend schema
  with the unchanged strict validator; its31-pin completion passes. No native retry.
  The complete `runs/qbert-replay-exposure-20260924.AEyDRM` pair tests R64:
  matched400,008 actions/24,913 updates score1/24 pyramids,mean1,139.58;
  primary1,600,032/99,915 scores24/24,mean8,673.96; control0/24,mean120.83.
  All state/moments,replays,videos and guards pass,zero frozen updates/cutoffs.
  Training takes9.368h,3.161x aggregate/.527x per-stream realtime,not utilization.
  Initial241-entry state exactly matches R256. More exposure helps this history,
  but the mean15,000 gate still fails. Preserve all terminal writers and the
  predeclaration restore-CLI fixture correction. No confirmation roots yet.
  Fresh `runs/qbert-r64-3m2-20260925.FrriIH` tests only doubled exposure at R64:
  3,200,064 actions/199,917 updates,immutable1,600,032/3,200,064 checkpoints,
  same nativea761ee5c,Tiny7fe9b252,seed0 and full gate. Twenty-six CPU checks pass;
  All five phases now complete: training18.799h,3.151x aggregate/.525x per-stream
  realtime; midpoint24/24 pyramids/mean8,673.96,primary final22/27/12,595.37,
  control0/24/120.83. Final fails both90% pyramids and15,000 mean. Full241-entry/
  146-moment states,exact prior initial state,all3.2M training actions and frozen
  replays/decoded videos pass,zero frozen updates/cutoffs. All guards and CPU
  writers are terminal; preserve them. No confirmation roots or budget doubling.
  No equivalent warm resume: checkpoints omit replay/live belief/RNG.
  September26 user priority is throughput before another long learner root.
  Isolated4c988a0 in `runs/meganeura-optimizer-20260926.TfWlJI` stages current
  Meganeura0dbfcc00 (optimizer arenas/batching,egglog3,Windows GEMV),sameBladefbb.
  CPU preparation and source audit are not qualification. Preserve the original
  packaging PATH failure; the separate completion keeps its evidence unchanged.
  Require independent optimizer references,full world/behavior gradients,Tiny/
  Large compatibility,state/moments/restore,pixels and untraced matched timing
  before adoption. Main and completed packages remain9746. Then separately
  profile the complete learner with Vulkan timestamps and test GPU-resident
  synchronization. Existing profiles disable optimizers and are insufficient.
  Measured3.2M wall costs:74.19% learning,22.77% observation,2.24% environment;
  mean update251.00ms includes20.90ms GPU->CPU->GPU parameter sync. Six environments
  already batch inference; trainingB16/T64 imagines1024 starts. The95 posterior/
  imagination readback batches per update include computation waits,not measured
  idle time. No NVML,concurrent learner service or undeclared recipe change.
  Declare/review every phase separately. Never mix backends within a campaign.
  Do not replicate unchanged failed
  recipes merely to occupy the GPU. Reuse the prepared fixtures and CPU evidence;
  `runs/breakout-gradients-20260920.dtzN0w` completes all four individually reviewed
  world/behavior gradient checks at eighteen/four actions. The twelve canaries in
  `runs/breakout-state-20260920.iuGWoA` also pass: all six same-width pairs,
  retained eighteen-action anchors and fresh-process repeats are exact. Preserve
  both terminal groups. Zero-update/common-shape initialization and restore also
  pass in `runs/breakout-initial-completion-20260920.DAvUch`: all shared parameters
  and moments are exact. Preserve ARoQmX's original serialized-hash reader failure;
  the separate completion verifies identical parsed headers and payload bytes
  without repeating native work. Both groups are terminal; use only completion
  `audit 18`/`audit 4`. All eight N6 pixel/replay/episode/memory and vocabulary-
  refusal checks in `runs/breakout-pixels-20260920.YrLIfy` now pass; preserve all
  terminal writers and use only its `audit NAME` readers. The five replay/schema/
  test files are adopted byte-identically from 8092790, with 742 passing main CPU
  tests; native and default full-action behavior are unchanged. The minimal-action
  learning recipe is not proven better. The seed-zero, 200,004-action eighteen-
  action Large pilot in `runs/breakout-action-pilot-20260920.kNeotb` completes
  all four phases: 49,652 training updates, frozen mean 30.7917 versus .9655
  untrained, two-wall successes 0/24 versus 0/29. Complete state, replays, videos
  and guards pass; this is learning, not a win. All four writers are terminal.
  Its `results.md` records the pair. The `a4-train/HOLD.md` reserves the four-action guard
  directory before native launch. Never remove it to resume the original queue.
  The two-width pilot is now incomplete by scheduling decision, not native failure.
  Preparation and its 105 inputs stay unchanged. Keep controls, videos and the
  two-wall gate; use `audit NAME` and `pair 18`, not the unfinished two-arm summary.
  Sizing evidence: `runs/model-sizing-20260920.kPIOWC/README.md`. Do not repeat the
  memory-heavy allocation-plan qualification on this unchanged runtime.
  Preserve the ~22:49 UTC unintended pipeline-test overlap documented in
  `runs/levjepa-tiny-cpu-20260920.e4QkQH/test-selection-20260920.md`: the process
  exited without model execution or an observed kernel fault. Do not treat that
  interval as uncontended timing. The staged GPU-only library test is now ignored;
  future CPU checks still require exact reviewed test names, not assumed-safe
  module filters.
- Preserve completed/failed `runs/` writers, source worktrees, packages and
  artifacts. Only documented audit modes are reusable. Correct reader failures
  separately; never overwrite failed evidence or rerun native work to repair it.

## GPU operation

- Ordinary bounded GPU work is authorized on driver 580.178.04. **No NVML calls**:
  no nvidia-smi, bindings, legacy health logger or vendor diagnostics. Use the GPU;
  do not add CPU learning fallbacks. Telemetry unavailable without NVML is
  unmeasured, not zero or healthy.
- Serialize GPU-heavy jobs. Use `python/examples/gpu_host_guard.py` around the
  direct native-bearing process, not a scheduler/Cargo/process tree. Bind the
  boot, driver, executable and fixed experiment inputs; review each result before
  individually starting its successor. No retries or automatic followers.
- Own each new guard with a persistent systemd user service, as tested in the
  September22 ownership incident: `Restart=no`, `KillMode=control-group`, and a
  service deadline slightly beyond the guard deadline. The guard still owns
  only its direct native-bearing child. Do not rely on a tool session to keep
  the guard alive; service cleanup prevents an orphan if the guard disappears.
- Retain native device assertions and >=2 GiB Vulkan estimated budget headroom
  after GPU stages. Budget-minus-usage is not physically free or peak VRAM.
  Current declarations bind boot `4f5152d1-e5fd-46cf-a0c4-06534c430d26`; a changed
  boot/driver requires a distinct declaration, not editing historical results.
- Stop on kernel faults, native failures, invalid evidence or incomplete budgets.
  No reset/reload/reboot/power cycle/driver changes without new user approval.
  Follow `docs/gpu_incident_response.md`. The four historical Xid 62/154 failures
  remain unexplained; successful no-NVML runs do not prove causality or safety.
  Failed 75dfe, 0a98775/02b600a1 and 070f4b51/7db0d05c bundles remain quarantined.
- Keep the current campaign's pinned `gpu_guard.py`, `gpu_host_guard.py` and
  `audit_canary.py` unchanged. The legacy guard is retained for readers/helpers;
  its NVML launch paths are not permitted.

## Implementation discipline

Keep changes reviewable. Research chronology belongs in run artifacts or the
archived research branch, not accumulating production docs. Delete unused staged
code instead of building speculative frameworks. Do not rebuild unchanged native
binaries or repeat complete qualification for documentation-only changes.
For future experiments, prefer one small source/config/artifact declaration and
result record over recursively pinning entire research histories. This does not
change existing declarations, safety checks or acceptance gates.

Minimize assistant monitoring overhead. Let existing guards and phase auditors
handle containment and routine validation. Check long training runs about every
30 minutes or at completion, not every minute; report meaningful results or
problems rather than incremental counters. Do not repeatedly audit unchanged
evidence or build extra tooling merely to occupy a training window. Keep GPU
successor phases individually reviewed; less assistant polling does not weaken
the native guard or authorize automatic successors.

Preserve unrelated user work. Only the user merges PRs; commits/pushes are allowed.
Keep history linear. Use one CPU, 2 GiB and zero swap for heavy CPU preparation,
with private targets/packages; do not compile during matched timings. Meganeura
has an unignored GPU library test: CPU-only checks require reviewed module filters.
Use relevant tests, formatting, Clippy and native numerical checks.
