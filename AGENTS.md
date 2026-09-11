# Kindle working direction

Kindle is a Rust agent that learns while acting. Each environment contributes
its own continuing stream of experience; vector collection shares one learner
and policy without joining causal histories. Games are the first testbed.
Intrinsic motivation and experience sharing between independent Kindles are
long-term goals; game rewards and human guidance are allowed while establishing
reliable learning.

- Favor minimalism, expressiveness, safety, and speed. Keep the native learning
  and inference path on Meganeura and Blade. Python is for adapters, controls,
  and analysis.
- Maintain one authoritative research plan at
  `docs/kindle_single_life_dreamer_plan.md`. Keep its claims tied to code and
  measured results. Experiment logs belong in `runs/`, not in an ever-growing
  chronological plan. Keep the roadmap decision-focused, with one current
  game-status table and direct video/world-report links. Put checkpoint-level
  chronology and repeated validation details in the linked experiment reports;
  a documentation update never changes a pinned declaration or acceptance gate.
- Prioritize one learning actor: Atari breadth, video/world pretraining and fast
  accelerated playing plus training, then mind-games (vkQuake2/TMNF), GOG/Wine games,
  cross-game adaptation and retention. Pong's initial-learning gate is achieved,
  not consistent mastery. Require strong single-actor GOG and transfer results
  before swarm learning. Prioritize vectorized environments and batched live
  inference for one shared learner/policy, as explicitly requested. This is a
  collection/throughput protocol, not swarm learning or separate learner services.
  Keep each environment's visual cache, recurrent belief, RNG and replay sequence
  independent. Count actual interactions across all environments, not vector ticks;
  preserve train-ratio credit and report aggregate and per-environment throughput.
- Preserve a measured Dreamer control. Add JEPA-style prediction as a causal,
  action-conditioned objective that predicts an observation before consuming
  it. Predicting the current frozen DINO features from the posterior is already
  the existing feature-reconstruction control.
  Retain the accepted DINOv3 plus causal-prediction Pong controls. Native
  batched DINO is an explicit matched-control candidate, not a default frontend
  switch; require GPU stream parity and a newly declared comparison before use.
  All three LeVJEPA vectorized seeds have completed frozen evaluation. Seed 2
  passes the predeclared mastery gate; seeds 0 and 1 fail, so the recipe does
  not pass the all-seeds gate.
  Do not equate training wins or one frozen seed with three-seed mastery,
  or describe the DINO stepping stone as the full video pivot.
  Native LeVJEPA work and the stronger, predeclared three-seed Pong mastery gate
  are tracked in `docs/experiments/2026-09-06-levjepa-pong.md`. Its 16-arrival
  causal chunks reset only perception; episode boundaries also reset belief.
  Do not confuse chunked prefixes with a sliding window or reset the RSSM every
  chunk. Checkpoint format 3 records the actual frontend and encoding semantics;
  historical format-2 runs require their original executable.
  The fresh vectorized Pong protocol is in
  `docs/experiments/2026-09-06-vector-pong.md`. Do not replace the binaries,
  runners or auditors of an active pinned experiment. Stage follow-on candidates
  separately and keep GPU-heavy checks serialized with measured training.
  The readback worker, timer and host-buffer-reuse hardware/canary/pixel gates
  have completed; preserve their controls and do not restart their queues.
  Original pinned inputs are in `runs/readback-hardware-20260907/manifest.json`;
  earlier results are in `docs/experiments/2026-09-08-runtime-hardware.md`.
  Device-resident imagination is adopted after exact hardware, synthetic and
  pixel checks. It removes redundant host feature/state transfers and scratch
  without changing learning arithmetic. The first Meganeura refresh adopted
  a7e2efd9 (main df11bb0c plus the two required LeVJEPA cache patches), with
  shared registry Blade 0.9.0 and Rust 1.92 minimum. Do not lose cached query
  blocks or cache aliases when updating again. Require logical weights and all
  optimizer moments on restore, excluding only plan-identified Winograd caches.
  Keep backend identity checks and historical executables intact.
  Main now adopts ce80e9cd. Its fresh source-matched integration passes 92 Rust/
  253 Python CPU tests, formatting, both Clippy checks and three focused GPU checks.
  `runs/meganeura-runtime-20260911.Nnfxk4/main-package` is native 1735b8a6:
  source-matched integration evidence, not long-run Atari runtime qualification.
  The separate source-adoption audit freshly reverifies the complete runtime
  and main evidence, binding 346 source/artifact pins including new checkpoints.
  The previous backend 4d45ba3a is upstream runtime e59bd32d plus the required
  cache corrections. The September 11 remote recheck still finds main ce80e9cd,
  superseding the earlier documentation-only 4d669394 check. It fixes generated
  matmul epilogues and now includes the required LeVJEPA cache patches upstream.
  The isolated 1e00e818 candidate at `exp/meganeura-upstream-20260910` changes
  only dependency/identity files. Its 95 Rust/547 Python and 80 focused backend
  CPU tests pass in `runs/meganeura-upstream-20260910.ERT7QD`; all 561 package
  input pins, source/wheel/import identity and historical/active controls reverify.
  Native abf4ae5d is now runtime-qualified on recovered driver 595.91.07 in
  `runs/meganeura-runtime-20260911.Nnfxk4`. All 1,364 input pins, 47 CPU wrapper
  checks, 18 control/19 upstream hardware checks, complete update-1/eight-update
  state and optimizer moments, pixel AB/BA traces and override accounting reverify.
  The unchanged control exactly reproduces its archived old-driver canary and
  pixel results. All six canary/ten pixel native windows pass with at least
  3,303 MiB directly free. Timing ratios 1.005750/1.009460 pass the declared gates,
  but this 0.6–0.9% short-window gain still leaves R256 near 0.573x aggregate
  real time and does not establish learning reliability. Preserve all completed
  gates and packages. Keep queued packages fixed and GPU work serialized; a
  dependency adoption does not rewrite old checkpoints or switch pinned experiments.
  See `docs/experiments/2026-09-11-meganeura-runtime.md`.
  Upstream's block-matmul operator and experimental tuning options are not
  automatically enabled in Kindle by this dependency update.
  The qualified 4d45ba3a package preserves frame-prefix query attention alongside
  upstream's different token-causal blocks and early cache aliases. Its
  95 Rust/547 Python CPU tests and 18 GPU checks pass, including
  production all-gradient and LeVJEPA N4/N6/N8 parity. Both eight-update full-state
  canary pairs also match exactly. The N6 pixel AB/BA and override gate completed
  in `runs/meganeura-refresh-20260909.xfF3AZ`: all 374 pins, complete state/reports/
  traces and ten GPU phases independently reverify, with at least 3,302 MiB
  directly free. Timing ratios 1.004068/0.998255 pass the regression guard,
  not the speedup gate. That source integration passed 92 Rust/253 matched Python
  CPU tests and three main GPU checks before the later ce80e9cd update.
  Preserve the initial mixed-Python-package failure and its old-backend negative
  control. Do not pair main's historical Pong auditor with the newer Atari
  accounting module. The qualified Atari package is `package` (native f6a2b6ad);
  `main-package` (6c630ecb) is source-matched integration evidence, not a long-run
  runtime qualification. Keep their matching Python sources and runners together.
  Historical default extensions are unchanged. Do not restart the completed gates.
  See `docs/experiments/2026-09-09-meganeura-update.md`; keep this backend update
  separate from the still-unrun world-sync fan-out candidate.
  Historical N8 exact pixel pairs reach 8.64–8.70 actions/s, only 0.576–0.580× aggregate
  real time and 0.0720–0.0725× per stream. GPU activity spans 66–70%; 14,212 MiB
  peak usage left only 1,631 MiB of directly reported free memory in the completed
  pilot. The old total-minus-used check omitted driver reservations and does
  not establish the 2 GiB safety gate. Preserve those raw results but withdraw
  the reserve-pass claim. Record memory.free and memory.reserved directly;
  require at least 2,048 MiB measured free before new long-run replication or
  a larger batch. The unchanged pilot has finished; do not restart its queue.
  Keep GPU-heavy work serialized. That earlier backend update was not a major
  speedup; see `docs/experiments/2026-09-08-meganeura-refresh.md` for the tested
  historical package and the preserved audit-only failure plus completed continuation.
  A CPU graph check verifies 588 MiB of F32 visual KV cache per stream: N6/N4
  would remove 1,176/2,352 MiB versus N8, without changing B16/T64 or full BPTT.
  These are logical bytes, not measured free VRAM or throughput. The LeVJEPA
  `memory_candidate_streams_match_serial` GPU test now passes for N4/N6/N8,
  with zero measured dense-feature error against serial encoding and unchanged
  pooled/dense tolerances. This does not establish combined learner memory.
  The fixed-R256 N8/N6/N4 then N4/N6/N8 comparison completed in
  `runs/vector-memory-runtime-20260908.CcWv0d`: every same-N full checkpoint and
  action/episode/reset trace repeats exactly. Select N6 for the unchanged
  repaired native package: 8.55–8.57 actions/s, at least 3,302 MiB directly free.
  N4 retains 4,889 MiB but is slower; N8 retains only 1,630 MiB and fails the
  2 GiB gate. N6 is about 1.2% slower than N8, not a throughput improvement.
  These short runtime repeats do not test training-seed reliability. Preserve
  the completed evidence and pinned inputs. The isolated replication-v2 checker
  at `exp/atari-replication-v2` binds a declared N4/N6/N8 to matching complete
  runtime evidence; real N4/N6 pass and N8 is rejected. Its 504 passing CPU
  tests are not a replication result. Keep all five game gates and fresh seeds.
  A grouped-RSSM gate candidate passed focused output/gradient checks but failed
  exact full-learning parity from update 3. It is not adopted; preserve branch
  `exp/rssm-gate-batching` and `runs/grouped-rssm-20260908.Vodj6w`. Main source
  is restored, but its old release binaries still contain that candidate;
  use the documented isolated package or rebuild before running root binaries.
  Its September 10 CPU postmortem reproduces the failure and verifies an exact
  independent control repeat. First differing report 3 does not locate the first
  gradient difference: zero-LR update 1 still updates optimizer moments. Capture
  intermediate state from update 1 in any future diagnostic; no cause, fix or
  new GPU queue is established. See the grouped-RSSM experiment report.
  The separate `exp/block-matmul` candidate (Kindle 4ae539a/Meganeura 70803c2)
  groups small-batch F32 block products, retaining serial GEMV, large imagination
  batches and original GRU gates. Its 94 focused backend/98 Kindle CPU checks
  and block-only 65-to-2 dispatch counts are not GPU parity or a speedup. Preserve
  its 33-pin evidence and all active queues; no candidate GPU follower or adoption exists.
  See `docs/experiments/2026-09-10-block-matmul.md` for required hardware gates.
  Do not repeat large CPU graph
  compilation alongside training: the first memory-plan probe caused host
  pressure, and its two capped follow-ups failed without yielding smaller-row
  world estimates. Preserve those failures; CPU-only does not mean low impact.
  Track remaining world-training/recurrent/perception costs and profiler
  coverage in `docs/experiments/2026-09-08-device-imagination.md`.
  The isolated `exp/world-sync-fanout` candidate reads shared world weights
  once for the six core inference sessions, retaining backend cache refresh.
  Its CPU checks are not GPU parity or a speedup; require full state/trace
  equality, memory headroom and an AB/BA timing gain before adoption. Preserve
  all existing queues. See `docs/experiments/2026-09-09-world-sync-fanout.md`.
  The current profiler's alternate mode recovers queue-submission coverage,
  not per-dispatch kernel detail or verified idle gaps. Completed captures are
  diagnostic artifacts, not another pending queue or traced speed benchmark.
  The tested Python package is isolated; the default editable extension remains
  the pinned historical control. Select the documented package or build current
  source into a fresh package for new experiments. Do not overwrite controls.
- Change one scientific variable per comparison. Report real interactions,
  learner updates, wall time, model/data provenance, all seeds, and failures.
  A short integration test or an historical score is not a matched benchmark.
  Match head structure and initialization when comparing objectives, and version
  changed heads or intrinsic hash schemes instead of reinterpreting old state.
  Verify the actual encoder file on restore; matching shapes are not identity.
  Require complete checkpoint tensors; a detected torn save is not atomic recovery.
- Profile learner stages, synchronization, and GPU idle time before committing
  days of compute. Check existing branches and local run artifacts before
  repeating old experiments. Preserve corrected full-precision gradients and
  full-recurrence row microbatching when integrating backend work.
  Judge useful throughput at the declared replay ratio, not GPU busy percentage
  alone. Retain the GPU memory safety margin; a larger batch needs both a timing
  win and a learning-quality comparison before becoming the new control.
  Lower replay ratios are separate learning-throughput ablations, not identical-
  recipe speedups; retain the original-ratio control and test learning quality.
  The completed Boxing 40k–50k windows attribute 74.4% of R256 time to learning
  and 56.1% of R64 time to observation. Use actual emulator-frame increments
  for game clocks, and reprofile the dominant stage after recipe selection.
  The current-backend fresh Boxing seed-1009 readout covers 195,996 post-warmup
  actions: 73.2% learning, 26.2% observation and 0.5645× aggregate real time.
  Updates stay around 345–346 ms; actual-frame clocks leave 140 ms/update for
  aggregate 1×, below world training alone at 161 ms. World sync is only 3.48%
  of wall time. Preserve `runs/boxing-runtime-20260910.CSrdK6`; this retrospective
  CPU readout is not a speedup or seed-reliability result. Keep active queues fixed.
  Host readback waits include unfinished producer computation and transfers;
  do not relabel them GPU idle time. Substage timings are contained in their
  parent stage totals, not additional elapsed time.
  GPU traces synthesized from host submission times are not calibrated
  GPU idle-gap measurements; distinguish pass durations from timeline placement.
  External captures require usable imported output and expected GPU workload
  coverage across the run; a successful CLI exit or one GPU row is insufficient.
  Preserve raw results when a later coverage audit rejects a preliminary gate.
  Batch row-independent replay encoding and heads across time without batching
  away recurrence or introducing future inputs. Check production-sized losses,
  all parameter gradients and reset causality; composed losses need complete
  scalar reductions, not backend workgroup partials.
- Test dense Atari, sparse Atari, and a small native persistent environment.
  Positive terminal return is a Pong win rule, not a general Atari competence
  criterion. Keep game-specific wins separate from generic episode accounting.
  The active five-game objective targets Pong, Boxing, Freeway, Breakout and
  Qbert; see `docs/experiments/2026-09-08-atari-five.md`. The isolated
  `exp/atari-five` v2 runner completed the fixed 200k-action Boxing R64/R256
  pilot and 75k-action N8 frozen evaluations. R64 seed 0 passes its frozen gate:
  40/40 wins, mean +51.55, with complete checkpoint/declaration/replay checks.
  The original queue and follower stopped on a zero-update control restore
  failure. The save-only repair now passes CPU/GPU checks, exact 12M initial
  actions/parameters and complete trained-state preservation, retaining strict
  restore checks. Preserve the original artifacts. The completed continuation in
  `runs/atari-five-continue-20260908.JrdVto` completed the repaired zero-update
  control: 21/40 wins, mean +0.125, with full checkpoint/replay checks. R256
  finished 200k actions and 49,619 updates with the original native package.
  Its 75k frozen evaluation passes: 162/162 natural wins, mean +92.4877,
  no cutoffs or updates, with complete checkpoint/declaration/replay checks.
  Use R256 provisionally for its larger score margin, retaining R64 as the faster
  ablation; this costs roughly 2.29 times the training-loop wall time. Preserve
  the completed queue, repair evidence, packages and shared auditors. One pilot seed does not establish
  reliability: require a separately declared fresh three-seed replication
  using 1009, 2017 and 3019.
  The fresh Boxing confirmation started at 01:10 UTC on September 10 in
  `runs/boxing-confirmation-20260910.hTEDcu`, with 429 pins and 61 passing CPU
  checks. It reverified all raw current-backend runtime evidence before launching
  seed 1009. Use qualified native f6a2b6ad and matching source 90b4763, N6/R256;
  each root 1009/2017/3019 receives 200,004 fresh training actions, 75,000 sampled
  unassisted frozen actions and a separately restored same-seed untrained control.
  The full confirmation completed normally at 00:51:13 UTC on September 11.
  All three paired gates pass: trained natural wins 123/123, 207/207, 51/51;
  means +83.8699/+90.5845/+83.5294. Untrained means −0.7222/+0.8056/+1.25
  fail competence and score lower. Every training root completed 200,004 actions
  and 49,651 updates; all six 75k frozen evaluations have zero updates and no
  cutoffs. All 18 commands, 12 raw GPU windows, complete finite checkpoints,
  optimizer moments, actual encoder identities, scores and replay/video bindings
  independently reverify with all 429 experiment and 756 handoff pins. Directly
  free memory remains at least 3,302 MiB overall and 3,413 MiB in frozen evaluation.
  All three original zero-update saves and actual restore headers reverify:
  every pair differs in all 31 nonconstant parameter tensors; the 64 expected
  constant tensors match and all 146 optimizer moments per save are zero.
  Distinct initial values and disjoint RNG inputs are not a statistical proof
  of independence. Boxing meets the declared three-root gate, not a guarantee
  for arbitrary seeds or protocols. Preserve this completed queue and all pins;
  never restart it. This separate confirmation does not bypass the old
  replication-v2 runtime checker or satisfy all five games.
  See `docs/experiments/2026-09-10-boxing-confirmation.md`. Keep optional
  world-sync fan-out separate; its ~16 ms/update scope must not indefinitely
  displace actual learning, and its old-backend CPU checks are not adoption.
  All three first 20,004-action / 4,651-update Boxing saves are archived, with
  all 241 tensor entries complete/finite and prefix GPU coverage retaining at
  least 3,302 MiB directly free. Preserve these completed early-health snapshots;
  they never replace the declared final frozen models or all-seed gates.
  The Python-only episode-count carry at `exp/current-episode-evaluation`
  (`24b2968`) passes all 580 CPU tests on the unchanged f6a2b6ad native.
  `runs/current-episode-package-20260910.etyDN4` binds the source-matched bundle
  and evidence with 61 pins. Its complete current-package runtime gate in
  `runs/current-episode-runtime-20260910.uRF9VK`, with 498 pins and 47 passing
  CPU tests, finished normally at 01:36:19 UTC on September 11. Independent raw
  rechecking passes all 12 commands/eight GPU phases, exact default-learning
  state/reports/traces and retained anchor, full frozen state and exact prefixes.
  Episode stopping occurs at 10,716 actions; the six-action negative cap remains
  incomplete as required. Directly free memory stays at least 3,302 MiB. The
  single-pair throughput ratio 1.002376 clears the regression guard, not a speedup
  gate. This qualifies the bundle for the separately declared new protocols;
  it does not reinterpret historical fixed-action evaluations or establish
  learning. Preserve all completed inputs and the earlier live-parent refusal;
  do not restart this gate. See `docs/experiments/2026-09-09-episode-evaluation.md`.
  The first Breakout/Qbert seed-0 pilots were declared in
  `runs/breakout-qbert-pilots-v2-20260910.9zf9T3`, with 521 pins and 92 passing CPU
  tests. The existing serial follower launched the worker after the runtime
  gate's actual exit. It independently reverified the complete raw runtime proof
  before fresh Breakout training started at 01:36:44 UTC on September 11.
  Actual startup confirms source 24b2968/native f6a2b6ad, the encoder, fresh
  seed 0/zero counters and no restore or overrides. Qbert remains queued; no
  learned frozen result is available yet. Each game gets 200,004 fresh R256 actions,
  no exploration overrides, final-checkpoint v4 evaluation with four completed
  episodes per stream/cap 600,000 actions, and a separately restored untrained
  control. Keep all outcomes, task thresholds and fresh-seed requirements;
  these are pilots, not reliability. Preserve the declaration and longer frozen
  timeout. See `docs/experiments/2026-09-10-breakout-qbert-pilots.md`.
  Breakout finished 200,004 actions / 49,652 updates at 08:11:08 UTC on
  September 11. `runs/breakout-final-training-20260911.6Mnq8o` independently
  verifies the full reset-dependent ledger, all 241 finite tensor entries and
  optimizer moments, actual encoder, 756 pins and whole-training GPU coverage
  with at least 3,303 MiB directly free. These are completed training checks,
  not task wins. The next device guard failed with NVML exit 18 before frozen
  evaluation; the pilot and serial follower stopped, and no successor started.
  Preserve completed training and the failure; never restart the original queue.
  Breakout's first 20,004-action save is archived and checked in
  `runs/breakout-first-save-20260911.H7qmnT`: 4,652 actual updates, all 241 tensor
  entries complete/finite, 180 positive reward events and ≥3,303 MiB directly
  free through the save. The prefix remains explicitly incomplete; this is
  early health, not frozen competence. Preserve the completed inspection and
  its CPU import-name failure; do not rerun its exclusive archive operation.
  At 06:42 UTC on September 11, an unattended host update installed NVIDIA
  595.91.07 user-space while the loaded kernel remained 595.71.05. Fresh NVML
  queries failed with exit 18. The original Breakout trainer and declared
  logger finished on old mapped libraries and are now absent. Preserve the
  completed training and all pins; do not bypass device guards, restart the queue or change/reboot
  the host without user approval. Read `docs/experiments/2026-09-11-host-driver-incident.md`
  before any new runtime handoff. Its completed observer used the original
  logger and recorded the fresh query failure; it is not a live handle now.
  A changed driver requires runtime requalification and a separately declared
  continuation. The September 11 14:36 UTC recheck now observes matching
  loaded/NVML 595.91.07 after an external host reboot; no host change was made
  by this investigation. `runs/meganeura-runtime-20260911.Nnfxk4` completed fresh
  old-backend/new-driver then latest-backend qualification at 15:56:10 UTC,
  including exact archived old-driver state/pixel anchors and direct memory
  coverage. This is runtime qualification, not a resumed learning queue. A
  separate continuation must retain Breakout's original backend and explicitly
  bind both actual driver headers to the new proof; do not rewrite headers or
  rerun completed training. Preserve all original queues and Breakout's final checkpoint;
  see `docs/experiments/2026-09-11-meganeura-runtime.md`.
  The separate continuation in `runs/atari-driver-continuation-20260911.LR9yT3`
  binds 1,593 pins and 55 passing CPU checks. Actual launch freshly reverifies
  completed Breakout training, the original episode gate and raw recovered-driver
  qualification, then starts the driver episode fixture at 16:19:10 UTC on
  September 11. Require its exact old-driver 10,716-action trace and complete
  frozen state before the missing Breakout evaluation/control, then fresh Qbert.
  That fixture completes at 16:25:45: full state and trace match exactly, all
  1,578 GPU samples retain at least 3,413 MiB directly free, and independent raw
  state/trace/command/memory and 1,593-pin rereads pass. Do not restart this gate.
  Breakout frozen evaluation starts at 16:25:49. Actual startup confirms the
  original complete checkpoint hashes and 200,004 / 49,652 restored counters,
  native f6a2b6ad and the unchanged unassisted v4 rule on driver 595.91.07.
  Breakout's full pair completes at 16:40:33: trained mean 58.4583 across 24
  natural episodes versus untrained 0.9655 across 29, with zero updates/cutoffs.
  Neither has any two-wall completion: 0/24 and 0/29, so the pilot fails its
  unchanged competence gate despite learned improvement. Complete checkpoints,
  ledgers, scores, whole replay videos, six finished commands and four native
  GPU windows independently reverify in
  `runs/atari-recovered-confirmations-20260911.xPz5ud/breakout-result.json`,
  retaining at least 3,413 MiB directly free and 1,632 pins. This is a completed
  Breakout prefix, not whole-queue completion. Its separate completion checker
  passes 15 CPU tests; no new confirmation or follower is declared by that work.
  Do not replicate the failed Breakout recipe for a competence claim: declare a
  bounded repair comparison after the existing queue, then confirm a successful
  choice on fresh seeds. Preserve both trained and untrained outcomes and videos.
  Qbert fresh seed-0 training starts at 16:40:35, with actual zero counters,
  no restore, the original LeVJEPA encoder and unchanged 200,004-action N6/R256
  recipe without overrides. No Qbert frozen result is available yet.
  Its first 20,004-action / 4,651-update save completed at 17:18:58 UTC and is
  archived in `runs/qbert-first-save-20260911.XTUykj`. All 241 finite tensor
  entries, optimizer moments, actual encoder, full prefix ledger and 1,632 pins
  reverify. The prefix has 438 positive reward events, 55 natural episodes and
  14,075 aggregate reward, not a per-episode score or competence gate. All 9,199
  GPU samples through the save retain at least 3,303 MiB directly free. Preserve
  this completed archive and its wrong-counter negative; do not rerun the
  exclusive archive or substitute this early model for the declared final one.
  The completed Breakout diagnostic in `runs/breakout-diagnostic-20260911.BTTecu`
  rechecks all 200,004 actions / 49,652 updates and both complete frozen replays,
  preserving video RGB hashes, episode/life/bitmap accounting and 1,637 pins.
  There are 5,251 positive training reward events and no reported positive-free
  replay batches; late training returns average about 44–46. Even the best
  frozen episode leaves 70/108 first-wall bricks. This is not Freeway-style
  reward starvation, near competence or a causal diagnosis. Seven CPU fixtures
  pass; RAM remains strictly post-hoc. Prioritize a separately declared minimal-
  action comparison after the existing queue, retaining the eighteen-action
  control and unchanged gates, without also changing budget/reward/perception.
  The common-input CPU mapping in `runs/breakout-minimal-cpu-20260911.NcHj9r`
  verifies 12,288 paired decisions / 24,576 actual wrapper interactions across
  three seeds; reversed left/right is rejected at action 2. This proves the
  shared NOOP/FIRE/RIGHT/LEFT subset, not equivalence of all eighteen actions,
  native gradients/state/restore/memory, learning or a new GPU follower. Actor
  output and RSSM action-input widths change; require matching qualification
  and a fresh paired declaration before training. See
  `docs/experiments/2026-09-11-breakout-diagnostic.md`.
  The isolated `exp/breakout-minimal` candidate at `0591eda` carries only the
  four native dependency/identity changes from qualified `1e00e818` onto the
  current-episode source. All native/build inputs match that source; the fresh
  `runs/breakout-minimal-package-20260911.fwfepW/package` combines unchanged
  qualified abf4ae5d bytes with matching Python. All 620 CPU tests and 102 package
  evidence pins pass, with all 1,622 active scheduler pins unchanged. Minimal
  Breakout replay requires an explicit four-action declaration and v2; published
  full-action replay stays v1. Checkpoints require a matching action-count schema.
  Collection, recurrence, learning arithmetic, episode stopping and task gates
  are unchanged. Both future learning arms must use this same new-backend bundle;
  the old f6a2b6ad Breakout pilot is context, not its matched eighteen-action arm.
  These CPU checks are not four-action GPU/runtime/learning qualification and
  launch no follower. Existing production-gradient and synthetic-canary fixtures
  hardcode eighteen actions; qualify matching four-action fixtures, complete
  state from update 1, restores, traces and combined memory before learning.
  Do not require identical learning states across different action widths or
  infer a speedup. Preserve this package, every historical control and the active
  Qbert -> Freeway -> Pong order. See `docs/experiments/2026-09-11-breakout-minimal.md`.
  Its separate `exp/breakout-minimal-gates` fixture source at `18c7ffb` changes
  only cfg(test) code and the canary example, preserving the 0591eda/abf4ae5d
  package. Four-action B16/T64 world gradients, full H15 actor/value row-gradient
  comparisons for both vocabularies, and `dreamer_canary --actions 4` are built,
  not GPU-validated. The completion in `runs/breakout-minimal-fixtures-complete-20260911.dv2SqC`
  passes 98 Rust workspace tests, fmt and Clippy, preserving 134 completion,
  102 package and 1,622 scheduler pins. The enforced one-core/2 GiB host scope
  peaks near 990 MiB; no production ML graph or GPU fixture executes. Preserve
  the first root `runs/breakout-minimal-fixtures-20260911.EaDRLF`, which passed
  81 library tests then failed an incorrect workspace-count assertion. The
  completed continuation is not a numerical fix or GPU gate. Keep the compiled
  fixture identities, all original binaries and active queues fixed. Its new
  hardware/synthetic declaration is `runs/breakout-minimal-hardware-20260911.PLAL8H`,
  with 1,870 pins and 92 passing CPU checks, including the actual live-parent
  refusal before GPU work. The one-shot follower starts at 18:58:29 UTC on
  September 11 and waits for the actual Qbert -> Freeway -> Pong scheduler
  PID 42730/start ticks 1021056; its own PID is 52404/start ticks 1665628.
  No diagnostic GPU phase has run. Preserve this queue and never manually start
  its worker. It requires independent complete raw prerequisite evidence,
  including Pong's 24 commands/12 native phases, matching v4 ledgers, all states,
  frozen replays/videos, world-recording selection and GPU coverage; valid
  competence failures remain failures and do not block the diagnostic.
  The 16 native phases cover four full world/behavior gradient tests and six
  exact update-1/update-8 canary pairs with the qualified eighteen-action control
  and anchor. Compare complete logical weights/moments/reports within each width,
  never across widths. Keep every tolerance and ≥2,048 MiB directly free in every
  complete sampled window. No retries, automatic adoption or learning follower.
  Even a pass leaves four-action zero-update initialization/restore, N6 pixel
  traces/v4 replays and combined learner/perception memory unqualified. Require
  a new declaration for those gates and a matched learning trial; update 1 already
  changes optimizer moments. This is not a speedup or a Breakout reliability result.
  The remaining pixel preparations are CPU-only and preserved: the strict state
  inspector `runs/breakout-action-state-20260911.lOcNzn` (88 tests/53 pins), capture
  library `runs/breakout-pixel-capture-20260911.dJZZaz` (71/184), and read-only
  matrix `runs/breakout-pixel-runtime-20260911.k3pn4Z` (102/215). Actual source-matched
  imports and qualified eighteen-action states reverify; historical package and
  wrong-width negatives remain. No four-action native restore, GPU declaration
  or pixel follower exists. Bind the prepared matrix only in a new declaration
  after the hardware gate and complete raw prerequisite proof. Keep its 16 short
  native phases, ten complete frozen replays and six-action incomplete-cap prefixes
  distinct from learning budgets. Never weaken the complete-run reader to accept
  that cap, or mistake fake ledgers/state fingerprints for native/ALE evidence.
  Derive restore/update counts from complete producer ledgers, not fixture 610/611.
  Preserve the original capture freshness guard and require captured/uncaptured
  state/report/trace parity within each width. The old eighteen-action schema
  stays pinned: only dynin2 and actor output weights/bias plus six moments resize;
  all 241 entries and actual encoder identity remain mandatory. Compare common
  initial shapes across widths, never full trained state. Update 0 requires zero
  moments/normalizer; update 1 already has 122 nonzero moments. The 86,184-byte
  logical reduction is not measured VRAM. Preserve every input and all 1,870
  hardware pins; require actual command/replay/memory/timing proof before any
  qualification or learning trial. See the linked Breakout minimal report.
  The conditional executor at `runs/breakout-pixel-execution-20260911.n9akxr`
  passes 74 CPU tests with 1,990 pins and two actual live-hardware CLI refusals.
  It creates no runtime declaration or follower. Positive lifecycle tests are
  fabricated, not completed hardware or native capture evidence. Recheck the raw
  hardware/whole learning queue plus actual follower command/exit/output bindings
  before use. Keep historical prerequisite/measurement imports separate from new
  native/state/ledger/replay processes: their native packages and modules named
  checks differ. Actual import-only probes verify both paths, not GPU execution.
  The measured controller and ten replay bindings remain unconnected; all pixel
  qualification flags stay false. Preserve this preparation and all queue inputs.
  The recovered Breakout/Qbert worker retains source 24b2968/native f6a2b6ad,
  the original recipes and task criteria,
  four frozen episodes per stream/cap 600,000, actual reset-dependent updates,
  complete checkpoint/replay/video checks and measured direct-free headroom.
  These are still pilots, not fresh-seed reliability. This worker starts no
  Freeway/Pong successor itself. Their separate recovered-driver continuations
  now live under `runs/atari-recovered-confirmations-20260911.xPz5ud`: Freeway
  binds 1,608 pins and Pong 1,613, retaining all original recipes, roots, budgets,
  final frozen gates and untrained controls. The isolated runner passes 87 CPU
  checks, including exact old/new native commands except output paths, copied
  driver-template isolation, unchanged paired gates and scheduling/refusal errors.
  Both actual entrypoints refuse the live breadth controller before GPU queries
  or run outputs. Their declarations freshly reverify complete pilot/runtime data.
  A once-only serial follower starts at 17:11:03 UTC on September 11 with 1,622
  pins: PID 42730/start ticks 1021056, bound to controller 36135/705957. It waits
  for actual breadth exit and complete raw checks, then Freeway and Pong in that
  order. Neither fresh confirmation is training yet. Keep native f6a2b6ad and
  matching sources 90b4763/24b2968; main's latest backend does not switch them.
  Valid competence failures remain failures; incomplete data, changed host/input,
  runtime or integrity failure stops without retry. Preserve the actual driver
  headers, first-four-match Pong world-source selection and all old roots. No
  world GPU work or extra game/optimization follower is included. Never manually
  launch duplicates or restart this scheduler. See
  `docs/experiments/2026-09-11-recovered-confirmations.md` and keep the live scripts,
  pins and old queues unchanged; see also the original continuation report.
  Preserve but never launch the superseded unstarted v1 root
  `runs/breakout-qbert-pilots-20260910.h0l2PM`. It incorrectly hardcoded Boxing's
  update count. Reset observations enter replay without action credit and can
  advance warmup. Use the complete source-matched ledger auditor's exact count
  for each game's frozen/checkpoint counters. The real CPU Breakout reset
  histories and synthetic schedule tests are audit evidence, not native training.
  Adjacent roots reuse live policy/posterior RNG streams under `seed + stream`;
  keep the declared live-seed ranges disjoint, without reinterpreting old
  results or rewriting the completed pilot. Verify actual child
  processes before waiting, and do not treat completed tooling as five-game wins.
  Its CPU-only frozen-result follower pins the candidate match auditor,
  `replay_atari.py`, `atari_tasks.py` and their dependencies too; keep them
  unchanged while live. Task observers are post-hoc evaluation, never policy
  inputs or added training rewards. Scripted observer fixtures are not Kindle
  wins. Qbert's first pyramid alone is not the sustained-competence gate.
  Distinguish action-order sensitivity from visual-feedback or planning evidence;
  post-hoc action shuffles are not extra Kindle wins or held-out policy benchmarks.
  Breakout's 864-point rule has a verified actual-ROM scripted fixture and
  negative controls; preserve its distinction from Kindle's learned results.
  The separate candidate `audit_atari_tasks.py` covers Freeway/Breakout/Qbert
  final checkpoint and replay scoring. A task-gate pass alone does not verify
  campaign budgets or independent training seeds. The separate candidate
  `audit_atari_campaign.py` checks all 15 declared game/seed records, fixed
  budgets/config, fresh models, final checkpoints and replays. Its CPU checks
  are complete, not a replication result. Keep untrained controls and the
  broader goal-completion audit; do not infer them from `replication_passed`.
  The N6 sparse Freeway pilot in `runs/freeway-pilot-20260908.WWxHEM` completed
  at 06:06 UTC on September 9. The unchanged repaired package trained R256 for
  200,004 fresh seed-0 actions and 49,651 updates; all 96 natural rounds were
  unrewarded. Trained and separately restored zero-update policies each return
  zero in all 36 natural rounds of their 75k frozen evaluations, with no cutoffs
  or updates. Both complete checkpoints, full CPU replays and all four GPU-phase
  coverage/memory checks pass; directly free memory stays at least 3,303 MiB.
  Whole stream-0 videos are `evaluation.mp4` and `untrained-evaluation.mp4`, not
  successes. Preserve all 34 pins and artifacts; do not restart this completed
  pilot or call it reliable learning. Its follower has completed common-world and
  exploration validation; the separately declared learning pilot has now completed.
  The completed CPU-only Freeway discovery check in
  `runs/freeway-discovery-20260908.Zig71a` compares hold lengths 1/16/64 at
  200,004 actions each on three seeds. Independent random actions find no
  crossing rewards; hold16 finds 293/324/328 and hold64 finds 706/716/721.
  These are random exploration controls, not learned task wins or matched
  native-policy evaluations. All action-generation/episode accounting passes;
  independent ALE replay was not performed. If the native control remains
  reward-starved, prioritize a separately declared persistent-exploration
  ablation over simply extending it. The isolated `exp/persistent-exploration`
  candidate adds explicit native action overrides and versioned per-stream random
  holds, with 95 Rust and 547 Python CPU tests passing. Its full runtime gate
  and the matched seed-0 Freeway learning comparison now pass; the provisional
  Freeway choice below still needs independent fresh-seed confirmation.
  Preserve actual executed actions in RSSM/replay,
  independent RNG, default-path parity and strictly unassisted frozen evaluation.
  See `docs/experiments/2026-09-08-persistent-exploration.md` for the required gates;
  run only after Freeway and the declared common-world diagnostic release the GPU.
  Existing campaign declarations reject the changed exploration protocol. A new
  package requires its own matching runtime/memory evidence before long training.
  That gate completed in `runs/persistent-exploration-gate-20260908.OVSB5q`,
  with 66 pins and 31 passing CPU gate/handoff tests. Its 107-pin follower
  completed Freeway/common-world prerequisites and launched the gate at 06:47 UTC
  on September 9, finishing at 07:30 UTC. Three native tests, exact default
  state/report/trace parity, the pixel override integration and all 13 GPU
  memory/coverage phases pass. Minimum directly free memory is 3,303 MiB;
  warmed candidate/control throughput ratios are 0.999197 and 0.999774, not
  a speedup. Preserve inputs and results; do not restart this completed queue.
  This gate starts no long learning run and does not adopt exploration.
  A separate conditional learning declaration reverified the actual raw evidence:
  `runs/freeway-persistence-learning-20260909.C0GoqT`, 82 pins and 47 passing
  CPU launcher/proof tests. Hold64 started at 07:31 UTC on September 9, after
  raw runtime verification; the worker compares .5 exploration probability
  with hold64 versus hold1, each fresh seed 0, 200,004 training and 75,000
  unassisted frozen actions, followed
  by a separately restored untrained control. Both arms use the same new package;
  the old plain-policy pilot is context, not the matched hold1 arm. Keep its
  inputs fixed. Hold64 completed 200,004 actions / 49,651 updates at 13:59 UTC:
  96 rewarded natural rounds, mean 14.1146, still assisted. Complete final state,
  exploration accounting and frozen restore checks pass; full-training GPU
  coverage retains at least 3,302 MiB directly free. Its completed 75k unassisted
  frozen evaluation passes: 36/36 natural rounds reach 25 crossings, mean 31.0556,
  no cutoffs or updates, with complete checkpoint/replay/video checks. Frozen
  GPU coverage retains at least 3,413 MiB free. Hold1 also completed 200,004
  actions / 49,651 updates and its 75k unassisted frozen evaluation: 36/36
  qualifying natural rounds, mean 29.0278, no cutoffs or updates, with complete
  state/replay/video checks. Both arms pass on this seed; hold64's 2.0278-crossing
  mean advantage does not establish that persistence is necessary or reliable.
  The restored untrained control returns zero in all 36 natural rounds and tails.
  The full pilot is complete: all 11 command exits, complete state/replays and
  six GPU phases reverify, with at least 3,302 MiB directly free overall.
  Provisionally use hold64 for fresh Freeway confirmation for its larger score
  margin and rewarded training coverage; retain the successful hold1 control.
  This is a post-pilot choice, not reliability or automatic launcher adoption.
  Do not assume held exploration benefits other games. No fresh replication yet;
  preserve all 82 pins and do not restart the completed learning queue.
  See `docs/experiments/2026-09-09-freeway-persistence.md`.
  Fresh Freeway hold64 confirmation is now conditionally declared in
  `runs/freeway-confirmation-v2-20260910.4w3RV8`, with 626 pins and 102 passing CPU
  checks. The actual launch refuses the bound live Boxing predecessor before
  GPU work; no Freeway worker is active. Preserve Boxing -> current episode
  runtime gate -> Breakout/Qbert pilots -> Freeway confirmation order, with
  complete predecessor checks and valid competence failures kept distinct.
  Use the qualified f6a2b6ad/90b4763 package, N6/R256, roots 1009/2017/3019,
  200,004 training actions with probability .5/hold64 and 75,000 unassisted v2
  frozen actions per root, plus separately restored same-seed untrained controls.
  Require all three unchanged Freeway gates, complete state/replays and distinct
  initial/trained parameter fingerprints. This is declared, not a reliability
  result or five-game completion; see `docs/experiments/2026-09-10-freeway-confirmation.md`.
  The unstarted v1 root `runs/freeway-confirmation-20260910.megagv` is superseded
  before launch; preserve its 617 pins but never run it. V2 binds the corrected
  Breakout/Qbert declaration and its audited update counters; the Freeway recipe
  and its own fixed schedule are unchanged.
  The original plain-policy Freeway training log at 200,004 actions extends the preserved 72k
  diagnostic: all 49,651 updates have zero reported absolute advantage, despite
  declining prediction training loss and finite learner scalars/saved state.
  Imagined-policy entropy is near its maximum. Prioritize rewarded discovery,
  not an unsupported claim of numerical collapse; this is not frozen competence
  or a Pong diagnosis. Full-training GPU coverage passes with 3,303 MiB minimum
  directly free and 69.15% mean activity, not a matched speedup or whole-pilot gate.
  Details: `docs/experiments/2026-09-09-freeway-zero-signal.md`.
  The CPU episode-budget check freshly replays Breakout/Qbert random rewards
  and verifies the 25,000-decision wrapper cap. Their rewards are discoverable
  without Freeway's random holds. The separate `exp/episode-budget-evaluation`
  candidate at `4281242` adds frozen-only v4 stopping when every stream reaches
  a predeclared episode count, retaining every completed episode and a hard cap.
  Its 580 CPU tests and exact old-ledger accounting are not GPU validation or
  adoption. See `docs/experiments/2026-09-09-episode-evaluation.md`; preserve all
  current fixed-action queues, game criteria and fresh-seed gates. Require GPU
  prefix/state checks and a new declaration before using this stopping rule.
  Its frozen-only GPU check started at 22:30 UTC on September 9 in
  `runs/episode-evaluation-gate-20260909.8f1yKj`: 55 pins, 30 passing CPU tests,
  bound to the actual persistence-learning launcher, which has completed normally.
  It refuses live-parent execution and starts no follower. The first two phases
  pass state/ledger checks, but an interruption signal stops the third at
  9,150/18,000 actions at 22:50 UTC. Preserve that incomplete original queue;
  no numerical failure or complete gate result is claimed. The separately
  declared continuation `runs/episode-evaluation-continuation-20260909.6YKbjp`
  rechecks and reuses the two completed phases, with 100 pins and 51 CPU tests.
  It completed at 23:10 UTC: a fresh 18,000-action candidate restore matches
  the original control exactly, and the six-action negative cap case remains
  incomplete as required. Independent CPU rechecking confirms all 100 pins,
  complete frozen state, both shorter prefixes and all four GPU windows, with
  at least 3,413 MiB directly free. This validates the stopping implementation,
  not training or automatic protocol adoption. Preserve both completed phases
  and the interrupted original; do not restart either queue. The newly requested
  Meganeura refresh precedes the still-unrun world-sync comparison, keeping the
  backend and fan-out changes separate.
  Keep external reward and intrinsic reward separate. Retain an extrinsic-only
  control for every intrinsic-reward experiment.
  Record human guidance and the action actually executed; distinguish assisted
  behavior from unguided evaluation and game rewards from human feedback.
  Distinguish agent-collected online learning from forced-random coverage tests;
  verify learned behavior under frozen evaluation against untrained controls.
  Evaluate the declared final checkpoint and confirm independent training seeds;
  do not select a winning checkpoint or weaken acceptance after observing results.
  Record sampled/greedy action mode and recurrent-state initialization; isolate
  their effects when diagnosing a frozen-policy failure.
  Visual novelty is not task competence. Evaluate intrinsic exploration through
  held-out dynamics and later guided adaptation, with explicit reward provenance.
- Gate video encoders and pretraining on evidence: a usable pinned checkpoint,
  causal streaming semantics, native numerical parity, latency, and improved
  held-out control-relevant probes. A paper alone is not an implementation plan.
  Probe the trained recurrent belief before inferring a need for more temporal
  input from single-frame feature probes.
  The completed three-seed motion diagnostic finds useful motion information
  in every final belief, without matching the gameplay ranking. Prioritize
  sparse-positive discovery, reward/value calibration and action-use diagnosis
  over speculative visual expansion. Delayed first wins are not proof of
  numerical training collapse. Declare longer budgets for all seeds as a new
  experiment; never relabel the failed 200k-action mastery campaign.
- Evaluate the world model separately from its policy. The completed frozen
  first-match replays in `docs/experiments/2026-09-08-world-evaluation.md` match
  every recorded action and transition without learning. All three use action
  information in feature prediction. Own-policy reward errors motivated the
  completed common-recording comparison below; they do not establish causation
  or a global ranking of world-model quality. Prioritize reward generalization
  and experience coverage before speculative perception expansion.
  Forecast before consuming the target; separate prior from posterior reward
  estimates, include persistence/unrelated-action/zero-reward baselines, and
  report positive/negative/terminal counts. LeVJEPA cache resets can inflate
  persistence error, and a fixed stride can miss sparse classes entirely.
  Feature error is not imagined RGB, AUC is not magnitude calibration, and a
  strong model score is not policy competence. Preserve the original executable
  for historical model diagnostics; do not rewrite backend metadata to restore.
  The isolated `exp/common-world-probe` candidate at `a425b29` has 282 passing
  Python CPU tests. The declared follow-up in `runs/common-world-20260908.7gWHsJ`
  CPU-reconstructs all 11,388 first-match transitions and pins 35 inputs. It
  explicitly conditions every old model on the same three recordings, preserving
  strict unforced replay separately. Require three exact same-model H1 diagonals
  before the six cross-model runs, with common initial/target RGB and feature
  hashes. All nine GPU runs completed at 06:47 UTC on September 9: three exact
  diagonals, six common-input checks, zero updates and passing memory/coverage
  checks with at least 7,469 MiB directly free. Preserve the 35 pins, historical
  native f663dd93 and completed data; do not restart the diagnostic.
  Forced controls are offline diagnostics, not that model's policy rollout;
  another policy's logged return
  is not unbiased ground truth for the evaluated critic.
  The report in `runs/common-world-report-20260909.O7nqqe/report.html` is now
  generated after runtime validation, rechecking all nine results and video
  identities in a fresh historical-native CPU process. All models predict
  features and positive rewards best on their own recording; even the strongest
  player predicts other recordings' rewards poorly. Model 1 has the worst
  common-pool positive error but the best negative error; do not call its whole
  world model uniformly worst. Four of six cross-model all-frame prior reward
  errors exceed the zero baseline. Preserve per-recording and pooled event
  counts, visual-cache strata and the three-terminal limitation. This is limited cross-trajectory
  generalization, not proof of the policy failure's cause. Keep current learning
  arms fixed. Measure reward-event coverage and replay batches lacking each
  reward class; repeated samples are not distinct experience, and posterior
  training estimates are not held-out prior forecasts. Declare fixed all-seed
  budgets, own-policy frozen results and a new multi-match forecast set before
  selecting a changed recipe.
  The report builder's 13 fabricated-fixture tests are implementation evidence.
  The isolated multi-match world-probe candidate `b2f0ddd` has 633 passing Python
  CPU tests and 25,136 independently ALE-replayed actions; all three historical
  Pong prefix selections remain exact. Its 111-pin evidence is in
  `runs/multimatch-world-cpu-20260910.5SQRCl`. This validates extraction, not native
  forecasts or adoption. Select a predeclared first-N stream-zero subset only
  from complete v2/v4 frozen recordings; retain other-stream/tail audit checks,
  reset/target identities and exact strict-mode counters/actions. Conditioned
  vector forecasts may differ in warmup-dependent learner counts at the same
  action budget; keep both counts and all other identity checks. Require GPU
  serial/vector and same-model forecast parity before cross-model use. Preserve
  current queues; no longer Pong budget or follower is declared by this candidate.
  See `docs/experiments/2026-09-10-multimatch-world-probe.md`.
  A separate fresh Pong exposure confirmation is now declared in
  `runs/pong-confirmation-20260910.zFks3A`, with 735 pins and 95 passing CPU tests.
  Its actual CLI refuses the bound live Boxing controller before GPU queries or
  outputs. No Pong worker is active. Preserve Boxing -> current episode
  runtime -> corrected Breakout/Qbert -> corrected Freeway -> Pong order.
  Use unchanged 24b2968/f6a2b6ad, N6/R256 and fresh roots 1009/2017/3019, each
  400,008 training actions without overrides. Bind update counters to the complete
  reset-dependent ledger. Require all three unchanged Pong gates and separately
  restored untrained controls, using final v4 four-episode-per-stream evaluation
  with cap 600,000 and every outcome retained. This is a new larger-exposure
  confirmation, not an isolated budget ablation or repair of the failed 200k gate.
  Its first four complete stream-zero final matches per root are preselected
  without score filtering for the common H1 world set. CPU selection is not native
  forecasts; require the separate serial/vector/strict/forced GPU gate before use.
  No world GPU work or automatic follow-up starts here. Preserve all inputs and
  require complete predecessor evidence. See
  `docs/experiments/2026-09-10-pong-confirmation.md`.
  The separately declared serial follower in
  `runs/atari-serial-handoff-20260910.zF8Hfh` started at 04:40 UTC on September 10,
  with 756 pins and 52 passing CPU scheduling tests. PID 2318785/start ticks
  108736692 originally bound Boxing controller 2303115/107474767. Boxing has
  exited normally, and the unchanged episode runtime gate completed at 01:36:19
  UTC on September 11. The follower then launched the corrected B/Q controller,
  PID 2454804/start ticks 116273468. After complete Breakout training, the next
  device guard failed; this controller stopped at 08:11:10 UTC and the follower
  at 08:11:11 UTC. All original processes are absent. Qbert, corrected Freeway
  and Pong never started. Preserve this terminal handoff rather than waiting on
  its old PIDs or restarting it. Each future entrypoint must check its raw predecessors
  before GPU work. Valid task failures remain failures;
  incomplete data, integrity/runtime failure or a changed stage stops the handoff
  without retries. Preserve all inputs and do not manually launch duplicate
  successors, restart this follower or displace it with GPU-heavy diagnostics.
  The child declarations retain no automatic follow-up; this outer declaration
  supplies scheduling only. Its completion is not five-game success: B/Q still
  require fresh-seed confirmation, and native world forecasts are not scheduled.
  See `docs/experiments/2026-09-10-atari-serial-handoff.md`.
- Distinguish video-encoder initialization, action-conditioned world pretraining
  and policy-skill transfer. Missing action/reward labels are not NOOP/zero.
  The isolated `exp/world-pretraining` candidate includes world-only updates and
  strict fresh-runtime dynamics initialization, not supported dataset training or
  transfer. Require content-verified ingestion and GPU/adaptation gates before
  adoption. Initialized checkpoints require format-4 offline source lineage;
  do not silently reinterpret them as ordinary format-3 checkpoints.
  Hold target titles out of source data and tuning; measure adaptation and
  forgetting. Retain the source policy when testing full-policy transfer, while
  declaring head, optimizer, normalizer, replay and recurrent-state resets.
- Reuse mind-games' launch, time-control, capture and input infrastructure.
  Verify its Kindle revision/API before integration; legacy BatchAgent adapters
  are not the current Dreamer path. Keep privileged reward/task observers outside
  policy inputs, and do not inherit unreported shaping or scripted gameplay.
- Respect each stream of external consequences. Natural deaths and respawns
  are allowed; cloning or rewinding a live game for training is not. Independently
  initialized vector environments are allowed under a declared new protocol;
  do not relabel their experience as a continuation of a single-life experiment.
  Distinguish uncapped stepping, super-real-time playing plus training, and a
  free-running game without time control. Measure simulated/wall time with
  learning enabled; fast frozen inference is not training throughput. Preserve
  arrival order, actual action durations and observation gaps, and bound training
  debt. Try measured serial scheduling before any actor/learner separation.
- Delete superseded code and redundant documentation when they have no current
  purpose. Git retains history. Prefer small concrete modules over speculative
  frameworks, broad configuration surfaces, or premature swarm infrastructure.
- Work autonomously on authorized implementation, diagnostics, and experiments.
  Follow `/mnt/data/GUIDELINES.md` when available: attention is expensive, keep
  code self-describing, and only the user merges pull requests. Commit and push
  are permitted; do not send messages to other people without authorization.
- Preserve unrelated working-tree changes. Serialize GPU-heavy tests on a
  shared device and record the selected adapter. Run relevant formatting,
  Clippy, Rust/Python tests, and numerical checks for learning/backend changes.
