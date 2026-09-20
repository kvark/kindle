# Kindle working direction

Kindle is a Rust agent that learns while acting. Games are the first testbed;
explicit rewards and human guidance are allowed. Favor minimalism, expressiveness,
safety and speed. Keep learning/inference on Meganeura + Blade; Python is for
adapters, controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Product and research

- Keep one authoritative plan: `docs/kindle_single_life_dreamer_plan.md`, with
  one current game-status table and direct rollout/world-report links.
- Maintain `STATUS.md` as the short user-facing dashboard: timestamp, completed
  work, active phase, next actions, results and known limitations. Update it at
  meaningful phase boundaries, not every poll. Keep detailed gates in the plan
  and raw chronology in runs; link the dashboard prominently from README.
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
  Native pretraining is staged separately at `exp/levjepa-tiny-pretrain-20260920`
  (`b30b9c69`; package source `cf95966b`), on Meganeura `cc5fea74` / Blade
  `eaff5092`; the new upstream
  `dbb43648` scheduled-RMSNorm fix is carried intact before GPU qualification.
  Native AdamW, GPU EMA, complete-state save/restore and encoder export exist;
  100 CPU tests and both independent reference-fixture checks pass. The native
  Python interface, deterministic video sampler/restore contract and explicit
  standalone Tiny probing are implemented; this is not agent adoption. Fresh
  corpus `runs/levjepa-tiny-atari-corpus-20260920.oTjdon` completes with 250,000
  observations / 999,112 emulator frames and whole-recording held-out splits.
  B128/V4 batches measure 0.82–0.93s on one CPU; include all offline experience in
  comparisons. Private package build `runs/levjepa-tiny-package-20260920.rDhKjm`
  is built (native `60f7060b`); all 751 Python CPU tests pass against its actual
  import, and wheel/install/source module identities match. The dense causal
  reference completes in `runs/levjepa-tiny-streaming-reference-20260920.4xgb5h`;
  its weights are untrained numerical fixtures, not learned Tiny weights.
  The seven checks in `runs/levjepa-tiny-accuracy-20260921.qa3GqK` now pass:
  all 155 gradients, AdamW/EMA/export, exact 763-tensor continuation, dense
  causality/reset and N6/serial parity. Preserve the earlier AMD-selection and
  strict-trig failures; the latter's new primitive accuracy bound is explicit,
  not a pass of its old gate. No full-model or game gate changed. Use the
  NVIDIA-only Vulkan loader selection for environment-independent test contexts.
  The actual package's 32-update B128/V4 pilot completes in
  `runs/levjepa-tiny-fit-20260921.oDmZVn`: .45s native plus .80s data per step,
  all complete state/export checks pass. The held-out noncollapse screen in
  `runs/levjepa-tiny-feature-check-20260921.wSQUJj` also passes; not model quality.
  Its observation-only package cfa1749a / source 1d9f0e96 exposes Vulkan budget.
  Fresh seed 743's 4,096-update pretraining is running in
  `runs/levjepa-tiny-pretrain-20260921.JaPZpW`, using unchanged 60f7060b, checkpoints
  every 512 updates. Review completion before frozen feature/position/motion checks
  against its own untrained export. No gameplay adoption or automatic successor.
  Pretraining-only
  backend extensions do not automatically update the gameplay backend; use
  the same qualified runtime for both sides of a later size comparison. The
  B128/V4 static allocation estimate falls 25.3→9.4 GiB with native erf GELU;
  this includes Adam, not driver/staging/EMA costs or measured runtime usage.
  Continue from `runs/levjepa-tiny-cpu-20260920.e4QkQH/README.md`; do not rebuild
  that graph from scratch or relabel staging as an adopted runtime.
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
  Profile dominant stages before optimizing. Readback waits are not GPU idle.
  Preserve gradients, complete state/moments and action traces; use untraced
  matched-order timing. A lower replay ratio is a learning ablation, not parity.

## Current work

- Qualified block runtime: native `886bae68`, Meganeura `589d73ab`, Blade
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
- Next qualify/pretrain the compact causal encoder, then return to separately
  declared game comparisons. Do not replicate unchanged failed
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
