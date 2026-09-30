# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

- The September 27 user direction adopts `docs/strategy_reset_plan.md`:
  iteration speed -> test LeVJEPA's value -> exploration/reward -> video priors
  for dynamics/behavior -> later asynchronous real-time deployment. Stop
  unchanged five-game/mastery confirmations and Atari-only gate tuning.
  The current roadmap/evidence stays in `docs/kindle_single_life_dreamer_plan.md`.
  GPU acting/capture is integrated. Phase 1 is complete: fused learner/grouped
  RSSM pass, and the three-seed MinAtar screen completes in 8m18s. The 3x update
  target is unmet; the user explicitly accepts it as a stretch target, not a
  Phase 1 exit gate. Phase 2's offline probes and five integration smokes pass;
  Large decodes best, Tiny pretraining is mixed. The old native learned-RGB arm
  used a patch CNN/dense decoder; it was not an exact upstream visual model.
- **September 30 user-approved reset: replication first, screen small.** The
  final active upstream Seaquest seed2017 finished at 16:43 UTC: 200,004 actions,
  49,651 updates, online last-50 score 454.0, finite checkpoint and passing guard.
  All 24 completed runs pass their audits; all 21 unstarted entries are cancelled.
  The original queue and drain services are inactive, with workers reaped. Never
  restart that queue. Evidence:
  `runs/representation-learning-20260928.kjidlR/queue-cancellation.json`.
  The faithful small Dreamer RGB replacement now passes four-update upstream
  values/losses/raw-gradients and common-gradient optimizer/EMA checks: 1,524
  comparisons. GPU RGB64 filtering matches Pillow; JEPA pixels are unchanged.
  Raw-gradient diagnostic capture preserves all 1,280 saved production tensors.
  See `docs/results/2026-09-30-small-dreamer-replication.md` for methodology and
  retained near-zero-gradient sensitivity. These are synthetic tests, not
  learning. New GPU work remains serial and guarded. The Size1M batched
  learning pilot below now meets the sub-hour comparison target and shows
  useful upstream learning at the selected capacity/budget. Published
  Dreamer Atari scores are reference evidence, not something to rediscover.
  Local controls need only establish the smaller/custom protocol and our port.
  The user caps the new replication round at **10 training runs total**, across
  upstream and native, including failed/interrupted attempts. Target three
  paired learner seeds (six runs); at most four additional pilot/debug runs.
  Reuse qualifying pilot results when the recipe is unchanged. Do not silently
  extend the cap or follow it with another large JEPA matrix.
  The first small upstream Seaquest pilot completes 200,000 actions / 49,939
  updates in 516.71s plus 39.70s construction; first20/last50 online scores
  57.0/292.4. This is provisional one-seed learning evidence. Native attempt2
  was stopped cleanly at90,264 actions /22,505 updates:51.31 actions/s projects
  beyond the declared one-hour deadline. Guard, complete-prefix counters and
  finite checkpoint pass; this is not a native/kernel fault or a completed
  matched run. World updates dominate (~55/74ms). The bottleneck is now
  qualified: bounded split convolution gradients reduce synthetic full updates
  72.62→30.58ms (2.37x) at unchanged learning settings. Independent F64 checks,
  68 raw-gradient and 292 saved-state comparisons pass. Unsplit long reductions
  match sequential F32 exactly but fail the stricter F64 gate; the chosen splits
  pass without relaxing it. Fresh native attempt3 completes 200,000 actions /
  49,939 updates in 1,569.60s: 127.42 actions/s, 1.06x real time per stream.
  First20/last50 online means 83.0/328.8; 236 episodes, finite state, zero debt,
  guard/seal/counter audits pass. The successful pair takes 35m31s including
  construction (the interrupted 29 minutes remain extra). This qualifies the
  cheap recipe, not seed-level superiority or frozen competence. Three attempts
  used. Native seeds2017/3019 are needed under either proposed budget allocation;
  further upstream/JEPA allocation awaits the user's pending choice. See
  `docs/results/2026-09-30-small-rgb-profile.md`,
  `docs/results/2026-09-30-small-replication-learning.md` and PR31 for live state.
  Fresh split-kernel package Python tests pass (945); the expanded small JEPA
  curve auditor passes 29 focused tests. CI234/235 pass on all platforms.
  Return to >=3-seed JEPA comparisons on the qualified inexpensive recipe;
  promote only promising results to larger confirmation. Preserve native JEPA
  input detail; an explicit RGB64 Dreamer control is not RGB64-upscaled JEPA.
  Keep Phase 2 open until learning evidence supports an explicit frontend
  decision. Do not turn the cancelled matrix into a completed benchmark or
  attribute its architectural differences solely to backend correctness.
  Track evidence in `docs/results/2026-09-27-representation-learning.md`, the
  revised `docs/strategy_reset_plan.md`, and PR31. Do not mix speed and learning
  changes. Historical protocols and the September 28 interrupted attempt remain
  preserved, including their extra compute and pretraining disclosures.
- The first shared-parameter step passes exact 241-tensor/146-moment and report
  parity over 36 synthetic updates, plus 1,536-action/34-update N6 Pong per arm.
  Mean 12M update falls 227.03 -> 212.78 ms (6.28% less time), not yet game
  throughput or the 3x target. See `docs/results/2026-09-27-shared-parameters.md`.
  Fused T64/H15 recurrence now uses GPU Gumbel sampling with CPU-owned draws;
  outputs/gradients have independent references. Grouped RSSM layout improves
  the shared-control synthetic update 212.70 -> 184.43 ms and short native N6
  Pong R256 throughput 17.36 -> 20.16 actions/s. The 3x target is not met.
  See `docs/results/2026-09-27-fused-learner.md`. CPU targets/slow-critic EMA
  remain; shared mutable weights require serialized actor/learner access.
- `docs/screening.md` defines the separate Size1M/B8/T16/H15/R32 MinAtar recipe,
  eight streams and seeds 1009/2017/3019. Public 10x10 observations are packed
  losslessly into 7x7x64 and use a jointly learned encoder; no frozen frontend,
  pretraining or reward/action aid. CPU environment/upload is the allowed
  temporary fallback, not GPU-resident simulation or a CPU learner. Never
  compare its R32 speed with the R256 Atari control as an optimization gain.
  All three 32,768-action / 8,135-update runs pass. Final online scores are
  .44/.40/.68; mean .507 with seed-bootstrap 95% CI [.400,.680]. This does not
  establish reliable improvement or frozen competence. See
  `docs/results/2026-09-27-minatar-screen.md`. Keep the unresolved 3x target
  distinct from completed implementation; no unchanged mastery queue resumes.
  Next: the faithful small RGB control and matched learning screen above; a
  frozen JEPA frontend must demonstrate its value. No asynchronous learner or swarms.
- The user's September 26 direction supersedes historical checkpoint/pinning
  requirements: finish the new GPU encoding/acting path and remove obsolete
  implementations freely. Do not build migration layers for old checkpoints or
  recursively pin old experiment histories. Historical measurements remain
  historical; do not relabel them as results of changed code.
- `TASK.md` is the user's high-level intent. The acting path is GPU capture ->
  preprocessing -> causal LeVJEPA -> belief/policy -> action readback. Rewards,
  checkpoints and requested diagnostics may cross the host boundary. A buffer
  entry point alone is not capture integration. Validate ownership, producer
  completion and GPU memory visibility, including ring-buffer reuse.
- The current reference uses the independently pretrained ~5.5M causal Tiny frontend,
  not DINO or the 303M Large default. Keep Dreamer12M and the 7x7x64 observation
  contract for speed comparisons. Small learners/learned encoders are allowed in
  explicitly separate screening/representation experiments. JEPA must earn its
  place on probes and learning curves; it is not an obligation for 2D games.
  Preserve native image detail; no downscale-then-upscale adapter pipeline.
- Prioritize one reliable actor before swarms: Atari, accelerated playing plus
  learning, video/world pretraining, mind-games vkQuake2/TMNF, GOG/Wine, then
  held-out cross-game adaptation and retention. No concurrent learner service
  now; the strategy's later deployment phase may introduce one after measuring
  effective single-actor learning, actor latency and learner debt.
- Six environments share batched perception/policy and one learner, not causal
  histories. Preserve per-stream recurrent state, RNG, replay and resets. The
  encoder's 16-arrival chunk reset is not an environment/RSSM reset.

## Research and status

- One authoritative plan: `docs/kindle_single_life_dreamer_plan.md`. Keep it
  decision-focused with one game-status table and direct rollout/world-report
  links. Detailed results belong in experiment reports and `runs/`.
- Use the active phase's PR description as the status dashboard:
  dated done/running/next items, results and limitations. No STATUS.md. Update
  at meaningful boundaries, not each poll. PR29 is merged and remains the
  historical Phase 0/1 dashboard; Phase 2 uses
  [PR31](https://github.com/kvark/kindle/pull/31), `phase2-levjepa-evaluation`.
- The JEPA bet is cheaper useful latent world prediction than pixel
  reconstruction. Measure whole-agent and world-model time/memory/learning
  against a faithful Dreamer control at matched actual interactions, screening
  small before larger confirmation. Backend parity alone can
  share bugs: retain independent value/gradient references.
- Historical non-sticky fixed-protocol reliability is 3/5 (Boxing, Pong,
  assisted-training Freeway). Qbert and Breakout fail their gates. Pong's one-root
  sticky .25 evaluation fails: 2/24 wins, mean -7.1667. Do not call the historical
  non-sticky wins robust mastery. Tiny same-title video pretraining is additional
  experience (45k train + 5k validation frames/game), not online-only learning.
- Current trained Tiny encoder: `7fe9b252`, full path linked in the plan. Do not
  truncate Large weights or silently substitute an untrained product encoder.
- Current backend: Meganeura `75d08173` adds opt-in, bounded split convolution
  gradients and fixes split-measurement pipeline selection over `22c31b94`
  (tested batched last-two-axis transpose). Kindle uses512-position partitions
  on low-parallelism training convolutions; no new kernel or learning setting.
  The old matrix used `367e53d4` (Blade dependency update over
  `ee3aea42`); Blade `7cca6377` adds checked external Vulkan
  imports/ownership over `fbb4f28c`. Both branches are pushed. September 28's
  upstream check finds `7c29497` adding caller-owned submission APIs, not a new
  fix for the existing step path. September 30 rechecks that same Meganeura
  head; Blade `1da9ccb` changes Rapier/physics, not GPU execution. No new
  training correctness fix is identified. Check upstream
  before diagnosing already-fixed issues. Keep dependencies reproducible, but
  do not delay implementation to preserve obsolete runtime/checkpoint identities.
- GPU pixel v2 and its N6 train/frozen/sticky plumbing tests pass. The stock
  upstream Dreamer/JAX sanity also passes (5,990 actions, 1,149 updates); this is
  not a matched learning comparison. Phase 2 now independently aligns
  reset/update accounting, replay capacity and artificial-cutoff semantics;
  matched learning results are still required. See
  `docs/experiments/2026-09-26-gpu-pixels-and-pong-robustness.md` and
  `docs/experiments/2026-09-26-upstream-control-protocol.md`.
- Single and vector pixel actors now share resident encoder/pooling/belief/
  categorical sampling and paged GPU replay collection. Only selected actions
  leave the acting path. Explicit probes/checkpoints and sampled learner batches
  may read back. Imagined recurrence is resident; scalar target construction
  still uses the host.
  Dullahan GPU_SYNC supplies a fenced EXTERNAL ownership lease, not a SHM flag.
  vkQuake capture-to-keyboard plumbing passes at native 640x480: 12M frozen
  256 actions/2.758s; a separate tiny learner completes 105 finite updates.
  These are short plumbing tests, not 12M training speed or competence. Keep its
  sparse reward/terminal adapter and full mind-games GameSession integration distinct
  from transport success. See `docs/experiments/2026-09-26-gpu-resident-acting.md`.
- Screen small, confirm big. Use one changed factor, >=3 learner seeds and
  score-vs-actions/time curves for development learning comparisons, with
  bootstrap uncertainty or suite IQM; no development mastery gates. Numerical
  smoke tests are not multi-seed learning experiments. One matched timing plus
  numerical/learning parity check suffices for a speed change; no micro-campaign.
  Commit each new result as compact JSON + Markdown in `docs/results/`, including
  config, seeds, aids, curves and limits. The active PR remains the dashboard.
  Keep original competence gates for final confirmed claims. Retain failures
  and all completed episodes/unfinished tails.
  Plumbing tests and online wins are not frozen competence. Report aggregate
  and per-stream real time. A lower replay ratio is a learning tradeoff.
- Frozen evaluation never updates weights. Restore without replay/RNG/live
  belief is not equivalent to uninterrupted training. Separately evaluate prior
  forecasts with persistence/unrelated-action/reward controls and event counts;
  posterior estimates are not forecasts, features are not imagined RGB, and
  privileged game observers never enter the policy.

## GPU operation

- Ordinary bounded GPU work is authorized on driver 580.178.04. Normal JAX/CUDA
  initialization (including its internal NVML use) is permitted. Separate NVML
  polling/legacy health loggers remain off. Do not build an NVML-free backend or
  a CPU learner workaround. Unavailable utilization is unmeasured, not zero.
- Serialize heavy GPU jobs. Use `python/examples/gpu_host_guard.py` around the
  native-bearing process with a persistent systemd user service (`Restart=no`,
  `KillMode=control-group`, bounded deadline). A game integration needs explicit
  ownership/cleanup of its game process too. Review failures before any follow-up.
- Require the expected native device and >=2GiB sampled Vulkan estimated
  budget headroom. Budget-minus-usage is not physical free or peak VRAM.
  Last observed boot: `3e89d55c-a9e5-472f-a18a-06508c5bafa7`.
- Stop on kernel faults/native failures. No reset, driver reload/change, reboot
  or power-cycle without new user approval. Historical Xid62/154 incidents remain
  unexplained; successful no-NVML runs prove neither causality nor safety. Never
  retry quarantined 75dfe, 0a98775/02b600a1 or 070f4b51/7db0d05c bundles. Follow
  `docs/gpu_incident_response.md`; do not restart historical queues/followers.

## Implementation discipline

Keep exercised production code small. Delete obsolete encoding/compatibility
code instead of extending it. Use proportionate correctness/integration tests,
formatting and Clippy. Do not rebuild unchanged source for documentation edits.
One concise run configuration/result is enough; no new artifact-pin framework.

Automate routine validation. Check long training about every 30 minutes or on
completion; report meaningful results, not counters. Never compile during
matched timing. Use one CPU, 2GiB and zero swap for heavy preparation. Meganeura
has an unignored GPU test, so review CPU test filters.

Preserve unrelated user changes, especially in Meganeura/Blade/mind-games.
Only the user merges PRs; commits and pushes are allowed. Keep history linear.
