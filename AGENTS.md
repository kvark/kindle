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
- **October 1: Phase 2 complete; learned RGB is the 2D screening default.**
  Three upstream/native RGB pairs finish in **7/10 replication attempts**,
  including the retained interruption. The ten-attempt cap applies to replication,
  not an implicit cap on all Phase 2. The separately declared **six JEPA runs**
  also complete, reusing all three native RGB controls; no more allocation is needed.
  Seaquest Size1M/N8/B8/T16/H15/R32, seeds1009/2017/3019, 200,000 actions /
  49,939 updates each. Final online means: RGB368.0 [328.8,440.4],
  pretrained Tiny225.333 [206.8,258.0], initial Tiny230.667 [218.8,239.2].
  Paired pretrained-minus-RGB is−142.667 [−229.2,−76.8];
  pretrained-minus-initial is−5.333 [−32.4,24.0]. No frozen competence claim.
  Curves cross mid-training; RGB does not uniformly dominate. Tiny halves world
  training (12.77→6.21ms) but takes18% longer end to end (26m31s→31m17s).
  Its first replay wait includes earlier shared-queue perception work, not just
  copying; readback is already batched. GPU utilization remains unmeasured.
  The native RGB port passes1,524 upstream numerical comparisons and all three
  learning pairs; upstream mean307.733 and about3.08x faster remain disclosed.
  The chosen split convolution gradients pass independent F64/raw-gradient/state
  checks and improve synthetic full updates72.62→30.58ms without learning changes.
  All learning guards, counter/trajectory and finite-checkpoint audits pass.
  `atari_vector.py ENV --output PATH` now selects learned RGB; frozen causal
  Tiny requires `--encoder-checkpoint PATH`. No old positional/sentinel shim.
  Fresh/default and frozen restore routes pass976 Python tests. The native
  learner is unchanged; no redundant training or Rust rebuild for this CLI change.
  See `docs/results/2026-10-01-frontend-decision.md` for the decision, costs,
  implementation, validation and limitations. CI250 passes on all platforms
  for the implementation at `f84d9b6`; PR31 tracks final status.
  Causal Tiny remains an explicit video/3D hypothesis, not an obligation for 2D.
  Native-detail JEPA preprocessing is unchanged; RGB64 is a separate learned
  Dreamer path, never an RGB64-upscaled JEPA input.
- **Next: Phase 3, exploration/reward.** Declare one small GPU-compatible
  mechanism versus extrinsic-only, three seeds, without Freeway's action aid.
  No old CPU feature-readback visitation workaround, new representation matrix,
  unchanged mastery queue, asynchronous learner or swarms. Phase 2 completion
  does not start a new training campaign automatically.
- **Cancelled historical 12M matrix:** retain all24 completed/audited runs and
  the21 cancelled unstarted entries. The final upstream Seaquest seed2017
  finished September30 at16:43 UTC; old queue/drain services and workers are
  inactive/reaped. Never restart them. Evidence:
  `runs/representation-learning-20260928.kjidlR/queue-cancellation.json`.
  The old learned-RGB arm was a patch CNN/dense decoder, not exact upstream;
  it cannot isolate backend correctness. Preserve failures and extra pretraining/
  interrupted compute disclosures. Numerical smokes are not learning evidence.
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
  The faithful small RGB and representation screens above now supersede this
  weak-learning development baseline. No asynchronous learner or swarms.
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
- The 2D Atari default is learned RGB on the qualified small recipe. The frozen
  video/3D reference uses independently pretrained ~5.5M causal Tiny, not DINO
  or 303M Large. Keep Dreamer12M and 7x7x64 for unchanged historical speed
  comparisons; do not label the small recipe an unchanged-learning speedup.
  JEPA must earn its place on probes and learning curves.
  Preserve native image detail; no downscale-then-upscale adapter pipeline.
- Prioritize one reliable actor before swarms: Atari, accelerated playing plus
  learning, video/world pretraining, mind-games vkQuake2/TMNF, GOG/Wine, then
  held-out cross-game adaptation and retention. No concurrent learner service
  now; the strategy's later deployment phase may introduce one after measuring
  effective single-actor learning, actor latency and learner debt.
- Environments share batched perception/policy and one learner, not causal
  histories (N8 small screen; N6 historical reference). Preserve per-stream
  recurrent state, RNG, replay and resets. The
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
