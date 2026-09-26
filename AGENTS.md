# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Current priority

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
- Use the independently pretrained ~5.5M-parameter causal-video Tiny frontend,
  not DINO or the 303M Large default. Keep Dreamer12M and the 7x7x64 observation
  contract until separately measured representation experiments justify change.
  Preserve native image detail; no downscale-then-upscale adapter pipeline.
- Prioritize one reliable actor before swarms: Atari, accelerated playing plus
  learning, video/world pretraining, mind-games vkQuake2/TMNF, GOG/Wine, then
  held-out cross-game adaptation and retention. No concurrent learner service.
- Six environments share batched perception/policy and one learner, not causal
  histories. Preserve per-stream recurrent state, RNG, replay and resets. The
  encoder's 16-arrival chunk reset is not an environment/RSSM reset.

## Research and status

- One authoritative plan: `docs/kindle_single_life_dreamer_plan.md`. Keep it
  decision-focused with one game-status table and direct rollout/world-report
  links. Detailed results belong in experiment reports and `runs/`.
- Use [PR29](https://github.com/kvark/kindle/pull/29) as the status dashboard:
  dated done/running/next items, results and limitations. No STATUS.md. Update
  at meaningful boundaries, not each poll.
- The JEPA bet is cheaper useful latent world prediction than pixel
  reconstruction. Measure whole-agent and world-model time/memory/learning
  against Dreamer12M at matched actual interactions. Backend parity alone can
  share bugs: retain independent value/gradient references.
- Historical non-sticky fixed-protocol reliability is 3/5 (Boxing, Pong,
  assisted-training Freeway). Qbert and Breakout fail their gates. Pong's one-root
  sticky .25 evaluation fails: 2/24 wins, mean -7.1667. Do not call the historical
  non-sticky wins robust mastery. Tiny same-title video pretraining is additional
  experience (45k train + 5k validation frames/game), not online-only learning.
- Current trained Tiny encoder: `7fe9b252`, full path linked in the plan. Do not
  truncate Large weights or silently substitute an untrained product encoder.
- Current backend: Meganeura `367e53d4` carries only the Blade dependency update
  over latest upstream `ee3aea42`; Blade `7cca6377` adds checked external Vulkan
  imports/ownership over `fbb4f28c`. Both branches are pushed. Check upstream
  before diagnosing already-fixed issues. Keep dependencies reproducible, but
  do not delay implementation to preserve obsolete runtime/checkpoint identities.
- GPU pixel v2 and its N6 train/frozen/sticky plumbing tests pass. The stock
  upstream Dreamer/JAX sanity also passes (5,990 actions, 1,149 updates); this is
  not a matched learning comparison. Reset/update accounting, replay capacity
  and artificial-cutoff semantics still need alignment for that claim. See
  `docs/experiments/2026-09-26-gpu-pixels-and-pong-robustness.md` and
  `docs/experiments/2026-09-26-upstream-control-protocol.md`.
- Single and vector pixel actors now share resident encoder/pooling/belief/
  categorical sampling and paged GPU replay collection. Only selected actions
  leave the acting path. Explicit probes/checkpoints and sampled learner batches
  may read back; host imagination/target construction is not yet eliminated.
  Dullahan GPU_SYNC supplies a fenced EXTERNAL ownership lease, not a SHM flag.
  vkQuake capture-to-keyboard plumbing passes at native 640x480: 12M frozen
  256 actions/2.758s; a separate tiny learner completes 105 finite updates.
  These are short plumbing tests, not 12M training speed or competence. Keep its
  sparse reward/terminal adapter and full mind-games GameSession integration distinct
  from transport success. See `docs/experiments/2026-09-26-gpu-resident-acting.md`.
- Declare budgets, seeds, controls and competence gates before learning
  comparisons. Retain failures and all completed episodes/unfinished tails.
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
  Last observed boot: `4f5152d1-e5fd-46cf-a0c4-06534c430d26`.
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
