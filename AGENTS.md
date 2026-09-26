# Kindle working direction

Kindle is a Rust agent that learns while acting. Games are the first testbed;
explicit rewards and human guidance are allowed. Favor minimalism, expressiveness,
safety and speed. Keep learning/inference on Meganeura + Blade; Python is for
adapters, controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Product and research

- Keep one authoritative plan: `docs/kindle_single_life_dreamer_plan.md`, with
  one current game-status table and direct rollout/world-report links.
- Maintain [PR29](https://github.com/kvark/kindle/pull/29) as the status dashboard:
  timestamp, done/running/next work, results and limitations. No STATUS.md.
  Update at meaningful boundaries, not every poll. Keep checkpoint chronology,
  logs and repeated validation in `runs/`, linked through
  `docs/experiments/README.md`. Documentation changes no pinned gate.
- Prioritize a reliable single actor: Atari, accelerated playing plus learning,
  video/world pretraining, mind-games vkQuake2/TMNF, GOG/Wine, then held-out
  cross-game adaptation and retention. Strong GOG/transfer results precede swarms.
  Do not build a concurrent learner service.
- Gameplay uses native causal LeVJEPA, not DINO/RGB. The chosen small frontend is
  a separately pretrained ViT-Tiny/16: 12 layers, width 192, three heads,
  5,486,592 parameters. Keep Dreamer12M and the 7×7×64 observation contract.
  Do not slice Large weights or substitute an untrained encoder as the product.
  Tiny is frozen during gameplay and opt-in; Large remains default pending
  broader evidence. Pretraining and its mixed probes are linked from the plan.
- Perception's 16-arrival chunks reset visual history only; episode boundaries
  also reset belief. Prediction reads the deterministic prior before its target.
  Six streams share batched inference and one learner, not causal histories:
  preserve independent visual caches, recurrent state, RNG and replay streams.
- Count actual interactions and retain replay-ratio credit. Report aggregate and
  per-stream real time. Lower replay ratio is a learning tradeoff, not a parity
  speedup; more environments do not remove learner cost.
- Change one scientific variable per comparison. Declare budgets, seeds, controls
  and final-policy gates beforehand. Retain failures, every completed episode and
  unfinished tails. Online wins and adapter fixtures are not frozen competence.
- Verify complete saved state, optimizer moments and encoder identity. Frozen
  evaluation never updates weights. Restore omits replay, RNG and live belief:
  it is not an equivalent uninterrupted training resume.
- Evaluate prior world forecasts separately from posterior estimates and policy
  competence, with persistence/unrelated-action/zero-reward controls and event
  counts. Features are not imagined RGB; recorded future actions do not validate
  counterfactual policies. Missing video labels are not NOOP/zero. Privileged
  game observers are never policy inputs.

## Current decisions

- Five-game reliability is **3/5**: Boxing, Pong and Freeway pass their fixed
  three-root gates. Qbert and Breakout fail. See the plan's single table for
  results, controls, gates and videos. Confirm only passing recipes on fresh
  roots 1009/2017/3019.
- Keep trained Tiny encoder `7fe9b252` fixed for the next bounded comparison.
  The first Breakout pretraining ablation shows no benefit in one seed, not a
  diagnosed capacity limit. Existing world forecasts beat feature/reward
  controls but not continuation, with few terminal events.
- Qbert's completed R64 3.2M pair in `runs/qbert-r64-3m2-20260925.FrriIH` scores
  22/27 pyramids and mean 12,595.37: both final gates fail. Do not promote the
  midpoint. `runs/qbert-tail-analysis-20260926.m0P7ZI/results.md` locates five
  early failures and a later 8–9k plateau despite a 16,850 median. Exact CPU
  life-event replay in `runs/qbert-life-events-20260926.5rlCmF/results.md` finds
  repeated zero-progress deaths and two inspected edge falls. Most life losses
  are not terminals. The historical-checkpoint strict forecast probe is CPU-ready
  in `runs/qbert-hazard-probe-cpu-v2-20260926.GnWOvb`, not GPU-declared; preserve
  ACLJca's lazy-import preparation failure. Diagnose reward/value/policy alignment
  as well as continuation before another budget increase. Keep every episode/gate.
- Throughput qualification is complete. Current production is native
  `b00ce7be` / Meganeura `ee3aea42` / Blade `fbb4f28c`, the exact five-file
  production delta from `30bfa1a`, adopted without rebuilding its package.
  All compiled Rust/Cargo/Python inputs match that qualified source. The optimizer-only
  0db candidate is 0.8–0.9% slower; its nonregression pass is not a speedup.
  Backend correctness alone has not improved Qbert learning.
- Full-learner profiling in `runs/learner-timeline-20260926.6FJAqf` finds 62.4%
  GPU pass coverage, not SM utilization, and 26.09ms world command recording.
  Readback waits include computation. Four submissions cut synthetic core time
  7.4–7.5% with exact state/reports in `runs/world-submissions-20260926.9nJMTy`.
- Adopted source `30bfa1a` / native `b00ce7be` / Meganeura `ee3aea42` adds
  one scheduling call plus dependency/identity locks, no profiler or new API.
  Its 92 Rust/768 Python CPU tests, fmt/Clippy, package identity, N6 pixel/state/
  restore/replays and latest gradient/frontend/default compatibility pass:
  `runs/world-chunks-gameplay-20260926.q4iNnd/results.md` and
  `runs/chunks-compatibility-20260926.opVUiI/results.md`. Preserve Tiny dense's
  original memory-label reader failure and separate `tiny_dense-completion.json`;
  no GPU run was repeated.
- `runs/chunks-atari-timing-20260926.e2OEhn` compares the complete candidate
  package against production, not just chunk count. Order: control A /candidate A
  /candidate B /control B; each 10,008 actions/2,153 updates, measuring 4,008
  actions/1,002 updates after warmup. Require exact qualified prefixes and same-arm
  full state/reports/trajectories, >=5% less total time in both orders and <=10%
  repeat drift. All four windows and the complete comparison pass: 6.5–6.6% less
  total time, <0.1% total repeat drift, exact same-arm state/moments/reports/actions.
  Aggregate real time is 1.0405–1.0415×, about .174× per stream. Preserve all
  terminal writers; only unrecorded audit/compare modes are reusable.
  **No heavy CPU work during future matched timing.**
- The fresh Breakout comparison is declared in
  `runs/breakout-minimal-comparison-20260926.xsQCaK`: four versus eighteen actions,
  fixed Tiny7fe9/native b00ce7be, seed0, 200,004 actions /49,652 updates each.
  Latest four-action gradients and exact N6 repeat/restore/replay checks pass.
  The complete four-action pair fails: trained mean10.9167 versus .875 control,
  both0/24 two-wall completions; all full-state/initial/replay/video checks pass.
  Its report is `a4/results.md`. Eighteen-action training started September26
  at12:49 UTC; review its full result before frozen evaluation and control.
  Both arms remain required even though the first fails.
  Preserve zero-update initial snapshots, every episode, full state/replays and
  the original two-wall gate. Each GPU phase is individually reviewed; no queue.
- Next, complete that comparison and Qbert failure diagnostics.
  Preserve the old four-action hold. Do not repeat
  unchanged failed recipes merely to occupy the GPU. Imagination host work and
  GPU-resident parameter sync are separate optimizations; raw aliases must account
  for derived weights. No concurrent learner service.

## Evidence and immutable boundaries

- Inspect latest upstream fixes before backend diagnosis. Keep a qualified
  package fixed across an active campaign; adopt updates through a separate
  comparison. Never silently mix runtime packages across learner seeds.
- Include independent value/gradient references: same-backend parity can share
  bugs. Profile dominant stages before optimizing; use untraced matched-order
  timing with unchanged learning settings, complete state/moments and actions.
- Preserve completed/failed writers, source worktrees, private targets, packages
  and artifacts. Only documented audit modes are reusable. Correct reader
  failures separately; never overwrite failed evidence or repeat GPU work to
  repair a reader. Do not relabel old tests as qualification of changed sources.
- Original xPz5ud queue and reserved `pong/seed2017-train.stdout` stay terminal.
  Never remove that hold, restart retired followers or pool historical root 1009
  into the completed fresh Pong campaign `rBwdGF`.
- Freeway confirmation `tij9QW` includes the explicitly user-approved fresh
  seed2017 replacement `12z27y72`. Preserve interrupted2017 and guard-ownership
  incident `2xt6tnn_`; never resume it or combine its experience with another run.
- Preserve `runs/breakout-action-pilot-20260920.kNeotb/a4-train/HOLD.md`.
  Its eighteen-action Large pair is complete; the four-action arm never ran.
  Historical four-action gradient/state/pixel evidence uses an older backend,
  not automatic qualification of the latest one.
- Preserve historical reader failures, interrupted attempts and the September20
  unintended shader-pipeline test overlap. Their detailed evidence/corrections
  remain in immutable reports linked from the experiment index.
- Prefer one small source/config/artifact declaration and result record over
  recursively pinning entire research histories. Existing declarations, safety
  checks and acceptance gates remain binding.

## GPU operation

- Ordinary bounded GPU work is authorized on driver **580.178.04**. **No NVML**:
  no nvidia-smi, bindings, legacy health logger or vendor diagnostics. Use the GPU;
  do not add CPU learning fallbacks. Unavailable telemetry is unmeasured, not zero.
- Serialize GPU-heavy jobs. Use `python/examples/gpu_host_guard.py` around the
  direct native-bearing child, not a scheduler/Cargo/process tree. Bind boot,
  driver, executable and fixed inputs; review each result before starting its
  individually declared successor. No retries or automatic followers.
- Own each guard with a persistent systemd user service: `Restart=no`,
  `KillMode=control-group`, deadline slightly beyond the guard deadline.
  Do not depend on a tool session to keep it alive. The guard owns only its direct
  child; service cleanup prevents orphans if the guard disappears.
- Require native device assertions and >=2GiB Vulkan estimated budget headroom
  after GPU stages. Budget-minus-usage is not physical free or peak VRAM.
  Current boot: `4f5152d1-e5fd-46cf-a0c4-06534c430d26`. A changed boot/driver
  needs a distinct declaration, not editing historical results.
- Stop on kernel faults, native failures, invalid evidence or incomplete budgets.
  No reset/reload/reboot/power-cycle/driver changes without new user approval.
  Follow `docs/gpu_incident_response.md`. Historical Xid62/154 faults remain
  unexplained; successful no-NVML runs prove neither causality nor safety.
  Failed 75dfe, 0a98775/02b600a1 and 070f4b51/7db0d05c bundles stay quarantined.
- Keep active campaigns' pinned guards and auditors unchanged. Legacy
  `gpu_guard.py` remains for readers/helpers; its NVML launch paths are prohibited.

## Implementation discipline

Keep production code for exercised features; delete speculative staged code
instead of creating frameworks. Reuse unchanged qualified binaries: documentation
or source cleanup alone does not justify rebuilding or repeating qualification.
Keep decisions concise; detailed chronology belongs in reports.

Minimize monitoring overhead. Let guards/auditors handle routine validation.
Check long training about every 30 minutes or at completion, not every minute;
report meaningful results, not counters. Do not repeatedly audit unchanged
evidence or invent tooling to occupy a training window. GPU successors still
require individual review.

Use one CPU, 2GiB and zero swap for heavy preparation, with private targets and
packages; never compile during matched timing. Meganeura has an unignored GPU
library test: use reviewed CPU filters, not assumed-safe module filters. Run
relevant tests, formatting, Clippy and native numerical checks.

Preserve unrelated user work, especially the user's Meganeura/Blade worktrees.
Only the user merges PRs; commits/pushes are allowed. Keep history linear.
