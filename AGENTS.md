# Kindle working direction

Kindle is a Rust agent that learns while acting. Games are the first testbed;
explicit rewards and human guidance are allowed. Favor minimalism, expressiveness,
safety and speed. Keep learning/inference on Meganeura + Blade; Python is for
adapters, controls and analysis. Follow `/mnt/data/GUIDELINES.md`.

## Product and research

- Keep one authoritative plan: `docs/kindle_single_life_dreamer_plan.md`, with
  one current game-status table and direct rollout/world-report links.
- Prioritize a reliable single actor: Atari, accelerated playing plus learning,
  video/world pretraining, mind-games vkQuake2/TMNF, GOG/Wine, then held-out
  cross-game adaptation and retention. Strong GOG/transfer results precede swarms.
  Do not build a concurrent learner service.
- Current gameplay uses native causal LeVJEPA, not DINO. The frozen frontend's
  16-arrival chunks reset perception only; episode boundaries also reset belief.
  The prediction head reads the deterministic prior before observing its target.
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
- Continue `runs/pong-block-confirmation-20260916.rBwdGF`: fresh roots
  2017/3019/1009, each training -> frozen evaluation -> fresh initialization ->
  untrained evaluation, then complete replay/video analysis. Root 2017 training
  completed 400,008 actions / 99,652 updates. Check live results before launching;
  never duplicate a phase. Details: `docs/experiments/README.md`.
- The original xPz5ud queue and its reserved `pong/seed2017-train.stdout` stay
  terminal. Never remove the hold, restart old followers or count historical
  root 1009 as a root in the new matched campaign.
- After Pong, test Breakout's prepared minimal-action hypothesis, then separately
  declared Freeway/Qbert exposure comparisons. Do not replicate unchanged failed
  recipes merely to occupy the GPU. Prepared branches are not adopted features.
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

Preserve unrelated user work. Only the user merges PRs; commits/pushes are allowed.
Keep history linear. Use one CPU, 2 GiB and zero swap for heavy CPU preparation,
with private targets/packages; do not compile during matched timings. Meganeura
has an unignored GPU library test: CPU-only checks require reviewed module filters.
Use relevant tests, formatting, Clippy and native numerical checks.
