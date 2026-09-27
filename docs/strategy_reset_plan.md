# Kindle strategy reset: execution plan

**Audience:** an engineering agent picking up Kindle.
**Date written:** 2026-09-26.
**Code base:** branch `dreamer-jepa-kickoff` (PR #29), head `16afaa7`. All file
paths below refer to that branch, not `main`.

**Adopted September 27.** Current execution/evidence is maintained in
[the project roadmap](kindle_single_life_dreamer_plan.md#current-execution-order--strategy-reset)
and PR #29. Some initial premises below are historical: Tiny is now default,
native-detail GPU acting/capture is integrated, and sticky Pong fails its
robustness check. The historical upstream speed numbers are not a reconciled
comparison. Phase 0 encoder CI/disclosures pass; compact human-normalized data
is in [docs/results](results/2026-09-27-historical-scores.md).
Phase 1's implementation and fast-screening work is now measured:
[GPU sampling/fused recurrence/grouped RSSM](results/2026-09-27-fused-learner.md)
gives 1.15× updates / 1.16× short Pong throughput; the **3× target remains unmet**.
[Three Size1M MinAtar seeds](results/2026-09-27-minatar-screen.md) complete in
8m18s with committed curves; the low scores do not establish strong learning.
Do not confuse completed engineering deliverables with achieving that speed
target or demonstrating the value of JEPA. The user explicitly accepts **3× as
a stretch target, not a Phase 1 exit gate**; these completed implementation and
screening deliverables close Phase 1. Phase 2 is next.
The 864-point gate is unsupported at the tested budget, not mathematically
impossible. Current GPU pixel collection does not support the old host-based
visitation bonus; any intrinsic mechanism must honor the GPU path.

## 1. Goal and why the plan changes

Kindle's goal: *learn to play any game by naturally playing with sparse rewards,
like a human does.* Target infrastructure: game, frame capture, perception and
policy all on the GPU; only input events (keyboard/mouse/joystick) are read back
to the CPU; low-latency real-time play.

PR #29 delivered a correct DreamerV3 port with a frozen LeVJEPA encoder and a
six-stream Atari runner. It passes Boxing, Pong and Freeway. It spent most of its
effort on per-game reliability gates for five easy Atari games. That work does
not test the parts of the goal that are actually hard:

- **Exploration.** Freeway only learned with ~50% random actions held for 64
  steps (`--exploration-probability .5 --exploration-hold 64`); an unassisted
  pilot earned zero reward. Nothing in the agent explores on its own.
- **Priors.** The frozen LeVJEPA encoder has shown no benefit: on Breakout at
  200k actions, pretrained Tiny scored 10.9 and the same network at random
  initialization 12.5 (one seed each). No matched comparison against upstream
  DreamerV3 or DINO exists.
- **Speed.** Kindle learns at ~15.6 actions/s at replay ratio 256. Upstream
  DreamerV3 12M did ~59 actions/s on the same machine. One Atari seed takes
  7–11 hours, so iteration is measured in days.

New priority order: **iteration speed → settle LeVJEPA → exploration and reward
→ video pretraining for dynamics/behavior → real-time async deployment.**

## 2. Working rules for this plan

These replace the heavier process in `AGENTS.md` for *development* work. Keep the
existing rules for any result claimed as a final, confirmed game result.

1. **Screen small, confirm big.** Develop methods on fast environments and small
   models, reading learning curves. Only promising changes graduate to the 12M
   Atari recipe and multi-seed confirmation.
2. **One variable per comparison, ≥3 seeds, report curves.** Report score vs.
   environment steps and vs. wall-clock time, with mean and 95% bootstrap CI (or
   IQM for suites). Use human-normalized scores for Atari. No pass/fail gates for
   development.
3. **Disclose every aid.** Any exploration override, shaped/intrinsic reward,
   pretraining corpus (list its games), or privileged signal goes into the
   result record and the plan table. Never remove such a disclosure later.
4. **Keep evidence reachable.** Every reported result gets a compact summary
   committed to `docs/results/` (JSON + short markdown: config, seeds, curves,
   final numbers). Large artifacts may stay in `runs/`, but the summary must not
   depend on them.
5. **GPU safety stays.** Keep using `python/examples/gpu_host_guard.py`, no NVML,
   and the incident runbook `docs/gpu_incident_response.md`. Dropping ceremony
   does not mean dropping these.
6. **Don't qualify micro-optimizations.** A speedup needs one matched timing
   comparison and an equality/tolerance check of learning outputs, not a
   multi-window campaign.

## 3. Phase 0 — Clean up PR #29 (small, do first)

1. Restore the two disclosures removed in commit `4ba22b6`, in
   `docs/kindle_single_life_dreamer_plan.md`, `README.md` and the PR body:
   - Freeway training used persistent random-action exploration (probability
     .5, hold 64); evaluation was unassisted.
   - Tiny encoder `7fe9b252` was pretrained on random-play frames from Boxing,
     Pong, Freeway, Breakout and Qbert (250,000 observations, 50k per game);
     code lives on branch `exp/levjepa-tiny-pretrain-20260920`. Link it.
2. Add a CI encoder test that runs on lavapipe: deterministic random Tiny
   weights, streaming path vs. a small dense block-causal reference, covering
   the 16-frame wrap, resets and multi-stream gaps. Remove the need for
   `--skip vision::` in `.github/workflows/ci.yml` for this test (hardware-pinned
   tests may stay ignored).
3. Add the human-normalized results table and state that Breakout's gate (864
   in ≥90% of episodes) is not reachable at 200k actions. Stop further Breakout
   gate runs.

**Done when:** docs updated, new CI test green, PR body updated.

## 4. Phase 1 — Iteration speed

### 1a. Remove host round-trips from the learner

Where the time goes: `kindle/src/dreamer/agent.rs`, `sample_posterior_batch`
(64 sequential steps, each with a readback and CPU categorical sampling) and
`imagine_and_target` (15 steps with readbacks), plus `sync_matching` parameter
copies through the host (~21 ms/update).

1. **GPU categorical sampling.** Pass pre-drawn uniform noise as a graph input
   and sample with Gumbel-max (`argmax(logits + gumbel(noise))`, including
   unimix) inside the graph. Keep RNG ownership on the CPU (draw noise from the
   existing `DreamerRngs` streams) so runs stay reproducible.
2. **Fused graphs.** Build the posterior pass as one unrolled graph over T=64 and
   imagination as one unrolled graph over H=15, with no intermediate readbacks.
   Read back only what the CPU still needs (λ-return inputs), or move λ-returns,
   return normalization targets and two-hot encoding onto the GPU as well.
3. **Shared parameters.** Make inference sessions reference the training
   session's parameter buffers (or copy GPU→GPU) instead of host copies.
   Account for derived weights (the plan notes this caveat).
4. Consider bf16 compute for non-sensitive matmuls after the above; upstream
   DreamerV3 uses bf16.

**Acceptance:** learning statistics match the current implementation within
tolerance on the tiny canaries and a short Pong run (same seed, distributions
compared, not bit-exact since sampling changes). **Stretch target:** ≥3× faster
updates, explicitly not a Phase 1 exit gate (user decision, September 27).
Report actions/s at replay ratio 256 vs. the current ~15.6.

### 1b. A fast development environment

1. Add a GPU-resident, sparse-reward environment for method development.
   Recommended: **Craftax** (JAX, open-ended achievements, very high
   throughput). If JAX interop with Meganeura/Blade is impractical, use
   **Crafter** (CPU) or **MinAtar** as a stopgap, and track GPU-resident Atari
   (e.g. CuLE) as the path toward the TASK.md infrastructure.
2. Write a thin adapter to the existing vector agent interface
   (`begin_episodes` / `act` / `observe` / `learn_scheduled`). Symbolic or
   small-image observations may bypass LeVJEPA through a small learned encoder;
   add that encoder path (see Phase 2).
3. Define a **screening recipe**: size-1M model, small batch, a budget that
   finishes in ≤1 hour, 3 seeds. Record the learning curve format once and reuse
   it.

**Done when:** one screening run of the current agent on the fast environment
finishes in ≤1 hour with a committed curve summary.

## 5. Phase 2 — Settle the LeVJEPA question

### 2a. Offline probes (about a day, no RL)

1. Record held-out frames plus RAM-derived state (ball/player/enemy positions,
   velocities) for 3–4 Atari games, including at least one not in Tiny's
   pretraining corpus (e.g. Seaquest, Frostbite, Private Eye).
2. For each feature extractor, fit linear (and small MLP) probes from the 7×7
   features to state:
   - LeVJEPA Large, pretrained Tiny, randomly initialized Tiny;
   - a small CNN trained end to end (reconstruction or the DreamerV3 encoder);
   - input variants: current 64×64-upscaled vs. native max-pooled frames;
   - pooling variants: 2×2 mean vs. space-to-depth (4 tokens × 16 channels);
   - projection variants: fixed random vs. PCA fitted once on Atari tokens;
   - chunk phase 0 vs. 15 for the same frame (quantify the phase effect).
3. Report R² per target, especially small fast objects (ball) and velocities.

### 2b. One matched learning comparison

3 seeds each, same protocol, same budget (e.g. 200k actions), 2–3 games
(Pong, Breakout, and one non-corpus game):

- upstream DreamerV3 12M (`python/examples/run_upstream_control.py`);
- Kindle + Large; Kindle + pretrained Tiny; Kindle + random-init Tiny;
- Kindle + a learned CNN encoder, if Phase 2a shows it helps.

**Decision rule:** if no LeVJEPA variant beats the random-init or learned-encoder
baseline on probes *and* learning curves, stop using a frozen encoder for 2D
games. Keep LeVJEPA only as a candidate for 3D titles and re-test there.
If the best probe variants (native input, space-to-depth, PCA) help, adopt them
one at a time with a learning comparison each.

## 6. Phase 3 — Exploration and reward

All experiments here: extrinsic-only control vs. one added mechanism, 3 seeds,
on sparse-reward tasks **without** the random-action aid. Development on the
fast environment; confirmation on Atari hard-exploration games (Freeway
unassisted, Private Eye, Venture; Montezuma's Revenge as a stretch).

1. **Visitation bonus (already implemented, never used).** Enable
   `visitation_bonus` with a nonzero `intrinsic_reward_scale`
   (`kindle/src/dreamer/intrinsic.rs`). Cheapest first test: unassisted Freeway.
2. **Latent disagreement (Plan2Explore-style).** Add an ensemble of K small
   one-step predictors of the next latent (or next frozen feature) from
   (deter, stoch, action); intrinsic reward = ensemble variance, computed in
   imagination. Train the actor on a mix of intrinsic and extrinsic rewards,
   optionally with separate critics.
3. **Game-native signals.** Add life-loss as a configurable signal (small
   negative reward and/or continuation drop) to the Atari wrapper
   (`python/examples/atari.py`, read `ale.lives()`); keep it declared and
   ablated. Target Qbert and Breakout first.
4. **Model-based reward (experimental).** A vision-language model scores
   progress or proposes subgoals from frames every few seconds; its output is a
   separate, scaled reward channel. Compare against extrinsic-only; watch for
   reward hacking by inspecting videos.

**Done when:** at least one mechanism learns unassisted Freeway (3/3 seeds
nonzero and rising) and shows a gain on one other sparse-reward task.

## 7. Phase 4 — Use video for dynamics and behavior

Replace "video only trains a frozen encoder" with:

1. **Action-free world-model pretraining.** Pretrain the RSSM (or a latent
   predictor) on gameplay video to predict next latents with actions masked;
   then fine-tune with real actions online. Compare fresh vs. pretrained
   dynamics at equal online budgets (the plan already specifies this matrix).
2. **Latent or inferred actions.** Train an inverse-dynamics or latent-action
   model on a small labelled/online set, label the video, and pretrain a
   behavior prior (policy initialization or KL regularizer toward it).
3. Keep the target game out of action-conditioned pretraining unless the
   experiment explicitly tests same-game video, and disclose the corpus.

**Done when:** a pretrained variant reaches a fixed score threshold in fewer
online steps than the fresh baseline on ≥2 games, 3 seeds.

## 8. Phase 5 — Real-time GPU-resident deployment

1. Split into a low-latency GPU **actor** and an asynchronous GPU **learner**,
   with GPU-side replay writes and periodic GPU-side weight updates. This
   deliberately lifts the current "no concurrent learner service" rule; update
   `AGENTS.md` when starting this phase.
2. Measure p50/p95/p99 action latency, learner debt, and effective replay
   ratio under real-time play.
3. Multi-Kindle experience sharing = multiple actors writing to one replay
   (Ape-X/IMPALA pattern), only after a single real-time Kindle works.
4. First real-time target: a native game from `/x/Code/mind-games`
   (vkQuake2 preferred), with frame capture kept on the GPU.

## 9. What not to do

- Don't run more five-game reliability gates or three-root confirmations until
  a method changes.
- Don't tune Atari-only tricks (action subsets, gate tweaks) as progress toward
  the goal.
- Don't mix changes: speed changes (Phase 1) must not alter the learning recipe.

## 10. Reference points (from memory; verify before citing)

DreamerV3 (Hafner et al.), Plan2Explore (Sekar et al. 2020), RND (Burda et al.
2018), Go-Explore (Ecoffet et al.), BBF (Schwarzer et al. 2023), EfficientZero
V2, Craftax (Matthews et al. 2024), PureJaxRL (Lu et al. 2022), CuLE (Dalton et
al. 2020), EnvPool, APV (Seo et al. 2022), VPT (Baker et al. 2022), Genie (Bruce
et al. 2024), LAPO (Schmidt & Jiang 2023), Motif (Klissarov et al. 2023), ELLM
(Du et al. 2023), OMNI (Zhang et al. 2023), V-JEPA 2, Ape-X, IMPALA.
