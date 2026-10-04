# Kindle strategy reset: rationale and phase history

Written September 26, adopted September 27; direction updated October 4, 2026.
The authoritative execution order and current evidence are in
[the project roadmap](kindle_single_life_dreamer_plan.md#current-execution-order--strategy-reset)
and [PR31](https://github.com/kvark/kindle/pull/31). This file preserves the
strategy's rationale and historical Phase 0–2 proposals, not a second work queue.
PR29 is merged history. Original budgets, baselines and speed estimates below
must not be read as current measurements.

**October 4 user decision: CDP is the main path.** Fast iteration, small native/
upstream replication, the frozen/joint Tiny studies and the
[CDP evaluation](results/2026-10-04-cdp-learning.md) are complete. CDP improves
all three short Seaquest learning pairs at lower whole-agent cost; this is not
Atari-wide mastery. New main-path experiments explicitly use `--cdp`; the CLI
default is still learned RGB, retained as reference/fallback.

**Next:** Atari exploration/reward with CDP, starting with unassisted Freeway
(extrinsic-only versus one GPU-compatible mechanism, three seeds), then one
held-out exploration task, planned as Venture. Declare the bounded experiment
before launch. No training is started by this roadmap edit. Video priors follow,
then native games/GOG, transfer/retention and finally swarms. Asynchronous
learning is a later response to measured real-time constraints, not a prerequisite.

Phase 1's original 3x speed target remains an unmet, user-accepted stretch target.
Phase 2 used seven of ten replication attempts plus six separately declared Tiny
runs; no allocation remains open. The old 45-run matrix is cancelled (24 completed,
21 unstarted). No unchanged mastery queue, posterior-Tiny ablation or encoder
matrix resumes. Keep its failures and original measurements in the
[archive](experiments/README.md).

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

### 2b. Replication first, then qualify the cheap learning screen

The initial execution selected the largest proposed matrix: five methods ×
three games × three seeds, 200,004 actions each, Size12M/B16/T64/H15/R256/F32.
That is roughly 140 GPU-hours, not a fast engineering check. Six environments
and GPU inference/learning are already batched; the native learner calls/waits
account for about 98–99% of measured run time and emulation about 0.6%.
These stage timings are not GPU utilization measurements. More emulator
parallelism alone cannot remove the dominant cost.

The user approves replacing its unstarted work with this order:

**Budget cap (September 30): at most 10 new replication training runs total,**
counting upstream and native together and including failed/interrupted attempts.
Target three paired learner seeds (six runs), with at most four pilot/debug
attempts. A qualifying pilot can count toward the paired comparison if its
recipe is unchanged. Numerical unit/gradient checks remain small and bounded;
they are not additional learning campaigns. Report a remaining limitation when
the cap is reached instead of extending the queue. The subsequent JEPA comparison
needs an explicit compact allocation, not an automatic return to a large matrix.

1. **Faithful RGB baseline.** Replace the research patch CNN/dense decoder with
   the pinned upstream multiscale CNN/convolutional decoder for the baseline.
   Keep the GPU pixel path. Match normalization, output transform, losses,
   replay/reset semantics, optimizer and slow targets—not just the RSSM size.
   Do not preserve the research CNN as a checkpoint-compatibility branch.
2. **Fixed-batch numerical comparisons.** Use common weights, observations,
   flags/actions and explicit stochastic draws. Compare forward values, loss
   components, gradients and optimizer/EMA updates on small shapes, including
   reset/terminal/truncation boundaries and more than one update. Retain
   independent scalar/finite-difference references where useful. Passing native
   versus native execution variants is not a full upstream reproduction.
3. **Qualify a fast recipe before queuing seeds.** Start with the Size1M preset,
   batched environments and a declared short action budget. Time one bounded
   upstream/native pilot and check that upstream actually learns. Small capacity
   is a hypothesis, not guaranteed sufficient because the game looks simple.
   Target a sub-hour development comparison; increase capacity/budget only for
   a demonstrated limitation. The existing three-seed MinAtar screen finishes
   in 8m18s but has weak learning, so speed alone is not qualification.
4. **Matched learning regression.** Fix the selected recipe, then compare at
   least three learner seeds with score-vs-actions/time and uncertainty. A lower
   replay ratio or shorter sequence is a different screening recipe, not an
   optimization speedup at unchanged learning. No five-game mastery campaign.

Use the authors' [published Atari scores](https://github.com/danijar/dreamerv3/tree/main/scores)
and [configuration presets](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml)
as reference evidence. A local upstream control establishes the smaller/custom
recipe and native replication; it need not recreate the full published suite.
Published scores from other model sizes, interaction budgets or Atari protocols
are not directly matched targets for our short pilot.

**Completed October 1:** all three small upstream/native seed pairs pass,
using seven replication attempts including the interruption. Native/upstream
final online means are 368.0/307.733; native takes about 3.08× longer. This
[local qualification](results/2026-09-30-small-replication-learning.md) is not
the full published benchmark or frozen competence. The ten-attempt cap applies
to replication; it does not silently replace Phase 2c's separate declaration.

### 2c. Test the JEPA hypothesis on the qualified recipe

Keep the completed offline probes and all historical learning curves. The
[October 1 compact declaration](experiments/2026-10-01-small-jepa-comparison.md)
adds six Seaquest runs, reusing the three completed native RGB controls.
After 2b, declare a focused comparison of the faithful learned encoder, pretrained
Tiny and its own initial weights, with >=3 learner seeds and a held-out title.
Include Large only within the measured iteration budget or as a separately
justified confirmation. Fix methods, budgets and score summaries before running;
do not select a favorable completed seed from the stopped matrix.

**Decision rule:** a frozen encoder must earn its whole-agent cost through
useful probes *and* learning curves versus random/learned controls. Prefer the
simpler learned 2D frontend if a useful benefit is not established; do not call
an inconclusive small study proof that JEPA cannot work. Retain LeVJEPA as a
3D hypothesis. Native/JL64/mean remains the frozen-feature reference; existing
probes do not justify more pooling/PCA variants. Promote only promising changes
to larger confirmation, one factor at a time.

**Done when:** the native control has numerical and learning evidence against
upstream, the cheap representation comparison is published with its costs,
uncertainty and limitations, and the resulting frontend decision is implemented
and verified. Cancelling the expensive matrix does not by itself close Phase 2.

**Completed October 1:** all criteria above are met; [decision and verification](results/2026-10-01-frontend-decision.md).
The ten-attempt replication cap used seven attempts; the separate compact
JEPA study used its six declared runs and reused all three native RGB controls.
Do not extend either allocation or restart the cancelled matrix.

## 6. Phase 3 — Atari exploration and reward with CDP

CDP is the main-path agent; **extrinsic-only CDP**, not RGB or frozen Tiny, is
the control for exploration changes. Start on unassisted Freeway, three learner
seeds, with one bounded GPU-compatible intrinsic mechanism. Keep the qualified
small recipe and game protocol fixed; declare budgets/settings before launch.
No persistent random-action aid, new external reward shaping or pretraining.

Report first reward, extrinsic return versus actions/time, seed stability and
whole-agent cost; include predeclared frozen evaluations and complete rollout
videos. Do not use intrinsic return as game performance. The old CPU
feature-readback visitation workaround remains excluded.

Then confirm a selected mechanism on one predeclared second exploration task,
planned as Venture, with same-game controls. Seaquest's completed CDP screen is
the existing learning reference, not a reason to repeat it unchanged.
Model-disagreement, game-native shaping and model-generated rewards are separate
future alternatives, not four mechanisms to implement or queue together.

**Phase outcome:** useful unassisted learning across three seeds and improvement
on a second task at acceptable cost. A negative result is a reason to identify
the next limitation, not to expand the budget or revive the old mastery gates.
The [roadmap](kindle_single_life_dreamer_plan.md) owns the current sequence.

## 7. Phase 4 — Use video for dynamics and behavior

Build on the small online CDP agent. Replace "video only trains a frozen encoder" with:

1. **Action-free world-model pretraining.** Pretrain the RSSM (or a latent
   predictor) on gameplay video to predict next latents with actions masked;
   then fine-tune with real actions online. Compare fresh vs. pretrained
   dynamics at equal online budgets; start with one predeclared comparison.
2. **Latent or inferred actions.** Train an inverse-dynamics or latent-action
   model on a small labelled/online set, label the video, and pretrain a
   behavior prior (policy initialization or KL regularizer toward it).
3. Keep the target game out of action-conditioned pretraining unless the
   experiment explicitly tests same-game video, and disclose the corpus.

**Done when:** a pretrained variant reaches a fixed score threshold in fewer
online steps than the fresh baseline on ≥2 games, 3 seeds.

## 8. Phase 5 — Useful native-game learning, then real-time deployment

1. Integrate one native game through `/x/Code/mind-games`: vkQuake2 first, then
   TMNF. Qualify the current GPU capture ownership/reuse contract and complete
   sparse reward/terminal and action adapters. Old vkQuake plumbing is not game
   competence or qualification of the changed v4 producer.
2. Establish single-actor learning before adding services. In free-running games,
   measure capture-to-action p50/p95/p99 latency, observation gaps, learner debt
   and effective replay ratio. Add an asynchronous actor/learner only if those
   measurements require it; update `AGENTS.md` when deliberately starting it.
3. Extend to a small GOG/Wine panel and measure held-out cross-game adaptation
   and retention. These milestones precede swarm work.
4. Only then test experience sharing among Kindles, with explicit ownership and
   independent causal histories. Several environment streams are not a swarm.

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
