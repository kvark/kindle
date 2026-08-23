# ARC-AGI-3: first integration and diagnosis

> **2026-08-23 correction:** ARC retains `levels_completed` across
> `env.reset()` calls. The April “90%/100% L1 episode reach rate” was therefore
> a cumulative-state artifact: after the first L1 event, later attempts still
> reported level 1. The event itself was real, but repeated L1 reliability and
> policy mastery were not demonstrated. Across five current 30k-step seeds,
> random produced 7 CD82 events and reached L2 twice; the historical `full`
> Kindle preset produced 5 events and never passed L1. Current harness
> summaries report level gain per attempt and expose executable
> event/max-level gates.

## Pulled

- `arc-agi` python toolkit (0.9.8) and `arcengine` (0.9.3) — both
  available on PyPI without the full agents repo's heavy LLM
  dependencies (langchain / smolagents / langgraph etc.).
- Anonymous API key auto-obtained via `arc_agi.Arcade()` — no
  signup needed for local play.
- Local execution of 25 shipped ARC-AGI-3 environments via
  `arc_agi.LocalEnvironmentWrapper`, gym-style `reset()` /
  `step(action)` interface.

## Adapter

`python/examples/arc_agi3_batch.py` — minimal kindle↔ARC-AGI-3
bridge:

- **Observation**: 64×64 int8 grid (colours 0–15) → mean-pool to
  8×8 (64 cells) → flatten to 64-dim float in `[0, 1]`. Matches
  kindle's `OBS_TOKEN_DIM = 64`.
- **Action**: kindle emits a fixed discrete index 0..6, always mapped to
  `GameAction.value == index + 1`. A dynamic mask removes values absent from
  the current `available_actions` list without changing policy semantics.
  Kindle's shared policy head is padded to `MAX_ACTION_DIM = 18`. Complex
  actions additionally stage their exact normalized `(x, y)` in the
  world-model-only parameter tail (`WM_ACTION_DIM = 20`).
- **Homeo**: two variables:
  - persistent distance to the reported win level; this becomes less negative
    when `levels_completed` increases.
  - frame-entropy proxy (1 − unique_colours/16): small negative
    when the agent is in a visually monotonous state.
- **Attempt boundary**: `state ∈ {NOT_PLAYED, GAME_OVER, WIN}`
  or `ep_step ≥ max_episode_steps`. Completed-level count can persist across
  these resets, so an attempt is not an independent fresh game.

### Remaining adapter limitations

- **Complex-action planning**: `ACTION6` (and any other action reported as
  complex) receives seeded-random coordinates or coordinate-head samples, and
  the exact executed values train the world model. The built-in planner still
  searches only the discrete action identity and supplies a zero coordinate
  tail; it cannot optimize or queue `(action, x, y)` jointly.
- **Multi-frame state** (`frame` is a list; only frame[0] used).
  ARC-AGI-3 games can present multiple frames per step; we
  drop all but the first.
- **Visual-pattern reasoning**: the whole point of ARC-AGI-3 is
  that the agent reasons about the 64×64 grid. Our 8×8 mean-pool
  preprocessor destroys most of the visual information that a
  real ARC solver would need.

This is intentional — we're checking kindle's ability to learn
from a sparse-signal interactive env, not ARC-solving ability.

## Results

Tested on `ls20` (keyboard game, 4 simple actions, 7 levels to
win, baseline L1 = 22 expert actions).

| agent | steps | episodes | level events | notes |
|---|---|---|---|---|
| Random | 500 | 3 | **0** | baseline — random never reaches L1 |
| Kindle (advantage_clamp=20, no M6/M7) | 2000 | 15 | **0** | same as random; entropy pinned at 1.38 ≈ ln(4) |

Kindle's policy entropy stays at near-maximum (1.38 / 1.386)
for 2000 steps. No commitment, no level-1 reach. Same failure
mode we characterized extensively on LunarLander: the reward
circuit provides no signal that differentiates useful from
useless action sequences BEFORE the first level-completion
event, and level-completion is astronomically rare under a
random policy (L1 needs 22 expert actions from a 4-way choice,
so random completion probability is roughly 4^(−22) ≈ 10^(−13)
per attempt).

## Why this was expected

Structurally, ARC-AGI-3 is an even more extreme instance of the
task class LunarLander exemplified:

- **Event-ordered terminal success**, not steady-state homeo.
- **No dense shaping**: no approach signal, no distance-to-goal,
  no per-step reward. The *only* signal is `levels_completed`
  rising by 1 at sparse moments.
- **No natural homeostatic variable** to track. Unlike
  LunarLander (altitude, velocity, angle — well-defined
  quantities we can target), ARC-AGI-3 games have arbitrary
  internal logic per game; there is no universal homeostat.
- **Pure discovery**: no instructions, no rules. By design,
  agents must figure out the game. This explicitly tests the
  capability kindle's reward class doesn't cover.

The M6 PPO ceiling test (commit 486c129 in prior PR) showed
that even a best-in-class RL optimizer (sb3 PPO) cannot land
LunarLander under kindle's v3 homeo — because the reward's
optimum isn't landing. ARC-AGI-3 is worse: there isn't even a
hand-designed homeo shaping that approximates a ground-truth
dense reward. The intrinsic reward kindle has access to
(surprise, novelty, homeo, order) contains ~zero information
about level-completion.

## What would be needed to make kindle-class agents competitive
## on ARC-AGI-3

Multiple stacked problems, each genuine:

1. **Visual-pattern encoder**: the 64×64 grid is the agent's
   entire state. A conv-net encoder is the obvious choice;
   kindle's current encoder is a dense MLP on an 8-dim-range
   obs token vector. Redesign of the encoder stack is needed.
2. **Complex-action search**: several games need precise coordinate actions.
   Kindle now has a separate coordinate head and conditions learned dynamics
   on the executed `(x, y)`, while retaining an 18-way discrete policy label.
   What remains is joint planning/search over the discrete action and its two
   continuous parameters (or a principled finite object-centric action set).
3. **Goal / milestone discovery beyond kindle's current M7**:
   we showed M7's self-supervised prototype ranking fails on
   LunarLander where success is well-defined in obs-space
   features. ARC-AGI-3 success is a position in game-internal
   state space that the agent can only know by completing a
   level once — a bootstrap problem M7 doesn't solve.
4. **Stateful exploration**: ARC-AGI-3 explicitly tests
   exploration efficiency. Kindle's per-step-entropy
   exploration is near-random; meaningful progress would need
   novelty-driven, information-gathering exploration (RIDE,
   NGU, or similar).

Each of those four is a substantial research direction.
None of them fit within M6/M7 shaped patches.

## Status

ARC-AGI-3 integrated mechanically (installable, running,
stepping, recording). No learning on the tested game. The
integration is committed as a standalone example and documents
the gap precisely — useful for future project decisions about
scope.

## 2026-04-20 update: CNN encoder + curiosity tuning

Landed as first-class AgentConfig: `EncoderKind::Cnn { channels,
height, width }` (see commit 241cbec). When set, kindle builds
a small conv-net over the raw visual input (flat NCHW), folds in
the task embedding, and produces the latent for WM. Task
embedding post-fused via a small linear projection. Replay and
drift-measurement paths are guarded off in CNN mode (no visual
frames stored on transitions). Byte-parity on MLP-encoder gym
envs preserved.

### Result on ls20 (2000 steps, same harness):

  encoder  wm@500  wm@1500  steps/s  level events
  MLP      0.074   0.079    370      0
  CNN      0.016   0.010    294      0

**CNN WM loss is 7-8× lower** — the conv successfully models the
64×64 grid where the MLP-on-mean-pool couldn't. Throughput
penalty is ~20% (conv op overhead). But no level events.

### Curiosity sweep (5000 steps each, CNN encoder):

Tested several primitive-weight combinations:

  config                                        nov   surp  hom    ep_ent   events
  surprise=10, novelty=2, homeo=0.1, grid=0.5   0.02  0.34  −0.34  1.38     0
  surprise=1,  novelty=50, homeo=0.1, grid=0.5  0.02  0.34  −0.34  1.38     0
  surprise=1,  novelty=20, homeo=0.1, grid=0.05 0.02  0.34  −0.34  1.38     0

Surprise and novelty primitives are both STUCK at low values
regardless of weights. Once the CNN's WM loss settles (~0.007),
the surprise primitive (L2 norm of prediction error) is
constant at ≈ 0.34 across all actions, so the per-action
differential is ~0 — no gradient direction for the policy.
Novelty is similarly flat: the CNN compresses game states to a
tight latent region where visit counts are quickly uniform.
Reducing `grid_resolution` to 0.05 didn't help because the
problem is latent clustering, not bucket granularity.

Entropy pinned at 1.38 / 1.386 throughout.

### Why neither primitive works here

- **Surprise** stops being informative once the WM predicts well.
  Classic curiosity-death problem — by design, prediction-error
  decays as training converges. Fixable with techniques that
  decouple curiosity from WM quality (Random Network
  Distillation, predictor ensembles, etc.), but none of those
  are in kindle today.
- **Novelty** with latent-space visit counts fails when the
  encoder clusters all game states tightly. Would need raw-obs
  buckets or per-image hash keys — neither is kindle's current
  design.

### Exposed config knobs (useful regardless)

  - `AgentConfig::reward_weights` exposed via Python:
    `reward_surprise`, `reward_novelty`, `reward_homeostatic`,
    `reward_order`. Lets harnesses tune the primitives per env
    without rebuilding kindle.
  - `AgentConfig::grid_resolution` exposed via Python:
    `grid_resolution`. Lets harnesses tune novelty-bucket
    granularity per env.
  - `--encoder {mlp,cnn}` on the ARC-AGI-3 harness.

### Honest next-step options

Concretely for ARC-AGI-3 specifically:

  (a) **Random Network Distillation** curiosity — a fixed random
      target net + trained predictor; prediction error doesn't
      decay as WM converges. Adds a 4th intrinsic primitive and
      a second small session. Solid ~200 LOC.

  (b) **Raw-obs visit counts** — hash frame contents (or a
      spatial digest) into visit buckets instead of using the
      latent. Decouples novelty from encoder convergence.
      Smaller change, but the digest needs some design thought
      to avoid bucketing all game states together.

  (c) **Coordinate action head** — needed for games that use
      complex actions. Kindle's MAX_ACTION_DIM=6 currently only
      supports 6 simple actions (*superseded: now 18, so more
      discrete slots are available*); adding `(x, y)` parameters
      still requires either a new action shape or a separate
      continuous-coord head.

None of these solves the *reward-class* problem — level
completion is still outside the primitive class. But any of (a)
or (b) would give kindle exploration strong enough to at least
probe the game's reachable state space, which random and
current-curiosity policies don't.

## 2026-04-20 update: Random Network Distillation landed

Added `AgentConfig::rnd_reward_alpha` + supporting knobs
(`rnd_feature_dim`, `rnd_hidden_dim`, `rnd_lr`) and a new CPU
module `kindle/src/rnd.rs` implementing a minimal RND primitive:
2-layer frozen random target MLP + trainable predictor MLP, both
operating on the obs TOKEN (pre-encoder, 64-dim). Predictor
trains every step by MSE against the target; intrinsic reward is
the predictor's current squared error. Unlike kindle's surprise
primitive, RND's target is independent of the WM so the signal
doesn't decay when the WM converges.

4 unit tests verify: positive MSE at init, predictor converges
to target on repeated input, novel states yield higher reward
than familiar ones, target network frozen (never drifts).

Critical design choice: RND reads the **obs token**, not the
encoder latent. An early experiment using `z_row` directly
produced MSE ≈ 0.01 within 1000 steps — the CNN encoder
clusters all game states into a tight latent region where the
predictor can fit the target quickly on any input. Switching to
the pre-encoder 64-dim obs token gave MSE ≈ 0.06 → 0.02 decay
over 4000 steps, matching the classical RND signal shape.

### Result on ls20 (5000 steps, CNN + α=50 + homeo=0.1):

  step  wm    pi      entropy  rnd_mse  level_events
  1000  0.011 +4.95   1.37     0.06     0
  3000  0.007 +1.60   1.35     0.02     0
  4000  0.007 +0.86   1.35     0.01     0

**This is the first configuration where kindle's policy commits
on an ARC-AGI-3 game.** Entropy drops from 1.39 (max for 4
actions = ln(4) = 1.386) down to 1.35 — small but real
departure from uniform. Policy loss trajectory (+4.95 → +0.86)
shows meaningful learning. No level events yet; the specific
22-action expert sequence on ls20 isn't in the direction RND
randomly pushes the policy toward.

### What RND actually delivered

Mechanism: ✅ RND signal behaves as designed (initial high
MSE, decay as predictor fits, meaningful per-step magnitude
when scaled by α ≈ 50). Entropy drops. Policy committed.

Result: entropy-floor-breaking on kindle-with-curiosity was
previously uncrossable; now crossed. But commitment direction
is RND-arbitrary, not task-aligned. Without a reward signal
pointing at level-completion (which kindle's primitives still
don't provide), committing to arbitrary directions doesn't
solve 22-action puzzle games.

### Updated honest list of what kindle needs for ARC-AGI-3

  ✅ Vision encoder (commit 241cbec): WM models grids 7-8×
     better than MLP-on-mean-pool.
  ✅ RND curiosity: breaks entropy-at-max failure mode.
  ❌ Reward signal aligned with level-completion: still missing.
     Structural. Would require external supervision or
     self-supervised milestone discovery beyond M7's prototype
     ranking.
  ❌ Coordinate action head: needed for games with ACTION5/6/7
     (x, y). Most of the 25 shipped envs need these.

RND removes a real blocker, but the remaining blockers
(reward-class, coord actions) are each real additional
research/engineering items.

## 2026-04-20 update: complex action support + first level event

Two small changes to the ARC-AGI-3 harness:

1. **Complex actions enabled** — the adapter no longer filters
   out ACTION6 (the one coordinate-parameterized action kindle
   needs to handle for several games). Random `(x, y) ∈ [0, 63]`
   coords are attached per step via `GameAction.set_data()`.
   Kindle's policy controls the discrete action ID; the coords
   are exploratory noise. Games sb26, ft09, su15, cd82 that
   require ACTION6 now run without errors (they previously
   raised `KeyError: 'x'`).

2. **Homeo signal fixed** — the original implementation
   emitted `value = −delta_levels · scale`, which produced a
   one-step NEGATIVE spike at level completion (penalizing
   progress). Replaced with `value = (win_levels −
   levels_completed) · scale` — a persistent positive-valued
   deviation that decreases in magnitude as levels are
   completed, giving a sustained "distance-to-win" signal
   instead of a one-frame anti-reward at level-ups.

### First level event on cd82

    game=CD82 id=cd82-fb555c5d available_actions=[1, 2, 3, 4, 5, 6] win_levels=6
    config: CNN encoder, RND α=50, homeo weight 0.5, levels-reward-scale 5.0

At step 2000–3000: agent completes level 1 for the first time.
The historical claim that 90% of later episodes independently reached L1 was
invalid: `levels_completed` remained at 1 across resets. Never reaches level 2.

This is the **first level event on any ARC-AGI-3 game** under any
kindle configuration we've tested. Proves the infrastructure end
to end: vision encoder → RND-driven exploration → complex
actions → homeo pointing at distance-to-win. It does not by itself prove
credit propagation or that the policy learned the transition.

But the agent then stalls. Possible reasons (each testable
separately):

- RND saturates as predictor fits the post-L1 state
  distribution. Once rnd_mse → 0, curiosity stops driving.
- The policy may not have learned the discovered L0→L1 behaviour at all;
  comparison against multiple random seeds was not performed.
- Max_episode_steps = 400 might cap attempts before L2 sequences
  are tried. cd82 baseline L2 = 8 expert actions; random prob
  from the retained L1 state ≈ 6^−8 ≈ 10^−6, so feasible budgets may
  still miss it.

Other games (ls20, ft09, su15, sb26) still show 0 level events
in 15k steps with the same config. Possible reasons: more
restrictive action sets ([6] alone for ft09), different
specific-sequence requirements, spatial-clicking (complex
action coord specifics that random can't find).

### Updated blocker status for ARC-AGI-3

  ✅ Vision encoder (commit 241cbec): WM 7-8× better.
  ✅ RND curiosity (commit f24996b): entropy drops below max.
  ✅ Complex action plumbing (this commit): games with ACTION6
     now run; sb26/ft09/su15/cd82 playable.
  ✅ Homeo signal corrected: level-ups give positive advantage
     instead of one-step penalty.
  ✅ **First level event achieved on cd82.**
  ⚠ Multi-level progression: one L1 event was observed on cd82,
    but no repeat-learning advantage or L2 progression was established.
  ❌ Coordinate action policy: `(x, y)` still random, not
     policy-controlled. Games that need SPECIFIC coords for
     progress are gated by chance.
  ❌ Multi-game generalization: only cd82 shows level events.
     Most ARC-AGI-3 games still require specific action
     sequences that kindle's exploration doesn't find.

## 2026-04-20 update (2): RND reset + coord action head

Two more primitives added, matching the "continue" discussion:

1. **`Agent::reset_rnd_predictor()`** — re-initializes the RND
   predictor weights, keeping the target frozen. Re-activates
   curiosity when the state distribution shifts (e.g. on
   level-up). Exposed to Python as
   `BatchAgent.reset_rnd_predictor()`; ARC harness calls it
   automatically with `--rnd-reset-on-level` whenever
   `delta_levels > 0`. Unit test verifies: reset spikes MSE on
   a previously-familiar state, target weights stay identical.

2. **`coord` module + `CoordHead`** — CPU MLP
   `z → (μ_x, μ_y) ∈ [−1, 1]`, Gaussian-REINFORCE-trained. New
   `Agent::sample_coords(&mut rng)` returns per-lane `(x, y)`
   samples; `Agent::train_coord_head()` runs the REINFORCE
   update using the last step's reward (EMA baseline-centered,
   `α = coord_action_alpha`). ARC harness samples coords
   before `env.step`, scales `[-1, 1]` → `[0, 63]`, attaches
   via `GameAction.set_data()`. Now kindle controls both the
   discrete action ID and the `(x, y)` clickpoint — instead of
   random noise as before. 4 unit tests (forward-range,
   per-lane sample caching, positive-advantage-pulls-toward-
   sample, zero-advantage-noop) all pass.

### Results on cd82 (30k steps each, CNN + all primitives):

Each configuration produced one L1 event. The previously reported 90%/100%
reach rates counted the retained cumulative level after reset and are invalid.
These runs establish discovery of one transition, not a learned, repeatable
L1 policy.

But **no L2 events**. RND-reset fires on L1, MSE spikes 0.01 → 0.12,
and MSE decays again within roughly 5000 steps as the predictor fits the
post-L1 distribution. The former interpretation—that the policy repeatedly
reached L1 and then stopped—is unsupported. L2 still requires an 8-action
specific sequence from the retained L1 state, which was not discovered.

### sb26 (coord-controlled clickpoints):

Entropy drops to 1.09 (near max for 3 actions = ln(3)=1.099),
policy commits slightly, coord head trains every step. But 0
level events in 20k steps. Without any level events to
bootstrap from, the coord head's REINFORCE advantage signal is
~uniform (all rewards comparable, baseline tracks), so the
coord policy wanders randomly. Consistent with the
"exploration without ground truth" failure mode.

### Updated infrastructure checklist

  ✅ Vision encoder (commit 241cbec)
  ✅ RND curiosity (commit f24996b)
  ✅ Complex action plumbing (commit 019f3df)
  ✅ Fixed homeo signal
  ✅ First level event on cd82
  ✅ RND predictor reset
  ✅ Coord action head (kindle-controlled clicks)
  ⚠ One L1 event on cd82; repeatability over random is unproven
  ⚠ Multi-level progression: still gated on per-game dynamics
    and requires finding level-completion sequences that our
    current exploration doesn't discover (cd82 never sees L2,
    sb26 never sees L1).

### Remaining honest options for ARC-AGI-3 progression

Each is genuine research-scale work:

  - **Staged curriculum**: train on 2-3 adjacent level
    transitions explicitly. Breaks the "local-optimum at L1"
    problem by giving the agent L2-adjacent initial states
    via save-state replay.
  - **Learned go-explore**: archive novel states, periodically
    reset the agent to archive states for deeper exploration.
    Known technique for sparse-reward envs (Ecoffet et al.
    2019).
  - **Self-supervised sub-goal discovery**: explicitly identify
    state transitions that are POTENTIALLY level-boundaries
    (e.g. cluster "mode shifts" in the CNN latent) and give
    the agent sub-rewards for triggering them. Kindle-flavored
    variant of option discovery.

The current infrastructure supports any of these — we have
clean slots for additional primitives, a working vision
encoder, and a functional RND+coord stack to build on.

### M8 v1 — delta-goal bank (state-delta-triggered), 2026-04-20

First try at the "self-supervised sub-goal discovery" direction:
`DeltaGoalBank` in `kindle/src/delta_goals.rs`. Every per-step
obs-token delta above `delta_goal_threshold` writes `obs_cur` into
a rolling bank (size `delta_goal_bank_size`); `merge_radius` drops
near-duplicates. Reward each step is `-α · min_i ‖obs − g_i‖`
(clamped), pulling the policy toward the nearest bank entry.

Note: like RND, M8 reads the 64-dim obs TOKEN, not the post-encoder
latent. First run with the latent had `dg=0` across 2000 steps on
cd82 — the CNN latent clusters tight enough that near-zero per-step
deltas wash out the threshold. The obs token carries the raw
per-frame variation.

**A/B on cd82 (10k steps each, seed 42, full CNN+RND+coord stack):**

| config           | eps | levels (mean/end) | lvl_events | dg bank |
| ---------------- | --- | ----------------- | ---------- | ------- |
| no M8 (baseline) | 99  | 0.99 / 1          | 1          | 0       |
| M8 α=0.3 th=0.3  | 99  | 0.99 / 1          | 1          | 64 full |

**Task-level outcome identical.** Bank fills to capacity within
~500 steps, then saturates.

**Mechanism**: the bank represents the agent's routine trajectory
points, not genuinely goal-worthy states. Because the agent already
visits these states, the nearest-distance reward is ~uniformly
small everywhere it goes — no gradient toward new behavior.

**Next iteration (M8 v2)**: gate recording on world-model
prediction error, not raw obs-delta. "Record `obs_cur` as a goal
only when the WM failed to predict the transition that led here."
This banks surprising transitions, not routine ones. Still
self-supervised — no task-specific priors.

### M8 v2 — surprise-gated recording, 2026-04-20

Added `delta_goal_surprise_threshold`: a step's goal-candidate is
only banked when `pred_error >= threshold` AND `|Δobs| >= delta`.
`prev_obs` is updated regardless of the gate so the next step's
delta is measured against the immediate previous obs.

**cd82 (10k steps, full CNN+RND+coord stack, A/B/C):**

| config         | dg bank | eps | levels end | lvl_events |
| -------------- | ------- | --- | ---------- | ---------- |
| no M8          | 0       | 99  | 0.99 / 1   | 1          |
| v1 (no gate)   | 64      | 99  | 0.99 / 1   | 1          |
| v2 surp th=0.5 | 62      | 99  | 0.99 / 1   | 1          |
| v2 surp th=1.0 | 0       | 99  | 0.99 / 1   | 1          |

**sb26 (10k, same stack):**

| config         | dg bank | eps | levels end | lvl_events |
| -------------- | ------- | --- | ---------- | ---------- |
| no M8          | 0       | 45  | 0.00 / 0   | 0          |
| v2 surp th=0.5 | 1       | 45  | 0.00 / 0   | 0          |

**All four configs produce identical task-level outcomes across
both games.** The surprise gate correctly filters routine steps
(62 vs 64 on cd82 at th=0.5, 1 vs 64 on sb26) but the resulting
bank still doesn't alter policy progression. At th=1.0 the gate
blocks everything (pred_error ∈ [0.27, 0.56] post-convergence, never
crosses 1.0) — equivalent to M8 off, still identical outcome.

**Interpretation.** The M8 primitive (both variants) is empirically
null on ARC-AGI-3 with the current CNN encoder and 10k-step
budgets. The cd82 L1 and sb26 L0 policies are both nearest
local optima under existing reward primitives; an additional
distance-to-bank reward doesn't change that. This is consistent
with the `project_kindle_structural_cap.md` evidence: the
bottleneck is finding the specific multi-step sequences that
trigger level events, not the reward primitive class.

**Next lever**: cross-episode memory (ARC 2 in the reward-class
roadmap). Each ARC episode currently restarts fresh; if the agent
could retain "I tried action-sequence X → result Y" across attempts
within a game, systematic exploration of untried sequences becomes
possible. This is where "reach L1 every episode but never try
what's past L1" would become "reach L1 AND carry the memory that
nothing worked there last time, so try something different now".

### xeps memory — cross-episode state-action novelty, 2026-04-20

Module `kindle/src/xeps_memory.rs`: persistent (quantized_obs, action)
counter shared across lanes, spanning episodes. Per-step intrinsic
bonus `α / sqrt(1 + count(prev_obs, prev_action))`. Obs-keyed (not
latent-keyed) because the CNN-encoded latent saturates to 1-2 grid
cells on ARC — verified before switching (xeps=6 at default grid,
xeps=12 at fine grid, vs xeps=2200+ with obs-keying).

**cd82 A/B (10k steps, full stack):**

| config                | xeps pairs | eps | levels/end | lvl_events |
| --------------------- | ---------- | --- | ---------- | ---------- |
| no xeps               | 0          | 99  | 0.99 / 1   | 1          |
| xeps α=0.1 grid=0.05  | 2202       | 99  | 0.99 / 1   | 1          |
| xeps α=0.3 grid=0.05  | 2322       | 99  | 0.99 / 1   | 1          |
| xeps α=1.0 grid=0.05  | 2322       | 99  | 0.99 / 1   | 1          |

At α=1.0, entropy held pinned at ln(6)=1.79 — xeps reward dominated
the advantage signal and the policy stayed fully stochastic. At
α=0.1, entropy behaved normally but the signal was too weak to
change task outcomes. No α produced a level event that baseline
didn't also produce.

**sb26 A/B:** xeps=3 at α=0.3 (obs is near-static — the whole 10k
trajectory collapses into one obs cell × 3 actions). Same 45/0/0.

**Interpretation.** Cross-episode state-action novelty, at both
extremes of the saturation spectrum (tight obs → few pairs;
loose obs → many pairs but α required is too high to commit), does
not move cd82's L1→L2 gap or sb26's L0→L1 gap. Reward-primitive
experiments M6, M7, M8 v1/v2, RND, RND-reset, coord-head, and xeps
have all now been tried; **ALL null on ARC-AGI-3 multi-level
progression** under 10k-step budgets.

The L1→L2 gap on cd82 needs 8 specific actions in sequence AT L1.
At the agent's ~10 steps/L1 of random-with-xeps exploration, the
probability of stumbling on a correct 8-step action sequence is
~6^-8 ≈ 6e-7 — essentially zero in 10k steps. This is not a reward
circuit problem; it's an exploration-depth / action-abstraction
problem that no per-step intrinsic reward can solve.

**Next remaining lever**: **action macros** (ARC 3 in the roadmap).
Cluster frequently-occurring action subsequences into atomic
options the policy can commit to in one decision, shortening the
effective episode depth. Or model-based planning: use kindle's
world model to roll out hypothetical action sequences and pick
one that's predicted to reach a novel state. Both are substantial
architecture work beyond reward-primitive tweaks.

### Action-macro injection (harness experiment), 2026-04-20

Before committing to an agent-level macro-discovery module, tested
the upstream hypothesis cheaply: harness-side random k-action
injection. `--macro-len K --macro-inject-prob p` plays a uniformly
random K-action sequence committed over K consecutive steps.

**cd82 results (all with full CNN+RND+coord stack):**

| macro_len | prob | steps | macros | eps | levels/end | lvl_events |
| --------- | ---- | ----- | ------ | --- | ---------- | ---------- |
| 3         | 0.05 | 10k   | 329    | 99  | 1.00 / 1   | 1          |
| 5         | 0.10 | 10k   | ~200   | 99  | 0.81 / 1   | 1          |
| 8         | 0.02 | 10k   | ~25    | 99  | 0.90 / 1   | 1          |
| 8         | 0.05 | 30k   | 906    | 299 | 0.97 / 1   | 1          |
| 8, +xeps  | 0.05 | 15k   | ~440   | 149 | 0.93 / 1   | 1          |

**All null on L2**, consistent with the combinatorial argument:
L1→L2 requires 8 specific actions in sequence; random 8-sequences
hit the target at prob `6⁻⁸ ≈ 6e-7`; 906 random macros at prob
~5% started-at-L1 gives `906 × 0.05 × 6e-7 ≈ 2.7e-5` expected
successes — effectively zero at any feasible budget.

The historical “mean-levels” comparison for longer macros is not a valid
regression metric: it is dominated by when the first persistent event occurred,
not by independent per-attempt competence.

### Summary — what works and what doesn't on ARC-AGI-3

Tried and null on L2 progression within 10k–30k steps on cd82 after an L1
event had moved the persistent environment frontier:
- M6 outcome head (v1 + trajectory v2 + RTG v3)
- M7 approach reward (v1 + confidence-weighted v2 + novelty ranking)
- M8 delta-goal bank (v1 + WM-surprise-gated v2)
- RND curiosity + RND predictor reset on level
- Coord action head
- xeps cross-episode state-action novelty (α ∈ {0.1, 0.3, 1.0})
- Random action-macro injection (lengths 3/5/8)
- xeps + macros combined

Helped partially:
- RND + coord head: one historical L1 event, no demonstrated advantage over random
- Fixed homeo signal: present on the first observed L1 event

Not yet tried (would break generality constraint):
- Expert trajectory bootstrapping
- Curriculum with L1 save-states
- Hand-coded symbolic priors

**Conclusion (2026-04-20).** kindle's cold-start self-training
paradigm, combined with per-step intrinsic reward primitives and
naive exploration, cannot close the L1→L2 gap on cd82 within
feasible step budgets. This is a sample-complexity argument, not
an algorithmic one: the 8-action-sequence depth requires 1.7M
candidate sequences; kindle explores maybe 2000 random sequences
per 10k steps. Under the "preserve generality — no task priors"
constraint, this game is beyond kindle's reach.

**ARC-AGI-3 is not the right testbed for kindle algorithm work.**
Gym canaries (GridWorld/CartPole/Acrobot/Pendulum) and LunarLander
remain productive — shorter sequence depth, more responsive to
reward-class changes. Recommend redirecting to Lander 1 (homeo
decomposition) next per the April 20 roadmap.

### Track 3 — model-based planning with the WM, 2026-04-21

`kindle/src/planner.rs`: pulls the WM's 3-layer MLP weights into a
CPU cache, exposes `WmRollout::rollout(z0, actions) → trajectory`.
Agent method `plan_and_queue(num_actions)`: samples
`planner_samples` random K-action sequences, rolls each through the
frozen WM, scores by `Σ_t 1/sqrt(1+visit_count(predicted_z_t))`,
queues the best for `act()` to consume. +5 unit tests (62 total).

cd82 A/B (10k steps, full CNN+RND+coord stack, seed 42):

| config                        | mean levels | lvl_events |
| ----------------------------- | ----------- | ---------- |
| baseline                      | 0.99        | 1          |
| planner K=4 every=2 (100%)    | 0.00        | 0          |
| planner K=4 every=20 (20%)    | 0.89        | 1          |
| planner K=8 every=40 (20%)    | 0.64        | 1          |

Lander v6 A/B (50k steps):
  baseline v6:      5.9% soft, pi_loss 0.84 end
  v6 + planner K=4: 4.5% soft, pi_loss -51.3 end (destabilized)

Null to negative on both testbeds. Root cause: the WM has never seen
the cd82 L1→L2 transition (we never reach L2), so its predictions
for any action sequence FROM L1 are untrained-noise. The
novelty-score over noise picks arbitrary sequences; the agent
executes them; still random exploration at the L1→L2 gap. No
improvement over the `--macro-len 8` random-injection experiment.

On Lander, the WM IS well-trained (we see many landing/crash
trajectories), but the forced-action sequences bias policy
gradients off-policy and destabilize pi_loss.

**Track 3 infrastructure is kept** (sound CPU rollout, weight
refresh, plan-queue API — a general tool that may work on envs
where the WM is well-trained AND the reward gradient is
non-sparse). **But it does not solve the ARC-AGI-3 structural
cap** because the cap's origin is sample complexity on a sparse
reward, not under-planning.

**Final verdict for ARC-AGI-3 under the generality constraint**:
kindle's cold-start self-training paradigm, even augmented with
(in order of landed infrastructure) vision encoder, RND,
RND-reset-on-level, coord head, M6 learnable reward, M7 approach
reward, M8 delta-goals v1/v2, xeps cross-episode memory, random
action macros, and WM-based planning, cannot close the L1→L2 gap
on cd82 at feasible budgets. The gap needs 8 specific actions in
sequence; all general exploration mechanisms produce ~6e-7-hit-
rate attempts. The only viable directions violate generality
(demos, curriculum, symbolic priors). ARC-AGI-3 is off the
research path.

### 2026-08-23 correction and coordinate probe

The “final” wording above is a historical conclusion, not a current removal of
ARC from the project goal. Later validation corrected three invalid assumptions:
`levels_completed` persists across resets, fixed action indices must not follow
the changing position in `available_actions`, and different ACTION6 coordinates
must not share one dynamics token. Kindle now keeps an 18-way discrete policy
label while appending exact normalized `(x, y)` to a 20-value world-model token
through online, replay, surprise-replay, and k-step paths.

Coordinate REINFORCE now uses per-lane baselines and ARC supplies explicit
positive level deltas for click credit instead of the dense combined intrinsic
reward. A matched seed-42 5k-step probe on the six ACTION6-only games still did
not beat uniform random: learner 1 game/1 event/max L1 versus random 2 games/2
events/max L1. This closes a representation bug, not the ARC outcome gate.
Broader or object-centric coordinate exploration remains open.

The next remediation added joint coordinate shooting. ARC declares which
stable action slots consume `(x, y)`; batched MPC samples those values, rolls
the full 20D token through the WM, and queues the winning identity/coordinate
pair for exact execution. Stale queued actions are dropped when dynamic
availability changes. The discrete MCTS implementation automatically yields to
shooting because it cannot compare more than one coordinate per action.

The mechanism passes a synthetic GPU gate: after learning that negative x
changes state and positive x is a no-op, change-scored shooting selected
`x=-0.927`; its prediction was 0.0387 from the changed-state latent versus
0.2858 from the no-op latent. Real FT09 evidence remains neutral. Across seeds
42–44 at 5k steps, shooting produced two total level events/max L1, exactly the
same as uniform random; a denser seed-42 schedule produced none. The planner
transport/search bug is closed, but novelty/change is not yet a sufficient ARC
task-progress objective.

### Object-centric action correction, 2026-08-23

The first object-click probe could not start because the example imported
SciPy solely for connected components even though SciPy was not an ARC extra.
A deterministic 4-connected flood fill now handles the 64×64 frames without
that dependency. Validation also found two semantic errors in the initial
object proposal scheme: it assumed background color zero (FT09's is modal
color 5), and bounding-box centers could miss concave components.

After modal-background inference and on-component representatives, a one-step
FT09 sweep still showed that all eight state-changing cells were excluded by
the 11-slot policy budget: pure area ranking placed them at ranks 21–33.
Ranking components nested inside larger panels first moves those cells to
slots 0–7. Object slots now remain deterministic semantic actions in the world
model and planner; only the base ACTION6 exposes a freely searched coordinate.

Across seeds 42–44 at 5k total environment steps, this corrected object action
set produced one L1 event in each run (3 total/max L1), compared with uniform
coordinates at 2 total/max L1. This is directional evidence for structured
exploration, not a solve: no run reached L2, so task-progress scoring and
multi-step object interaction remain open.

### Clone-backed object-state search, 2026-08-23

The fixed 18-way policy can expose only 11 object-click slots after ARC's seven
base actions. That is already insufficient on FT09 L1, which has 13 distinct
state-changing cells. Raising the universal action width would penalize every
other environment and still fail on larger variable object sets, so the next
implementation uses ACTION6's existing continuous parameter channel instead.

`arc_agi3_object_search.py` runs only on cloneable local environments. It
extracts every visible component proposal, probes and deduplicates one-step
visual outcomes, and searches cloned states for an external level increment.
It empirically detects actions that are visually involutive and pairwise
commutative; for those levels it enumerates increasing-index combinations and
never explores reordered copies of the same action set. A cropped visual key
can exclude volatile one-pixel HUD borders such as FT09's countdown.

This produces the first deterministic multi-level ARC result in the project:
FT09 reaches L2 with four L0 clicks plus seven L1 clicks after 2,978 expanded
states. VC33 reaches L2 in ten solution actions and 45 expanded states. At a
500-state, depth-8 L0 cap over the six ACTION6-only games, FT09, LP85, R11L,
and VC33 reach L1 (4/6), versus 2/6 for the prior uniform-coordinate baseline.
S5I5 and TN36 remain at L0. FT09's next level has a 23-action basis and did not
reach L3 within 20,000 expanded states (78,798 states discovered), showing the
remaining combinatorial limit.

This is honest simulator-backed planning, not evidence that Kindle's learned
policy generalizes. Its next useful role is curriculum generation: execute the
found coordinate paths through Kindle, retain exact parameterized transitions,
and distill them into SIL/value/goal learning so future attempts need less
clone search.

## 2026-08-23 dynamic search and joint policy retention

The original search froze its root action basis for the whole level. That
pruned actions and coordinates that were initially no-ops but became useful
after another move. Non-commutative levels now regenerate proposals at every
expanded state, while verified commuting involutions retain the cheaper static
combination path. The harness also caps attempted cloned transitions and skips
engine-rejected branches instead of aborting the remaining games.

This adds L1 paths for SP80, CD82, LF52, and SU15. A curated eight-game gate
reaches all eight in 246 expanded states; over all 19 games exposing a complex
action, 8 reach L1 under 200–500-state caps. The broader result has no matched
random baseline and remains clone-backed planning.

Search demonstrations now store action availability and a hybrid object/pixel
token. Pure object tokens aliased four distinct LF52 states that required
different coordinates. Action identity trains through a dedicated positive,
masked cross-entropy update rather than a synthetic-reward/value workaround.
The coordinate head consumes the raw token plus Kindle's fixed task embedding;
its former random latent input added a GPU dispatch and made outputs depend on
training versus evaluation batch topology. The deterministic task context later
proved necessary to prevent coordinate interference when L2 demonstrations
were added to the joint corpus.

A 39-sample joint checkpoint reaches 100% action-label accuracy and coordinate
MSE 0.002191. Its raw coordinate head initially reached L1 on 7/8 games after
fresh batch-1 reloads: SU15 reproduced three clicks, then predicted `(31,33)`
instead of `(28,35)`. Nearest-object snapping chose another intervening cell
and still failed. Projecting after a generic constant-motion prior (activated
only after two matching executed displacements) reproduces all seven SU15
clicks and closes the fresh-checkpoint L1 gate at 8/8. No clone access is used
during evaluation. This is retained imitation with an object-aware executor,
not autonomous discovery or unseen-level transfer.

### Demonstrated L2 retention

The distiller can now merge compatible demonstration files, remove exact
duplicate model-visible states, and reject conflicting labels rather than
silently optimizing an impossible target. Combining the eight-game L1 corpus
with FT09/VC33 searches through L2 produces 53 unique rows after seven repeated
L0 rows are removed.

A shared token-only coordinate head at aggregate MSE 0.002194 regressed R11L's
second click and reduced the L1 gate to 7/8, even though both deeper games
replayed to L2. Adding the fixed task embedding removes this cross-game
ambiguity. The earlier MSE freeze threshold of 0.0022 was also too weak: a
fresh checkpoint just below it still missed a discrete object boundary.

A reproducible one-stage run with a 35,000-epoch ceiling and freeze threshold
0.0015 reaches 100% action-label accuracy and coordinate MSE 0.001792. After
fresh batch-1 reloads, one mixed-depth gate reaches L1 on all eight games and
L2 on FT09 and VC33. Evaluation clears the click-motion history at level
transitions, so a repeated-coordinate sequence from one level cannot leak into
the next. This extends retention to demonstrated multi-level paths, but still
does not show unseen-level transfer or online reward-driven discovery.

### VC33 L3 search, failed fixed heads, and pointer retention

A bounded deeper run reaches VC33 L3: the L2→L3 segment takes 23 solution
actions, 1,124 expanded states, and 34,067 attempted transitions; including
the retained L0/L1 paths gives 33 actions and 1,169 expansions total. Unlike
FT09's larger combinatorial level, this is practical clone-backed curriculum
generation.

It does not transfer through the current coordinate head. The combined corpus
contains 76 unique model-visible rows. Shared regression, equal-per-task
gradient weighting, and a VC33-only fit all stop at L2. A temporary alternative
turned deterministic object ranks into discrete actions and widened the
universal head from 18 to 64, but four capacity/rate probes fit at most 86.8%
of the training rows. Required ranks reach 50 while the compact hybrid token
explicitly represents only six objects. The widened head was therefore
removed rather than making every environment pay for an interface that could
not retain the demonstrated program.

The search export now records `object_index`, `object_candidate_count`, and 12
normalized features for every candidate. A generic two-layer pointer scores
that current variable set after Kindle selects the stable complex-action
identity. It fits all 67 complex labels in the 76-row joint corpus. A separate
zero-update process reloads the categorical policy and pointer checkpoints and
reaches L1 on all eight games, FT09 L2, and VC33 L3 in 33 actions. No clone,
coordinate projection, or motion extrapolation is used during evaluation.

This closes demonstrated L3 retention while keeping `MAX_ACTION_DIM = 18`.
The result remains behavior cloning on search-generated paths: unseen-level
transfer and online reward-driven discovery are still open.
