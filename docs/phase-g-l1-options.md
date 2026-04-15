# Phase G — L1 option-policy layer

## Goal

Add a second learned layer above the existing reactive policy so that
kindle can make decisions on a **coarser clock** than the env step.
L0 (everything we have today) stays the reactive controller over
primitive actions; L1 is a policy over discrete **options**, each
decoding to a goal-latent that conditions L0 for the option's
duration. L0 is what moves the joints; L1 is what decides what to do.

For `num_options = 1` (or equivalently a trivial option that always
picks a zero goal-latent) the runtime is behaviourally equivalent to
today's L0-only kindle — Phase G is a strict superset.

## Why L0 alone isn't enough

Three symptoms from the Phase E LunarLander runs (`N = 64`, 3000
synchronous steps, soft-landing rate stuck at ≈ 5%) point at L0's
structural ceiling rather than L0's tuning:

1. **The policy can never commit to a strategy.** `policy_entropy`
   pinned at `ln(num_actions)` — uniform. Each env step is an
   independent draw from the policy. There is no mechanism in the L0
   design that would ever cause a sequence like "fire main engine for
   10 steps" to be sampled as a coherent unit. Landing is a committed
   behaviour that unfolds over hundreds of steps; L0 samples it as a
   sequence of independent choices.

2. **The credit horizon is an order of magnitude too short.**
   `history_len = 16`, LunarLander episodes are 100–300 steps. The
   credit assigner structurally can't attribute "started decelerating
   early" → "soft landing 150 steps later". No amount of data fixes
   this — the graph is sized for 16. An L1 operating on a 20×-slower
   clock would see the whole descent in a single 10-option history
   window.

3. **Per-lane landing variance is bimodal.** All 64 lanes share
   parameters, yet soft-landing counts ranged from 0 (in 18 lanes) to
   7 (in 1 lane). That spread means the policy isn't representing
   "how to land" as a reusable skill — it's representing a
   distribution over per-step actions that *accidentally* lands
   sometimes. Skills (options) are how you turn that accident into a
   learned competence.

## Cheap precursor: K-step action persistence (not L1)

Before committing to a trained L1, take the trivial step: force L0's
sampled action to persist for K env steps. This is a degenerate "L1"
that always picks the same option as L0 — no new network, no new
reward pathway, no new credit assigner. Five to ten lines of code.

Why it's worth running first:

- It isolates whether the *entropy-at-uniform* problem is the
  dominant bottleneck. If K = 5 action persistence moves the
  soft-landing rate meaningfully, L1 becomes high-priority; if it
  doesn't, L1 wouldn't either — L0's action *distribution* itself is
  the problem.
- It stretches the effective credit horizon by K× with zero model
  changes. History still 16, but 16 persistent actions = 16 × K env
  steps.
- It lets us tune K alongside `history_len`, `warmup_steps`, and LR
  before adding a second learner.

This is implemented in Phase E.v3 (not Phase G). Phase G takes over
once persistence's ceiling is visible.

## L1 design

### API additions

```rust
pub struct AgentConfig {
    // ... existing fields ...
    pub num_options: usize,     // 1 = L0-only (default). Phase G: 4–16.
    pub option_dim: usize,      // goal-latent width. = latent_dim by default.
    pub option_horizon: usize,  // K env steps per option (fixed for v1).
    pub lr_option: f32,         // option-policy LR; ~ lr_policy × 0.5.
}
```

### New per-lane state

```rust
struct Lane {
    // ... existing ...
    current_option:   Option<u32>,      // index into the option set
    option_goal:      Vec<f32>,         // [option_dim] — decoded goal latent
    option_steps_left: usize,           // counts down from option_horizon
    option_return:     f32,             // reward accumulator for this option
    option_start_z:    Vec<f32>,        // latent when option was chosen
}
```

### New GPU graph

One additional compiled session: `option_session`. Inputs and outputs:

```
inputs:
  z             : [N, latent_dim]         // current latent
  option_taken  : [N, num_options]        // one-hot of the option taken
                                          //   during training rollback
  option_return : [N, 1]                  // advantage target for L1
outputs:
  loss          : scalar                  // option-policy MSE
  option_logits : [N, num_options]        // for sampling
  option_value  : [N, 1]                  // L1 value baseline
  goal_latent   : [N, num_options, option_dim]
                                          // one decoded goal per option
```

Shape-wise this is a tiny MLP — `latent_dim → hidden → num_options` for
logits, `latent_dim → hidden → num_options × option_dim` for the goal
decoder. No attention, no unrolls. Trains under the same Phase E
batched pattern as the existing policy graph.

### L0 conditioning

The existing policy graph's `"z"` input becomes `concat(z, g_o)` where
`g_o` is the goal-latent for the lane's currently-selected option:

```
inputs:
  z             : [N, latent_dim + option_dim]   // was [N, latent_dim]
  action        : [N, MAX_ACTION_DIM]
  value_target  : [N, 1]
```

The concat adds `option_dim` to the policy's input width — a small
graph change (one extra input slot), no retrain from scratch at
Phase G bring-up because `g_o = 0` for the L0-only default gives L0 a
constant-zero goal and the graph trains to ignore it. With
`num_options > 1`, L0 starts learning to produce actions that drive
`z_{t+1}` toward `g_o`.

### Per-step flow (Phase G with Phase E batching)

```
for each step:
    1. For each lane, check option_steps_left:
         - if 0: it's time for an L1 decision.
           Sample new option from option_session outputs. Fetch
           goal_latent[lane, new_opt] into lane.option_goal.
           Reset option_steps_left = option_horizon, option_return = 0.
         - else: keep current option, decrement option_steps_left.
    2. Build stacked policy input:
         [N, latent_dim + option_dim] =
           [z_stack, option_goal_stack] per lane.
    3. L0 policy + value: one batched forward, per-lane actions.
    4. Step envs, compute rewards (per-lane, as today).
    5. Batched WM forward+backward (as today).
    6. Accumulate reward into each lane's option_return.
    7. For lanes whose option just terminated at step 1 above (i.e.
       option_steps_left rolled over), run option_session backward:
         - target = option_return from the just-completed option
         - advantage = option_return − last_option_value
         - goal_achievement_bonus = −||z_end − option_goal|| (optional,
           promotes L0 actually reaching the goal)
    8. Credit + L0 policy update (as today).
    9. Replay (as today).
```

### Option termination

**v1: fixed-horizon.** `option_horizon = 10` env steps, committed. No
learned termination. Simplifies the option-MDP to a semi-Markov with
fixed option length.

**v2 (out of scope for v1): learned termination.** Adds
`termination_head : [N, 1]` to option_session — sigmoid probability of
terminating the current option. Option return is computed at the real
termination step, not a fixed horizon. More powerful, more to debug;
drop in once v1 is stable.

### L1 reward

L1's training signal is the **option return** — sum of L0 rewards
accumulated over the option's window. Plus optionally a
goal-achievement bonus `−||z_end − option_goal||` that encourages L0
to actually reach the goal (without it, the goal-decoder can collapse
to meaningless vectors).

Crucially, this is *not* a new reward primitive. Everything in
`reward_circuit` stays as-is — frozen, per-step, the same
surprise/novelty/homeo/order. L1 just sees a coarser-grained
aggregation.

### Credit assignment at L1

Phase G v1: **no L1 credit assigner.** The option_session is trained
with one-step TD on `option_return` as the advantage, full stop. The
Phase E credit assigner continues to work at L0 granularity only.

Phase G v2 could add an L1 credit head over (say) the last 8 options,
same contrastive-attention pattern as L0's credit. Defer until v1 is
landing softly.

### Shared state between lanes

Same discipline as Phase E: **all network weights are shared across
lanes; all buffers and local state are per-lane.** L1 adds one more
shared network (the option policy) and one more per-lane state
tuple (`current_option`, `option_goal`, `option_steps_left`,
`option_return`, `option_start_z`).

## Parity guarantees

- `num_options = 1, option_horizon = 1` → byte-identical to Phase E
  L0-only. The option policy is a 1-way softmax (always picks option
  0), the goal-latent is constant, the L0 policy's input-concat of a
  fixed zero is ignorable at training time.
- `num_options ≥ 2, N = 1` → parity with a hypothetical single-lane
  L1 reference (same as Phase E's parity promise for L0).
- Unit-test target: run the `option_horizon = 1, num_options = 1`
  agent for 500 steps on GridWorld and diff the diagnostics against
  a current-main run at the same seed. All fields must match to
  numerical noise.

## Things the Phase G design **does not** do

- **Multi-GPU / distributed training.** Same scope as Phase E — one
  session, one GPU.
- **Variable-length options (learned termination).** v2 concern.
- **L1 credit assigner.** v2.
- **Nested hierarchies (L2, L3).** Not until L1 is reliably useful.
- **Replace L0.** L0 stays. L1 sits above L0; both train
  concurrently.
- **Change the frozen reward primitives.** Surprise / novelty /
  homeostatic / order all keep their per-step semantics. L1 just
  aggregates them over the option window.

## Minimum-viable Phase G scope

For the first commit of this phase, we build:

1. `num_options`, `option_horizon`, `option_dim`, `lr_option` in
   `AgentConfig`.
2. `option_session` compiled graph with the shape above.
3. Per-lane option state on `Lane`.
4. Option-sampling path in `act()`; option-training path at option
   termination.
5. `z` input to `policy_session` widened to `latent_dim + option_dim`;
   goal-latent stack fed in alongside.
6. Diagnostics: per-lane `current_option`, `option_return`,
   `last_option_loss`, `last_option_entropy`.
7. Byte-parity test at `num_options = 1`.

LunarLander acceptance bar: **soft-landing rate doubles** vs the
Phase E.v3 (action-persistence) baseline at the same step budget.
If it doesn't, Phase G v1 doesn't ship — we iterate on the goal
representation and option count before investing in v2.

## What we learn from running it, win or lose

Whatever the LunarLander numbers look like, Phase G is also the
diagnostic instrument for the claim that "commitment matters." If L1
lifts soft-landing sharply: the reactive per-step policy was the
ceiling. If it doesn't: the bottleneck was elsewhere (reward shaping,
representation capacity) and adding layers hasn't helped. Either
outcome sharpens the picture.

## Rough priority ordering (once we decide to build it)

1. **Phase E.v3** — K-step action persistence. No new network.
   Measure the lift on LunarLander. *This file does not cover
   E.v3; see `python/examples/lunar_lander_batch.py --action-repeat`
   once the switch is added.*
2. **Phase G v1** — fixed-horizon L1 over a small discrete option
   set. Goal-latent concat into L0.
3. **Phase G v2** — learned termination + L1 credit.
4. **Phase H** (notional) — skill library, transfer across envs, L1
   transplantation when hopping.
