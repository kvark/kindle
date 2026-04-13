# IRIS

A continually self-training RL agent built on [meganeura](https://github.com/kvark/meganeura) — a cross-platform Rust neural network library using GPU-accelerated training and inference via [blade-graphics](https://github.com/kvark/blade).

The agent starts from a cold network (no pretrained policy), trains perpetually from experience, and is designed to recognize and attribute its own reward signals without external supervision beyond a frozen reward circuit grounded in three primitives: **surprise**, **novelty**, and **homeostatic balance**.

-----

## Table of Contents

- [Vision](#vision)
- [Architecture Overview](#architecture-overview)
- [Modules](#modules)
  - [Encoder](#encoder)
  - [World Model](#world-model)
  - [Reward Circuit (Frozen)](#reward-circuit-frozen)
  - [Credit Assigner](#credit-assigner)
  - [Value Head](#value-head)
  - [Policy](#policy)
- [Temporal Buffer](#temporal-buffer)
- [Tri-Level Learning Loop](#tri-level-learning-loop)
- [Continual Learning Strategy](#continual-learning-strategy)
- [Meganeura Confidence Plan](#meganeura-confidence-plan)
- [Implementation Phases](#implementation-phases)
- [Diagnostics and Observability](#diagnostics-and-observability)
- [Known Hard Problems](#known-hard-problems)
- [Open Questions](#open-questions)
- [Repository Structure](#repository-structure)

-----

## Vision

Most RL agents are trained in bounded episodes with hand-crafted reward functions and fixed environment contracts. This project explores a different shape: an agent that **trains continuously from first contact**, where reward is not handed down but derived from the agent’s own internal experience of the world.

The core claim is that three grounded, environment-agnostic reward signals — surprise, novelty, and homeostatic balance — are sufficient to bootstrap meaningful learning behavior, and that a learned temporal credit assigner can replace hand-tuned discount schedules as the agent matures.

This is a research and engineering project. Stability and correctness are pursued in that order.

-----

## Architecture Overview

```
 Raw Observation (o_t)
        │
   ┌────▼──────────┐
   │   Encoder E   │──────────────────────────────────┐
   └───────────────┘                                   │
        │  latent z_t                                  │
        │                                              │
   ┌────▼──────────────────────────────────────────┐   │
   │  World Model  W(z_t, a_t) → ẑ_{t+1}          │   │
   │  Loss: || ẑ_{t+1} − stop_grad(z_{t+1}) ||    │   │
   └───────────────────────────────────────────────┘   │
        │                                              │
   ┌────▼──────────────────────────────────────────┐   │
   │  Reward Circuit  (FROZEN)                     │   │
   │  R(z_t, z_{t-1}, action_history) → r_t        │   │
   │  Hardcoded primitives:                        │   │
   │    · Surprise   (world model prediction err)  │   │
   │    · Novelty    (density / count-based)       │   │
   │    · Homeostatic (env-defined signals)        │   │
   └───────────────────────────────────────────────┘   │
        │  r_t                                         │
   ┌────▼──────────────────────────────────────────┐   │
   │  Credit Assigner  C(r_t, history[t-H:t])      │   │
   │  Causal attention → α_i per past timestep     │   │
   │  credit_i = r_t × α_i                         │   │
   │  Scope H_eff = Σ_i (i × α_i)  (diagnostic)   │   │
   └───────────────────────────────────────────────┘   │
        │  credit[t-H:t]                               │
   ┌────▼──────────────────────────────────────────┐   │
   │  Value Head  V(z_t) → V̂                      │◄──┘
   │  TD bootstrap for variance reduction          │
   └───────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────┐
   │  Policy  π(z_t) → action distribution        │
   │  Updated by credit-weighted policy gradient   │
   └───────────────────────────────────────────────┘
```

All modules except the Reward Circuit are trained continuously through experience.

-----

## Modules

### Encoder

Converts raw observations into a compact latent representation `z_t`. Architecture is environment-dependent and defined at agent construction time:

- **Perceptual environments** (pixels): convolutional backbone → flatten → linear projection
- **Structured environments** (feature vectors): MLP with layer norm

The encoder is the shared backbone. All other modules consume `z_t`, not raw observations. This is intentional — the encoder is forced to learn representations that are simultaneously useful for world modeling, credit attribution, and policy generation.

Training signals flowing back into the encoder:

- World model prediction loss (primary)
- Policy gradient (secondary, scaled down)
- Value head TD error (secondary)

### World Model

A forward dynamics model that predicts the next latent state given the current latent state and action:

```
W : (z_t, a_t) → ẑ_{t+1}
Loss = MSE(ẑ_{t+1}, stop_grad(z_{t+1}))
```

The stop-gradient on the target prevents the encoder from collapsing (encoding everything to zero minimizes prediction error trivially). This is the same technique used in BYOL and Dreamer v3.

The world model serves two roles:

1. A self-supervised training signal that shapes `z` to be predictively meaningful
1. The **surprise component** of the reward circuit — high prediction error = high surprise

### Reward Circuit (Frozen)

The reward circuit is intentionally frozen during the early and mid phases of development. It will not receive gradient updates. Its weights are fixed at initialization.

This is a deliberate simplification. A learnable reward recognizer introduces a feedback loop (reward shapes policy, policy shapes observations, observations shape reward) that is destabilizing before the rest of the system is well-characterized. Revisiting this is explicitly scoped to a later phase.

The circuit computes `r_t` as a weighted sum of three primitive signals:

**1. Surprise** `r_surprise`

```
r_surprise = || W(z_{t-1}, a_{t-1}) − z_t ||₂
```

The L2 norm of the world model’s prediction error at the current timestep. High surprise → the agent transitioned to a state it did not expect. This signal is already computed by the world model training pass; the reward circuit reads it at zero additional cost.

**2. Novelty** `r_novelty`

A count-based approximation of state novelty. The latent space is partitioned into a grid (or a learned cluster set); each region maintains a visit count `N(z)`. Novelty reward is:

```
r_novelty = 1 / sqrt(N(z_t))
```

As a region is visited more, its novelty reward decays toward zero. This encourages exploration of unfamiliar state regions early in training, naturally fading as the agent becomes experienced.

Implementation note: exact counts over a continuous latent space require a density estimator. The initial implementation will use a fixed-resolution grid hash over a normalized latent space. A learned density model (e.g. a small flow network) is a later-phase upgrade.

**3. Homeostatic Balance** `r_homeo`

This signal is **environment-defined** and treated as an input to the agent, not computed by it. The environment exposes a vector of homeostatic variables (e.g. energy level, damage state, resource depletion) and a target range for each. The reward is:

```
r_homeo = −Σ_i max(0, |h_i − target_i| − tolerance_i)
```

Negative when any variable drifts outside its tolerance band; zero when all variables are in range. This signal is orthogonal to the agent’s internals — the agent doesn’t know what the homeostatic variables mean, only that deviating from target is costly.

The environment must implement the `HomeostaticProvider` trait (see `src/env.rs`).

**Combined Reward**

```
r_t = w_s · r_surprise + w_n · r_novelty + w_h · r_homeo
```

Weights `w_s`, `w_n`, `w_h` are hyperparameters set at agent construction. They are not learned. Surprise and novelty weights are expected to decay over training as the value head and policy mature; a configurable annealing schedule is provided.

### Credit Assigner

The credit assigner answers the question: *which past actions caused the reward I just received?*

Rather than using a fixed exponential decay (TD-λ with fixed λ), we use a causal self-attention network over the recent history buffer:

```
CreditNet(r_t, [(z_{t-H}, a_{t-H}), ..., (z_t, a_t)]) → α ∈ R^H
credit_i = r_t × α_i         (α normalized via softmax)
```

The attention is **causal** — position t only attends to positions t-H through t, never future positions.

**Training signal for CreditNet**

CreditNet is trained via a contrastive objective. When the agent visits similar latent states at different times and receives different rewards, the credit assigner should learn to attribute the difference to the diverging action sequences, not the shared context. Concretely:

- Sample pairs of timesteps (t, t’) where `||z_t - z_{t'}||` is small but rewards diverge
- The credit assigner should assign high attention to the steps where the action sequences diverged
- Implemented as a contrastive loss over pairs sampled from the experience buffer

This is bootstrapping from noise in the first phase. Expect poor credit attribution early. Track `H_eff` as the primary diagnostic.

**Effective temporal scope**

```
H_eff(t) = Σ_i (i × α_i)
```

This is a weighted average of how many steps back the credit assigner is looking. Track this over training:

- `H_eff` growing → the agent is learning that actions have longer consequences (healthy)
- `H_eff` shrinking → the agent is becoming myopic (investigate)
- `H_eff` oscillating → credit assigner is unstable (reduce learning rate)

### Value Head

A small MLP mapping `z_t → V̂(z_t)`, an estimate of future cumulative reward from the current state. Trained via TD(n) using the credit-adjusted reward signal.

The value head serves two purposes:

1. Variance reduction for policy gradient updates (standard actor-critic)
1. A secondary consistency signal — when the value head’s prediction diverges from realized credit, the discrepancy is diagnostic of either credit assignment error or world model error

### Policy

A standard stochastic policy network `π(z_t) → distribution over actions`. For discrete action spaces: categorical. For continuous: diagonal Gaussian.

Updated by the credit-weighted policy gradient:

```
∇L_policy = -Σ_i credit_i · ∇ log π(a_i | z_i)
```

With an entropy bonus to prevent premature collapse:

```
L_entropy = −β · H[π(· | z_t)]
```

β is annealed over training.

-----

## Temporal Buffer

All modules draw from and write to a shared circular buffer:

```rust
pub struct ExperienceBuffer {
    capacity: usize,
    observations: RingBuffer<Tensor>,  // raw o_t, for re-encoding
    latents:      RingBuffer<Tensor>,  // z_t
    actions:      RingBuffer<Action>,
    rewards:      RingBuffer<f32>,     // r_t from reward circuit
    credits:      RingBuffer<f32>,     // credit_i from credit assigner
    pred_errors:  RingBuffer<f32>,     // world model error (= surprise)
    visit_counts: HashMap<StateKey, u32>,  // for novelty
}
```

The buffer is the agent’s sole persistent memory. There is no episodic boundary — experience accumulates continuously.

**Replay mixing**: 20% of each gradient batch is sampled randomly from earlier in the buffer. This provides a re-anchoring signal that reduces catastrophic forgetting without requiring explicit EWC machinery in the early phases.

-----

## Tri-Level Learning Loop

Three learning processes run in parallel but at different learning rates:

|Module               |Learning Rate          |Update Frequency|Notes                                    |
|---------------------|-----------------------|----------------|-----------------------------------------|
|Encoder + World Model|`lr_wm` (base)         |Every step      |Self-supervised, most stable             |
|Credit Assigner      |`lr_credit` (0.3× base)|Every step      |Slower; contrastive signal is noisy early|
|Policy + Value       |`lr_policy` (0.5× base)|Every step      |Gated on value head warmup               |

The policy update is gated: it is suppressed for the first `N_warmup` steps until the value head has seen enough data to produce stable estimates. This prevents the policy from chasing noise before the value baseline is meaningful.

-----

## Continual Learning Strategy

“Always trains” means catastrophic forgetting is a first-class concern, not an afterthought.

**Experience replay mixing** (implemented from day one): every gradient batch includes 20% samples from the full buffer history, not just the recent window. This re-anchors all modules to past experience.

**Representation drift monitoring**: a small held-out probe set of (observation, expected-latent-cluster) pairs is fixed at initialization. Every K steps, the encoder’s output on this probe set is checked for drift. Excessive drift triggers a reduced learning rate on the encoder.

**Entropy floor on policy**: the entropy bonus `β` is never annealed below a floor value. The agent is never allowed to become fully deterministic, preserving the ability to explore previously-good regions that were subsequently abandoned.

**Frozen reward circuit as stability anchor**: because the reward circuit is frozen, the definition of “what is good” does not drift with the policy. This is a significant stability property that would be lost if the reward circuit were live-trained.

-----

## Meganeura Confidence Plan

meganeura is a young library (as of this writing, ~42 commits, competitive benchmark on SmolVLA inference). Using it as the training backbone — not just inference — for a complex multi-module system requires building active confidence in its correctness. This is **part of the project**, not a precondition.

The goal is to develop confidence in Rust, without maintaining a PyTorch twin. Numerical cross-checking against PyTorch is a one-time verification tool, not an ongoing maintenance burden.

### Tier 1 — Unit-level gradient verification

For each graph primitive used (linear layers, attention, convolution, layer norm, softmax):

- Implement a finite-difference gradient checker in Rust
- Compare analytical gradients (backprop) to numerical gradients (perturb → forward → diff)
- Tolerance: relative error < 1e-4 for f32
- These tests live in `tests/grad_check.rs` and run in CI

This catches incorrect backprop implementations before they silently corrupt training.

### Tier 2 — E-graph optimization parity

meganeura uses e-graph search to optimize computation graphs before generating GPU kernels. This is clever but opaque — the optimized kernel may not be numerically identical to the unoptimized version.

**Approach**: implement an `OptLevel` flag:

```rust
pub enum OptLevel {
    None,    // pure forward translation, no e-graph passes
    Full,    // default meganeura e-graph optimization
}
```

For every module, run both opt levels on the same inputs and compare outputs:

```
assert_tensors_close(output_none, output_full, atol=1e-5, rtol=1e-4)
```

If outputs diverge beyond tolerance, it is a meganeura bug to be reported and worked around. This test suite lives in `tests/opt_parity.rs`.

The `OptLevel::None` path also serves as a **debugging escape hatch** during development — when a training run produces unexpected behavior, disabling e-graph optimization isolates whether the graph transformation is the cause.

### Tier 3 — Training convergence canaries

Three small, well-understood problems with known convergence behavior are maintained as integration tests:

|Canary                             |Expected behavior                         |Failure signal                       |
|-----------------------------------|------------------------------------------|-------------------------------------|
|XOR classification (MLP, 4 samples)|Loss → 0 in < 500 steps                   |Broken optimizer or backprop         |
|CartPole balance (policy gradient) |Mean episode reward > 195 within 50k steps|Broken policy gradient or reward flow|
|Next-step prediction on random walk|Prediction error < 0.01 within 10k steps  |Broken world model training          |

These run on CPU (no GPU required) and are fast enough for CI. They are the smoke test for the full training pipeline.

### Tier 4 — One-time numerical cross-check against PyTorch

For the most complex module (likely the causal attention in the Credit Assigner), a one-time cross-check against a reference PyTorch implementation will be performed:

- Implement the same attention mechanism in Python/PyTorch with identical weight initialization
- Run one forward + backward pass on identical inputs
- Compare: activations at each layer, gradients at each parameter
- Document the comparison in `docs/pytorch_crosscheck.md`
- **The PyTorch code is then archived and not maintained**

This is a verification artifact, not a development dependency.

### Tier 5 — Long-run stability test

A 1M-step training run on a simple procedurally generated environment, logging:

- Loss curves for all modules
- Gradient norms (watch for explosion or vanishing)
- `H_eff` evolution
- Representation drift on probe set
- GPU memory and throughput

This is run manually at major milestones, not in CI. Results are committed to `docs/stability_runs/`.

-----

## Implementation Phases

### Phase 0 — Foundation (Weeks 1–2)

Goal: a compiling, runnable skeleton with the full data flow wired but untrained modules.

- [ ] Audit meganeura’s existing graph API — identify gaps for causal Transformer
- [ ] Implement `ExperienceBuffer` with ring buffer semantics (`src/buffer.rs`)
- [ ] Define environment trait: `Observation`, `Action`, `HomeostaticProvider`
- [ ] Scaffold all six module structs with placeholder forward passes
- [ ] Implement `OptLevel` flag in meganeura (fork + PR upstream if accepted)
- [ ] Write Tier 1 gradient checkers for all primitives used
- [ ] Implement toy environment: deterministic grid world with homeostatic energy variable

### Phase 1 — World Model & Reward Circuit (Weeks 3–4)

Goal: the agent perceives the world meaningfully and receives coherent reward signals.

- [ ] Implement Encoder (MLP for structured obs; CNN variant behind feature flag)
- [ ] Implement World Model forward + loss
- [ ] Implement Surprise component (reads world model prediction error)
- [ ] Implement Novelty component (grid-hash count-based)
- [ ] Implement Homeostatic component (reads from environment trait)
- [ ] Wire combined reward `r_t` into experience buffer
- [ ] Verify: reward signal is non-zero and varies meaningfully across states
- [ ] Run Canary 3 (next-step prediction)

### Phase 2 — Credit Assigner (Weeks 5–6)

Goal: the agent can attribute reward to past actions, however noisily.

- [ ] Implement causal Transformer (or LSTM fallback) as meganeura graph primitive
- [ ] Implement CreditNet forward pass
- [ ] Implement contrastive training loss
- [ ] Wire credit scores back into experience buffer
- [ ] Track `H_eff` as logged diagnostic
- [ ] Tier 4 one-time cross-check against PyTorch attention reference

### Phase 3 — Policy & Full Loop (Weeks 7–8)

Goal: the agent acts, receives credit-weighted gradients, and improves.

- [ ] Implement Policy network (categorical and Gaussian variants)
- [ ] Implement Value Head + TD(n) loss
- [ ] Implement policy gradient update with entropy bonus
- [ ] Implement value head warmup gate
- [ ] Run Canary 2 (CartPole)
- [ ] Run Tier 2 e-graph parity tests across all modules

### Phase 4 — Continual Learning & Stress Testing (Weeks 9–10)

Goal: the agent runs stably for long periods without forgetting or diverging.

- [ ] Implement replay mixing (20% historical samples per batch)
- [ ] Implement representation drift monitor
- [ ] Implement entropy floor enforcement
- [ ] Run 1M-step stability test (Tier 5)
- [ ] Document findings in `docs/stability_runs/run_001.md`
- [ ] Identify and file issues against meganeura for any gaps found

### Phase 5 — Learnable Reward Circuit (Future)

The frozen reward circuit is revisited here. Prerequisites before unfreezing:

- Credit assigner has demonstrated stable `H_eff` growth over 500k+ steps
- Policy has achieved non-trivial performance on at least one benchmark environment
- Representation drift is < threshold on held-out probe set

Only once these conditions are met is it safe to introduce reward circuit gradient updates without risking destabilizing the entire tri-level system.

-----

## Diagnostics and Observability

The following metrics are logged every K steps to a structured JSON log file (no external dependency required):

|Metric              |Description                    |Healthy range                  |
|--------------------|-------------------------------|-------------------------------|
|`loss_world_model`  |World model prediction MSE     |Decreasing over time           |
|`loss_policy`       |Policy gradient loss           |Noisy; trending down           |
|`loss_value`        |Value TD error                 |Decreasing over time           |
|`loss_credit`       |Credit contrastive loss        |Decreasing, slowly             |
|`reward_mean`       |Mean r_t over window           |Increasing over time           |
|`reward_surprise`   |Surprise component of r_t      |Decreasing as world is learned |
|`reward_novelty`    |Novelty component of r_t       |Decreasing as space is explored|
|`reward_homeo`      |Homeostatic component of r_t   |Near zero when agent is healthy|
|`H_eff`             |Effective credit scope (steps) |Increasing over training       |
|`policy_entropy`    |Policy distribution entropy    |Above floor; not collapsing    |
|`grad_norm_encoder` |Gradient norm on encoder       |Stable; not exploding          |
|`repr_drift`        |Encoder drift on probe set     |Below threshold                |
|`opt_parity_max_err`|Max abs diff between opt levels|< 1e-5                         |

A minimal terminal renderer for these metrics is provided in `tools/monitor.rs`.

-----

## Known Hard Problems

These are not bugs — they are fundamental difficulties that are acknowledged and tracked.

**Circular bootstrapping in credit assignment**: The contrastive training signal for CreditNet assumes the agent has visited similar states with different outcomes. Early in training, before the policy has explored meaningfully, such pairs are rare. Credit assignment is therefore noisy for the first phase of training. This is expected and does not prevent learning — it just slows it.

**The encoder serves too many masters**: Gradients from the world model, policy, and value head all flow into the encoder simultaneously. These objectives can conflict, causing the representation to compromise rather than excel at any one task. A stop-gradient schedule (gradually reducing the policy gradient’s share of encoder gradient) may be needed. This is flagged for Phase 4 tuning.

**Novelty as a long-term strategy**: Count-based novelty rewards decay to near-zero for frequently visited states. In environments where the optimal policy requires repeatedly visiting the same states, the novelty signal actively penalizes good behavior after sufficient exploration. The weight `w_n` should be annealed aggressively once the environment is well-explored. Consider tying the annealing to `H_eff` growth rather than a fixed schedule.

**meganeura e-graph optimization is opaque**: If the e-graph optimizer produces an incorrect but fast kernel, the error may be numerically small per step but compound over millions of training steps into a significant bias. The Tier 2 parity tests catch obvious cases; subtle numerical drift may not be caught until the Tier 5 stability run.

-----

## Open Questions

- Should the causal Transformer in CreditNet use learned positional encodings or fixed sinusoidal? Learned is more flexible; fixed is more stable early in training when the attention weights are noisy.
- What is the right initial weight ratio `w_s : w_n : w_h`? Starting hypothesis: `1.0 : 0.5 : 2.0` (homeostatic signal is the most grounded and should dominate initially).
- Should `H_eff` growth be used as a gating condition for anything, or just monitored? Candidate use: gate policy learning rate increase on `H_eff > threshold`.
- How should the agent handle environments that have no homeostatic signals? Fallback: `r_homeo = 0`, `w_h = 0`. The agent survives on surprise and novelty alone.

-----

## Repository Structure

```
IRIS/
├── src/
│   ├── lib.rs              # Public API surface
│   ├── agent.rs            # Top-level Agent struct and training loop
│   ├── buffer.rs           # ExperienceBuffer, RingBuffer
│   ├── encoder.rs          # Encoder graph definition
│   ├── world_model.rs      # World Model graph + loss
│   ├── reward.rs           # Frozen Reward Circuit (surprise, novelty, homeo)
│   ├── credit.rs           # CreditNet + contrastive loss
│   ├── policy.rs           # Policy + Value Head
│   └── env.rs              # Environment traits (Observation, Action, HomeostaticProvider)
├── tests/
│   ├── grad_check.rs       # Tier 1: finite-difference gradient verification
│   ├── opt_parity.rs       # Tier 2: e-graph optimization parity tests
│   └── canaries.rs         # Tier 3: XOR, CartPole, random-walk convergence
├── examples/
│   └── grid_world.rs       # Toy environment for early development
├── tools/
│   └── monitor.rs          # Terminal diagnostic renderer
├── docs/
│   ├── pytorch_crosscheck.md   # Tier 4: one-time attention cross-check record
│   └── stability_runs/         # Tier 5: long-run training logs
├── Cargo.toml
└── README.md
```

-----

## Dependencies

|Crate                 |Role                                  |
|----------------------|--------------------------------------|
|`meganeura`           |Neural network training and inference |
|`blade-graphics`      |GPU backend (Vulkan / Metal)          |
|`rand`                |Sampling, exploration noise           |
|`serde` / `serde_json`|Diagnostic logging                    |
|`hashbrown`           |Fast hash map for novelty visit counts|

No Python. No PyTorch. No runtime dependencies outside the Rust ecosystem.
