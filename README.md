# Kindle

> To kindle is to start a fire from nothing. This agent does the same with intelligence — a cold network that bootstraps its own understanding from environment-agnostic primitives, with no pretraining and no handed-down reward.

[![CI](https://github.com/kvark/kindle/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/kindle/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/kindle.svg)](https://crates.io/crates/kindle)
[![docs.rs](https://img.shields.io/docsrs/kindle)](https://docs.rs/kindle)
[![license](https://img.shields.io/crates/l/kindle.svg)](LICENSE)

Built on [meganeura](https://github.com/kvark/meganeura) — a cross-platform Rust neural network library with GPU-accelerated training and inference via [blade-graphics](https://github.com/kvark/blade).

-----

## Table of Contents

- [Vision](#vision)
- [Architecture Overview](#architecture-overview)
- [Modules](#modules)
- [Reward Circuit (Frozen)](#reward-circuit-frozen)
- [Temporal Buffer](#temporal-buffer)
- [Tri-Level Learning Loop](#tri-level-learning-loop)
- [Continual Learning Strategy](#continual-learning-strategy)
- [Meganeura Confidence Plan](#meganeura-confidence-plan)
- [Milestones](#milestones)
- [Diagnostics and Observability](#diagnostics-and-observability)
- [Python Bindings](#python-bindings)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)

-----

## Vision

Three beats.

**Cold start.** The network is initialised Xavier-uniform and sees its first observation with no prior. There is no pretraining, no offline dataset, no demonstration corpus. Intelligence has to ignite from contact with the environment.

**Self-grounded reward.** Reward is not handed down. It is derived from four primitives the agent can evaluate without external supervision: *surprise* (did the world do what I expected?), *novelty* (have I been here before?), *homeostatic balance* (am I in a healthy range?), and *order* (am I making structure?). The circuit that combines these is **frozen** — its definition of "good" does not drift with the policy.

**Emergent order.** Exploration and consolidation are duals. Novelty rewards the agent for venturing into unseen phase space; order rewards it for concentrating its recent trajectory into a smaller subset of the digested observation space than its historical average. A healthy agent oscillates between them on its own.

Stability and correctness come first, performance second. This is a research and engineering project.

-----

## Architecture Overview

```
 Raw Observation (o_t)
        │
   ┌────▼────────────────┐
   │   Env Adapter       │   obs_dim → OBS_TOKEN_DIM (universal)
   └────┬────────────────┘
        │  obs_token
   ┌────▼────────────────┐
   │   Encoder E         │──────────────────────────────────┐
   └────┬────────────────┘                                   │
        │  latent z_t                                        │
   ┌────▼─────────────────────────────────────────────┐      │
   │  World Model  W(z_t, a_t) → ẑ_{t+1}             │      │
   │  Loss: || ẑ_{t+1} − stop_grad(z_{t+1}) ||       │      │
   └────┬─────────────────────────────────────────────┘      │
        │                                                    │
   ┌────▼─────────────────────────────────────────────┐      │
   │  Reward Circuit  (FROZEN)                        │      │
   │  Four primitives → r_t                           │      │
   │    · surprise  (world model prediction err)      │      │
   │    · novelty   (inverse-sqrt visit count)        │      │
   │    · homeostatic (env-defined signals)           │      │
   │    · order     (causal entropy reduction)        │      │
   └────┬─────────────────────────────────────────────┘      │
        │  r_t                                               │
   ┌────▼─────────────────────────────────────────────┐      │
   │  Credit Assigner  C(r_t, history[t-H:t])         │      │
   │  Causal attention → α_i per past timestep        │      │
   │  credit_i = r_t × α_i                            │      │
   └────┬─────────────────────────────────────────────┘      │
        │  credit[t-H:t]                                     │
   ┌────▼─────────────────────────────────────────────┐      │
   │  Value Head  V(z_t) → V̂                         │◄─────┘
   └────┬─────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────┐
   │  Policy  π(z_t) → action distribution            │
   └────┬─────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────┐
   │   Env Adapter       │   MAX_ACTION_DIM → env-native action
   └──────────────────────┘
```

All modules except the reward circuit train continuously through experience. The adapter layer keeps the core graph's tensor shapes universal (`OBS_TOKEN_DIM`, `MAX_ACTION_DIM`) so the agent can hop between environments at runtime without recompiling any GPU graph. See [docs/universal-actions.md](docs/universal-actions.md) for the cross-environment design.

-----

## Modules

### Encoder

Converts observation tokens into a compact latent `z_t`. Shared backbone — all other modules consume `z_t`, not raw observations. The encoder is forced to learn representations that are simultaneously useful for world modelling, credit attribution, and policy generation. Gradient flow: world model loss (primary), policy gradient (secondary), value TD error (secondary).

### World Model

Forward dynamics predictor `W(z_t, a_t) → ẑ_{t+1}` with MSE loss against `stop_grad(z_{t+1})`. The stop-gradient prevents encoder collapse (BYOL / Dreamer v3 trick). Serves two roles: self-supervised training signal, and surprise-component input to the reward circuit.

### Credit Assigner

Answers *which past actions caused this reward?* via causal self-attention over the recent history buffer:

```
CreditNet(r_t, [(z_{t-H}, a_{t-H}), …, (z_t, a_t)]) → α ∈ R^H
credit_i = r_t × α_i   (α softmaxed; attention is strictly causal)
```

Trained contrastively: when similar latent states at different times receive different rewards, CreditNet should attribute the difference to the diverging action sequences. The diagnostic `H_eff = Σ_i (i × α_i)` tracks effective temporal scope — growing `H_eff` means the agent is learning longer-horizon consequences.

### Value Head + Policy

Small MLP `V̂(z_t)` (actor-critic baseline) and stochastic policy `π(z_t)`. Both branches use the same universal continuous-Gaussian graph — for discrete envs the adapter interprets the first `n` head dims as logits and samples categorically. Updated by credit-weighted policy gradient with an entropy bonus that never anneals below a floor (preserving exploration forever).

-----

## Reward Circuit (Frozen)

Four primitives, one frozen weighted sum:

```
r_t = w_s·r_surprise + w_n·r_novelty + w_h·r_homeo + w_o·r_order
```

The circuit receives no gradient updates. Freezing it is a stability claim: because *what is good* doesn't drift with the policy, the whole tri-level system is less prone to runaway feedback during early training. Unfreezing is explicitly scoped to a post-stability milestone.

### Exploration vs consolidation — novelty and order as duals

| Primitive | Rewards | Decays when | Role |
|---|---|---|---|
| **surprise** `‖ẑ−z‖` | world behaves unexpectedly | world model converges | shapes z to be predictable |
| **novelty** `1/√N(z)` | unvisited latent regions | region is revisited | exploration pressure |
| **homeostatic** `−Σ deviations` | staying in env-defined targets | agent is alive and happy | survival pressure |
| **order** `H_ref − H_recent` | concentrating recent trajectory vs historical baseline | recent converges to reference | consolidation pressure |

Novelty and order are the two ends of the exploration/consolidation axis. Novelty rewards entropy in the *visited-state* distribution (go somewhere new); order rewards negentropy in the *recent-observation* distribution (make the place you're in legible). A healthy agent trades off between them, and the weights anneal accordingly: `w_n` decays as phase space is covered; `w_o` can remain on indefinitely because its baseline moves with the agent.

### The order primitive

At agent construction the circuit samples a frozen random linear digest `φ: R^{OBS_TOKEN_DIM} → R^{d}` (small matrix, `d = 4`) and fixed per-dim bucket edges. φ never trains. Each step:

1. `bucket_id = quantize(tanh(φ · obs_token_t))`
2. Push onto two ring buffers — `recent` (W = 64) and `reference` (W_ref = 512).
3. `H_recent`, `H_reference` = Shannon entropies of the bucket histograms.
4. `r_order = H_reference − H_recent` (zero during reference warmup).

Why this formulation:

- **Grounded.** φ and the bucket grid are fixed at init, preserving the frozen-circuit invariant.
- **Causal.** The reference is the agent's *own past* — every step's signal depends only on the agent's recent actions vs its own longer-term behaviour.
- **Environment-agnostic.** Operates on adapter-normalized obs tokens, not env-specific feature shapes.
- **Ungameable by latent collapse.** The digest reads the observation token, not the encoder latent, so the encoder cannot cheaply maximise order by shrinking its representation.

The old `−H(|o_i| / Σ|o_j|)` definition is gone — it was trivially maxed by any one-hot observation and mostly measured the env's encoding scheme, not the agent's behaviour.

-----

## Temporal Buffer

A single shared circular buffer is the agent's sole persistent memory:

```rust
pub struct ExperienceBuffer {
    observations: RingBuffer<Vec<f32>>,   // obs_token
    latents:      RingBuffer<Vec<f32>>,   // z_t
    actions:      RingBuffer<Vec<f32>>,   // action_token
    rewards:      RingBuffer<f32>,
    credits:      RingBuffer<f32>,
    pred_errors:  RingBuffer<f32>,
    env_ids:      RingBuffer<u32>,        // which env produced this transition
    env_boundary: RingBuffer<bool>,       // true on the first step after switch_env
    visit_counts: HashMap<StateKey, u32>, // for novelty
}
```

No episodic boundary — experience accumulates continuously. Cross-env hops are tagged so the credit assigner and world-model replay don't try to attribute dynamics or reward across a switch. **Replay mixing**: ~20% of steps are shadowed by a gradient update on a random historical sample, which re-anchors the encoder against catastrophic forgetting without needing EWC machinery.

-----

## Tri-Level Learning Loop

| Module                | LR                     | Frequency    | Notes                                     |
|-----------------------|------------------------|--------------|-------------------------------------------|
| Encoder + World Model | `lr_wm` (base)         | every step   | self-supervised; most stable               |
| Credit Assigner       | `lr_credit` (0.3× base)| every step   | slower; contrastive signal is noisy early  |
| Policy + Value        | `lr_policy` (0.5× base)| every step   | gated on warmup + entropy floor            |

Policy updates are suppressed for the first `N_warmup` steps (so the value baseline can stabilise) and whenever policy entropy falls below a floor (so the agent can always explore).

-----

## Continual Learning Strategy

- **Replay mixing** — 20% shadow gradient steps on random historical samples, active from day one.
- **Representation drift monitor** — a held-out probe set of observations is captured at warmup completion; every `drift_interval` steps the encoder's output on the probe is compared against the stored reference. Excessive drift auto-scales the encoder LR down; low drift lets it recover.
- **Entropy floor** — the policy entropy bonus is never annealed below a floor. The agent never becomes fully deterministic.
- **Frozen reward circuit** — the definition of *good* does not drift.

-----

## Meganeura Confidence Plan

Meganeura is young. Using it as the training backbone for a complex multi-module system requires building active confidence in its correctness. This is part of the project, not a precondition.

### Tier 1 — Unit-level gradient verification

Finite-difference gradient checks on every graph primitive used (linear, attention, layer norm, softmax). Tolerance: relative error < 1e-4 for f32. Lives in `kindle/tests/grad_check.rs` (planned) — currently covered indirectly by the convergence canaries.

### Tier 2 — E-graph optimization parity

`OptLevel::{None, Full}` flag isolates meganeura's e-graph search. Every module is tested with both levels and outputs compared within tight tolerance. Tests in `kindle/tests/opt_parity.rs`. The `None` path also serves as a debugging escape hatch during training-time anomalies.

### Tier 3 — Training convergence canaries

| Canary                              | Expected                                    | Failure signal                          |
|-------------------------------------|---------------------------------------------|-----------------------------------------|
| XOR classification (MLP, 4 samples) | Loss → 0 in < 500 steps                     | Broken optimizer or backprop            |
| Next-step prediction on random walk | Prediction error < 0.01 within 10k steps    | Broken world model training             |
| Full kindle encoder + world model   | Loss decreases by ≥ 2× on deterministic env | Broken graph fusion or encoder gradient |

These live in `kindle/tests/canaries.rs`. GPU-gated tests run via `cargo test -- --ignored`.

### Tier 4 — PyTorch cross-check (archived)

A one-time numerical cross-check of the causal-attention credit assigner against a PyTorch reference was performed during Phase 2 and archived. The PyTorch code is not maintained; it was a verification artifact, not a development dependency.

### Tier 5 — Long-run stability

A long training run (target: 1M steps on a single env, multi-env variant to follow) logs loss curves, gradient norms, `H_eff`, drift, throughput. Re-run at each milestone; results committed to `docs/stability_runs/`.

-----

## Milestones

Outcome gates, not calendar weeks. Each gate has a measurable exit condition; we move on when the gate closes.

- **M1 — World model ignites.** `loss_world_model` decreases monotonically on random-walk and GridWorld; reaches < 0.01 prediction error within 10k steps. **Status: closed.**
- **M2 — Reward signal carries information.** All four primitives produce non-zero, finite values that vary across states; `reward_mean` separates trajectories that lead to homeostatic violations from those that don't. **Status: closed.**
- **M3 — Policy learns on its own signal.** Kindle agent solves CartPole from a cold start using intrinsic reward only (mean episode length > 195 within 50k steps). **Status: open.**
- **M4 — Cross-env generalisation.** Agent hops GridWorld → CartPole → MountainCar → Taxi → Acrobot → Pendulum without divergence; encoder representations remain meaningful after switches (drift < threshold). **Status: partial — no divergence, generalisation TBD.**
- **M5 — Long-run stability.** 1M-step run on real GPU hardware with no NaN, no gradient explosion, `H_eff` trending upward, entropy above floor. **Status: open.**
- **M6 — Learnable reward circuit (unfreeze).** Only attempted once M3–M5 are closed. Adds gradient updates to the reward weights under tight constraints documented at the time.

-----

## Diagnostics and Observability

Structured JSON diagnostics every step (no external dependency required):

| Metric              | What                                      | Healthy range                     |
|---------------------|-------------------------------------------|-----------------------------------|
| `loss_world_model`  | World model prediction MSE                | Decreasing over time              |
| `loss_policy`       | Policy gradient loss                      | Noisy; trending down              |
| `loss_credit`       | Credit contrastive loss                   | Slowly decreasing                 |
| `loss_replay`       | MSE on replay batches                     | Within an order of `loss_wm`      |
| `reward_mean`       | Mean r_t over window                      | Increasing over time              |
| `reward_surprise`   | Surprise component                        | Decreasing as world is learned    |
| `reward_novelty`    | Novelty component                         | Decreasing as space is explored   |
| `reward_homeo`      | Homeostatic component                     | Near zero when agent is healthy   |
| `reward_order`      | Order component (`H_ref − H_recent`)      | Positive when concentrating       |
| `h_eff`             | Effective credit scope (steps)            | Increasing over training          |
| `policy_entropy`    | Policy distribution entropy               | Above floor; not collapsing       |
| `repr_drift`        | Encoder drift on probe set                | Below threshold                   |
| `buffer_len`        | Experience buffer length                  | Up to `buffer_capacity`           |

-----

## Python Bindings

Kindle ships an optional pyo3 extension under `python/`, built via [maturin](https://www.maturin.rs/):

```bash
pip install maturin
cd python && maturin develop --release
```

Then in Python:

```python
import gymnasium as gym
import kindle

env = gym.make("CartPole-v1")
agent = kindle.Agent(obs_dim=4, num_actions=2, env_id=0, seed=0)

# A tiny adapter that returns obs as a plain list of floats.
class Wrapped:
    def __init__(self, env): self.env = env
    def reset(self):
        obs, _ = self.env.reset()
        return [float(x) for x in obs]
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(int(action) % 2)
        if term or trunc:
            obs, _ = self.env.reset()
        return [float(x) for x in obs], float(r), bool(term), bool(trunc), info

agent.train(Wrapped(env), steps=10_000)
print(agent.diagnostics())
```

The Python extension is a thin wrapper — training still runs on GPU through meganeura. Python is for scripting and for dropping kindle into a gymnasium loop; it is not a second implementation.

-----

## Repository Structure

This is a Cargo workspace.

```
kindle/
├── Cargo.toml                 # [workspace]
├── kindle/                    # core library crate (publishable)
│   ├── Cargo.toml
│   ├── src/                   # agent, encoder, world_model, reward, credit, policy, …
│   └── tests/                 # canaries, opt_parity, reward_signal
├── kindle-gym/                # Rust mini-gym + runnable examples
│   ├── Cargo.toml
│   ├── src/                   # grid_world, cart_pole, acrobot, …
│   ├── examples/              # cargo run -p kindle-gym --example cart_pole
│   └── tests/                 # stability (GPU-gated)
├── python/                    # pyo3 cdylib (maturin; excluded from workspace)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   ├── src/lib.rs
│   ├── kindle/__init__.py
│   └── tests/
├── docs/
│   ├── universal-actions.md
│   └── stability_runs/
└── .github/workflows/ci.yml
```

-----

## Dependencies

| Crate                  | Role                                        |
|------------------------|---------------------------------------------|
| `meganeura`            | Neural network training and inference       |
| `blade-graphics`       | GPU backend (Vulkan / Metal)                |
| `rand`                 | Sampling, exploration noise                 |
| `serde` / `serde_json` | Diagnostic logging                          |
| `hashbrown`            | Fast hash map for novelty visit counts      |
| `pyo3` *(python only)* | Optional Python bindings                    |

The core Rust crate has no Python dependency. The `python/` crate is an optional extension module.
