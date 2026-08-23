# Kindle

> To kindle is to start a fire from nothing. This agent does the same with intelligence — a cold network that bootstraps its own understanding from environment-agnostic primitives, with no pretraining and no handed-down reward.

[![CI](https://github.com/kvark/kindle/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/kindle/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Not on crates.io: kindle depends on git-pinned revisions of meganeura, so it is consumed as a git dependency for now.

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
- [Learning Benchmark](#learning-benchmark)
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
        │  r_t / n-step return / GAE / GRPO advantage        │
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

Enabled learning modules train continuously through experience. The adapter
layer keeps token sizes fixed: observations use `OBS_TOKEN_DIM`, the discrete
policy uses `MAX_ACTION_DIM = 18`, and dynamics use `WM_ACTION_DIM = 20`
(`18`-way action identity plus two optional normalized parameters). Adapters
with compatible observation encoders and action kinds can switch without
recompiling a GPU graph. Visual shape and discrete/continuous policy family are
still fixed at construction. See
[docs/universal-actions.md](docs/universal-actions.md).

-----

## Modules

### Encoder

Converts observation tokens or frames into a compact latent `z_t`. Most downstream modules consume this latent. Because the world-model target is stop-gradient, prediction loss alone does not train the encoder; enable reconstruction (or load and explicitly freeze a pretrained encoder) so the representation is not left at random initialization.

### World Model

Forward dynamics predictor `W(z_t, a_t) → ẑ_{t+1}` with MSE loss against `stop_grad(z_{t+1})`. The stop-gradient prevents encoder collapse (BYOL / Dreamer v3 trick). Serves two roles: self-supervised training signal, and surprise-component input to the reward circuit.

### Value Head + Policy

Small MLP `V̂(z_t)` (actor-critic baseline) and stochastic policy `π(z_t)`. The policy graph is picked at construction from the adapters' action kinds: all-discrete setups compile a categorical graph and all-continuous setups compile a Gaussian-mean graph. Mixing discrete and continuous adapters in one agent is unsupported. Policy updates use n-step returns, optional GAE/value bootstrapping, PPO, or within-task GRPO, with an entropy bonus.

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
    pred_errors:  RingBuffer<f32>,
    env_ids:      RingBuffer<u32>,        // which env produced this transition
    env_boundary: RingBuffer<bool>,       // true on the first step after switch_lane
    visit_counts: HashMap<StateKey, u32>, // for novelty
}
```

Experience accumulates continuously, with episode/environment boundaries explicitly tagged so world-model replay and return estimates do not cross resets. **Replay mixing**: optional shadow gradient updates sample historical transitions.

-----

## Core Learning Loop

| Module                | LR                     | Frequency    | Notes                                     |
|-----------------------|------------------------|--------------|-------------------------------------------|
| Encoder + World Model | `lr_wm` (base)         | every step   | self-supervised; most stable               |
| Policy + Value        | `lr_policy` (0.5× base)| configured cadence | warmup gate; low-entropy recovery     |

Policy updates are suppressed for the first `N_warmup` steps so the value
baseline can stabilise. Below the configured entropy floor, recovery label
smoothing shifts the target toward valid actions instead of freezing policy
updates at a collapsed distribution.

-----

## Continual Learning Strategy

- **Replay mixing** — 20% shadow gradient steps on random historical samples, active from day one.
- **Representation drift monitor** — a held-out probe set of observations is captured at warmup completion; every `drift_interval` steps the encoder's output on the probe is compared against the stored reference. Excessive drift auto-scales the encoder LR down; low drift lets it recover.
- **Entropy floor** — low entropy activates valid-action label smoothing and a
  recovery advantage. Deterministic policy-mode actions are available only as
  an explicit evaluation choice.
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

### Tier 5 — Long-run stability

A long training run (target: 1M steps on a single env, multi-env variant to follow) logs loss curves, gradient norms, drift, and throughput. Re-run at each milestone; results committed to `docs/stability_runs/`.

-----

## Milestones

Outcome gates, not calendar weeks. Each gate has a measurable exit condition; we move on when the gate closes.

- **M1 — World model ignites.** `loss_world_model` decreases monotonically on random-walk and GridWorld; reaches < 0.01 prediction error within 10k steps. **Status: closed.**
- **M2 — Reward signal carries information.** All four primitives produce non-zero, finite values that vary across states; `reward_mean` separates trajectories that lead to homeostatic violations from those that don't. **Status: closed.**
- **M3-E — Policy learns from task reward.** The seeded CartPole gates reach final recent mean returns of 263.38 in the native Rust environment and 264.75 through Gymnasium within 800k environment steps, using episode-GRPO and self-imitation. **Status: closed.**
- **M3 — Policy learns on its own signal.** Kindle solves CartPole from a cold start using only the generic homeostatic reward primitive. After exactly 50k training decisions, frozen greedy returns are 500/500/215 across seeds 42–44 (matched random: 22.11/22.74/22.42), so every seed exceeds 195. **Status: closed.**
- **M4 — Cross-env generalisation.** Agent hops GridWorld → CartPole → MountainCar → Taxi → Acrobot → Pendulum without divergence; encoder representations remain meaningful after switches (drift < threshold). **Status: partial — no divergence, generalisation TBD.**
- **M5 — Long-run stability.** 1M-step run on real GPU hardware with no NaN, no gradient explosion, and entropy above floor. **Status: closed — passes on GridWorld** (see [docs/next-steps.md](docs/next-steps.md) and `docs/stability_runs/`); a multi-env variant remains future work.
- **M6 — Learnable reward circuit (unfreeze).** A learned outcome-value head `R̂` over episode trajectories augments the frozen four-primitive circuit. **Status: implemented** — `kindle/src/outcome.rs`, design in [docs/phase-m6-learnable-reward.md](docs/phase-m6-learnable-reward.md).
- **M7 — Approach-reward primitive.** Approach-shaping reward from self-discovered prototypes, addressing the reward class M6 cannot express. **Status: implemented** — `kindle/src/approach.rs`, design in [docs/phase-m7-approach-reward.md](docs/phase-m7-approach-reward.md).

-----

## Diagnostics and Observability

Structured JSON diagnostics every step (no external dependency required):

| Metric              | What                                      | Healthy range                     |
|---------------------|-------------------------------------------|-----------------------------------|
| `loss_world_model`  | World model prediction MSE                | Decreasing over time              |
| `loss_reconstruction` | Unweighted encoder reconstruction MSE  | Finite; task-dependent            |
| `loss_policy`       | Policy gradient loss                      | Noisy; trending down              |
| `loss_replay`       | MSE on replay batches                     | Within an order of `loss_wm`      |
| `reward_mean`       | Mean r_t over window                      | Increasing over time              |
| `reward_surprise`   | Surprise component                        | Decreasing as world is learned    |
| `reward_novelty`    | Novelty component                         | Decreasing as space is explored   |
| `reward_homeo`      | Homeostatic component                     | Near zero when agent is healthy   |
| `reward_order`      | Order component (`H_ref − H_recent`)      | Positive when concentrating       |
| `policy_entropy`    | Policy distribution entropy               | Above floor; not collapsing       |
| `repr_drift`        | Encoder drift on probe set                | Below threshold                   |
| `buffer_len`        | Experience buffer length                  | Up to `buffer_capacity`           |

-----

## Learning Benchmark

The intrinsic-only CartPole gate closes M3 without consuming environment
reward or task-success events. It uses the environment-provided pole-angle
homeostatic variable, the frozen generic deviation penalty, per-step GRPO, and
no SIL. Setting `--extrinsic-alpha 0` also disables CartPole's score-triggered
learning-rate schedule, so the reported task return cannot affect training.
Exactly 6,250 vector steps across eight lanes provide 50,000 training
decisions; learning is then frozen for 160,000 greedy evaluation decisions per
seed:

```bash
cargo run --release -p kindle-gym --example native_learning -- \
  --env cartpole --steps 26250 --evaluation-steps 20000 \
  --seeds 42,43,44 --recent-window 200 --log-every 6250 \
  --extrinsic-alpha 0 --reward-homeostatic 2 \
  --per-step-grpo --no-sil
```

Frozen returns are 500, 500, and 215 for exploration seeds 42–44; uniform
random actions score 22.11, 22.74, and 22.42. Every learned seed passes the
conventional 195 threshold. This is genuine intrinsic-only optimization under
Kindle's intended homeostatic interface, although the environment still
declares which variable and target constitute health; it is not autonomous
goal discovery.

The matched random evaluation is:

```bash
cargo run --release -p kindle-gym --example native_learning -- \
  --env cartpole --steps 20001 --evaluation-steps 20000 \
  --seeds 42,43,44 --recent-window 200 --log-every 0 \
  --extrinsic-alpha 0 --reward-homeostatic 2 \
  --random-policy --no-gate
```

One shared task-conditioned policy now passes CartPole and Taxi
simultaneously. Eight lanes of each game train together; orthogonal task codes,
dynamic action masks, same-task GRPO, per-task SIL admission baselines, and
task-balanced SIL sampling keep reward/action semantics separated. The last
20k vector steps freeze all learning and evaluate the sampled policy:

```bash
cargo run --release -p kindle-gym --example native_multitask -- \
  --seeds 42,43,44
```

Across the three fresh agents, frozen CartPole returns are
485.40/482.37/402.43 and Taxi completion is 92.3%/95.6%/90.3%; every task on
every seed passes its 195-return/90%-completion gate. The 100k joint training
steps correspond to 800k decisions per task. A current 16/32 seed-42 ablation
lacks shared capacity (Taxi 78.7%), while the retained 32/64 model passes.
Greedy evaluation is available with `--greedy-eval`; on seed 42 it keeps
CartPole at 500 but Taxi reaches only 87.1%, so the executable gate
intentionally evaluates the frozen stochastic policy Kindle trained.

The same executable also tests continual retention. It trains all eight lanes
on CartPole, freezes a pre-switch evaluation, switches the same agent to Taxi,
then returns to CartPole with learning still frozen:

```bash
cargo run --release -p kindle-gym --example native_multitask -- \
  --sequential --seeds 42,43,44
```

Pre-switch CartPole returns are 488.18/493.80/486.31, Taxi completion after
the switch is 93.9%/92.8%/93.3%, and retained CartPole returns are
253.50/500.00/229.98; every phase passes on every seed. The reverse
Taxi→CartPole order passes two of three seeds: seed 43 retains Taxi at 89.6%,
just under gate. This establishes one reliable two-task curriculum, not
order-independent continual learning or positive transfer.

The extrinsic CartPole gate checks task learning rather than mechanical
correctness. It starts from cold weights, uses eight forks of one task,
whole-episode GRPO plus self-imitation, and fails with exit code 2 unless
the final recent return reaches the conventional 195 threshold. The native
Rust gate is the shortest reproducible path:

```bash
cargo run --release -p kindle-gym --example native_learning
```

The seeded run reaches a final recent mean return of 263.38 after 800k
environment steps. The same harness accepts `--env mountain-car`,
`--env acrobot`, `--env taxi`, and continuous `--env pendulum`, with return and
task-completion gates that can be selected independently.

Taxi benefits from its native legal-action mask, a small intrinsic
goal-distance signal, and event-filtered self-imitation. Event-filtered SIL
compares successful episodes against other successes and retains the whole
trajectory even when negative step costs make its early return-to-go negative.
Across seeds 42–44, the final 200-episode task-completion rates are 90%, 95%,
and 93% after 800k environment steps. This closes the reliable-completion
gate; negative mean returns show that route efficiency remains suboptimal:

```bash
cargo run --release -p kindle-gym --example native_learning -- \
  --env taxi --steps 100000 --seed 42 --recent-window 25 \
  --reward-homeostatic 0.5 --per-step-grpo --action-masks \
  --sil-event-filter --sil-loss-coef 2 \
  --require-recent-success-rate 0.9
```

MountainCar exposes why task events must be separate from reward and episode
termination: reaching the goal is a success even though that step returns -1,
while other environments such as CartPole can terminate on failure. The native
transition contract now carries an explicit task-event bit. With no intrinsic
reward, repeated actions provide temporally coherent cold-start exploration;
after the first real success, the repeat drops from eight to four and
event-filtered SIL retains the momentum-building trajectory. Across seeds
42–44, final 200-episode completion is 97.5%, 91.0%, and 99.5% after 400k
environment steps. Mean returns (-142.67, -151.24, and -134.70) show that this
closes reliable completion, not reward-optimal speed:

```bash
cargo run --release -p kindle-gym --example native_learning -- \
  --env mountain-car --steps 50000 --seed 42 --recent-window 25 \
  --extrinsic-alpha 0.001 --no-grpo --no-advantage-normalize \
  --action-repeat 8 --action-repeat-after-success 4 \
  --sil-event-filter --sil-loss-coef 4 --sil-buffer-capacity 50000 \
  --require-recent-success-rate 0.9
```

Acrobot uses the same event-replay mechanism with a delayed persistence switch.
An effectively reward-free policy completes only 1.9% of episodes with
per-step sampling; repeat four raises the calibrated temporal-exploration
baseline to 44.5%. The learner collects 50 successful repeat-four episodes
before switching to repeat two for finer control. Across seeds 42–44, final
200-episode completion is 95.0%, 93.0%, and 97.5% after 240k environment steps.
Mean returns (-311.24, -315.95, and -258.86) again show reliable but inefficient
completion:

```bash
cargo run --release -p kindle-gym --example native_learning -- \
  --env acrobot --steps 30000 --seed 42 --recent-window 25 \
  --extrinsic-alpha 0.001 --no-grpo --no-advantage-normalize \
  --action-repeat 4 --action-repeat-after-success 2 \
  --action-repeat-switch-successes 50 \
  --sil-event-filter --sil-loss-coef 16 --sil-buffer-capacity 100000 \
  --require-recent-success-rate 0.9
```

Pendulum uses Kindle's continuous Gaussian policy rather than a categorical
action approximation. Its policy loss now weights each row's squared action
error with an explicit signed advantage; the old target-scaling shortcut was
mathematically wrong for regression. Only the adapter's live action dimensions
contribute to the loss; Pendulum's 17 padded universal outputs are ignored.
With both corrections, reward-only Kindle scores -1860.34 versus the matched
random return of -1724.57 and therefore shows no autonomous progress.

The harness also exposes a clearly labeled online expert-correction (DAgger)
mode: Kindle visits states under its own noisy policy, a deterministic swing-up
controller labels corrective actions, and the last 2,000 steps freeze learning
for greedy evaluation. At 816k environment steps it reaches -359.02, -497.27,
and -362.44 across seeds 42, 7, and 123, so two of three seeds pass the -400
success threshold. This is retained imitation, not an autonomous reward-only
solve:

```bash
cargo run --release -p kindle-gym --example native_learning -- \
  --env pendulum --steps 102000 --evaluation-steps 2000 --seed 42 \
  --pendulum-dagger --require-recent-success-rate 0.9
```

The Gymnasium parity gate is:

```bash
python python/examples/gym_extrinsic.py \
  --env CartPole-v1 --lanes 8 --steps 100000 --seed 42 --log-every 5000 \
  --end-to-end-encoder --rollout-length 8 --advantage-normalize \
  --use-grpo --use-grpo-episode --use-sil --sil-loss-coef 1 \
  --value-loss-coef 0 --entropy-floor 0 \
  --lr-policy 0.0005 --policy-head-lr-mul 5 \
  --lr-drop-on-solve 10 --solve-threshold 195 \
  --require-recent-return 195
```

This is intentionally separate from the intrinsic-only milestone: it proves
that the policy can consume a valid task reward before harder exploration and
representation problems are blamed on the optimizer.

The seeded Atari progress gate uses the spatial Nature-DQN encoder, explicit
zero-intrinsic reward defaults, same-task GRPO, and terminal-frame-correct
Gymnasium autoreset handling. It demonstrates learning on Breakout, not game
completion: at 120k environment steps the final recent return is 1.94 over 319
episodes, versus a seeded random-policy baseline of 1.225 (gate: 1.5).

The random baseline is reproducible with the same harness:

```bash
python python/examples/atari_batch.py \
  --random-policy --game Breakout --lanes 64 --steps 1250 \
  --encoder cnn --recon-loss-coef 0 --log-every 0
```

```bash
python python/examples/atari_batch.py \
  --preset extrinsic_grpo --game Breakout --lanes 64 --steps 1875 \
  --lr 3.5e-5 --lr-policy 3.5e-5 --log-every 375 \
  --require-min-episodes 80 --require-recent-return 1.5 \
  --require-latent-std 0.005
```

Autonomous Pong remains open: the corresponding 200k-step reward-learning
probe ended at -20.43, indistinguishable from/slightly below the seeded random
baseline of -20.31. A separate, explicitly supervised controller-distillation
gate does retain useful play. It collects state-balanced labels from the
pixel-only controller, trains Kindle's masked policy directly, and evaluates a
freshly loaded checkpoint without further updates or controller access:

```bash
python python/examples/atari_pong_distill.py \
  --weights-dir /tmp/kindle-pong-policy
python python/examples/atari_pong_distill.py \
  --load-weights-dir /tmp/kindle-pong-policy
```

The seeded run reaches 100% accuracy on 768 training states and 98.8% on 256
held-out states after 1,175 epochs. Its zero-update reload scores -12.60 over
five games/120k decisions, versus -20.03 for matched random play and -11.50 for
the source controller. That checkpoint can then continue with raw Pong rewards
only; the controller and dataset are not used during online play:

```bash
python python/examples/atari_pong_finetune.py \
  --source-weights-dir /tmp/kindle-pong-policy \
  --weights-dir /tmp/kindle-pong-finetuned
python python/examples/atari_pong_finetune.py \
  --load-weights-dir /tmp/kindle-pong-finetuned
```

The continuation freezes the supervised motion encoder and applies same-task
GRPO to the policy head for 120k decisions across 24 lanes. On three matched
120k-decision evaluation seeds, a separate zero-update reload reduces the
source policy's point deficit from 368 to 237 (35.6%), improves the deficit on
every seed, raises the positive-point fraction from 27.2% to 32.9%, retains
86.0% of the source event count, and moves aggregate completed-game return from
-14.06 to -13.43.
The encoder's measured maximum parameter change is exactly zero while the
policy head changes. This establishes reward-driven improvement from a
supervised initialization, not reward-only learning from scratch or a Pong
solve. The distinction is intentional—an Atari progress gate is not an Atari
completion claim.

A separate task-specific shaping gate does start from random policy weights. It
turns agreement with the motion token's ball-intercept direction into a
class-balanced scalar consistency reward, while leaving raw signed Pong points
active:

```bash
python python/examples/atari_pong_intrinsic.py \
  --weights-dir /tmp/kindle-pong-intrinsic
python python/examples/atari_pong_intrinsic.py \
  --load-weights-dir /tmp/kindle-pong-intrinsic
```

No controller action is executed and no label is sent through Kindle's
supervised API; same-task GRPO learns only from the scalar rewards. After 320k
training decisions, the best fixed-validation checkpoint (step 2,600) is
restored. A separate zero-update reload then scores -13.20/-13.60/-10.75 on
three 120k-decision seeds, with at least 98.7% active tracking on every seed.
Aggregate completed-game return is -12.64 and the positive-point fraction is
32.8% (+246/-503), versus -20.03 for matched random play. This is useful
evidence that Kindle can optimize a dense, task-specific reward from scratch,
but the reward rule itself encodes the desired tracking decision. It is not
autonomous reward discovery, sparse-score-only learning, or a Pong solve.

ARC-AGI-3 is also open. Install its optional dependencies with
`pip install -e 'python[arc]'` under Python 3.12 or newer (the ARC packages'
minimum, while Kindle's core bindings retain Python 3.9 support). The harnesses
use fixed action identities, dynamic availability masks, seeded random
baselines, level-event/max-level exit gates, and level gain per attempt (ARC retains cumulative
`levels_completed` across resets). The joint trainer can learn coordinate
clicks with `--coord-alpha`; inactive non-coordinate lanes are excluded from
that REINFORCE update. Every ARC harness also supplies the exact executed
`(x, y)` to the world model, including seeded-random and fallback clicks, so
different ACTION6 targets no longer alias in online or replay dynamics.

Current evidence does not show above-random **online ARC reinforcement
learning**. On CD82, five 30k-step seeds of
the historical intrinsic `full` preset produced 5 events and max level 1;
matched random runs produced 7 events and reached level 2 twice. On all 25
games at seed 42, random reached 5 games/6 events/max level 2 by 2k steps,
while the documented PPO + visual-reconstruction + goal-bonus recipe reached
4 games/4 events/max level 1. A coordinate-focused seed-42 probe over the six
ACTION6-only games at 5k steps also remained below random: per-lane explicit
level credit with `--coord-sigma 1.0` reached 1 game/1 event/max level 1,
versus random at 2 games/2 events/max level 1. Correct coordinate
representation has therefore not yet translated into an above-random solver.
The planner now searches the coordinate tail as well as the discrete action,
but its first matched FT09 result is also neutral: across seeds 42–44 at 5k
steps, coordinate shooting and uniform random each produced two total events
and max level 1. A synthetic GPU task verifies that shooting selects a learned
state-changing coordinate, so the remaining problem is the generic ARC score
(novelty/change does not reliably identify task progress), not queue plumbing.
An object-centric action probe now infers the modal background, prioritizes
pieces nested inside panels, and guarantees each proposed click lands on its
component. On FT09 this moved all eight immediately state-changing cells from
inaccessible ranks 21–33 into slots 0–7. Across seeds 42–44 it reached one
level event in every 5k-step run (3 total/max level 1), versus uniform
coordinates at 2 total/max level 1. This is improved cold-start coverage, not
yet multi-level completion.
These commands reproduce the calibrated random baseline and express a
prospective gate that is deliberately above it:

```bash
python python/examples/arc_agi3_multi.py \
  --random-policy --steps 10000 --seed 42

python python/examples/arc_agi3_multi.py \
  --steps 10000 --seed 42 --use-ppo 1 --entropy-beta 0.01 \
  --recon-loss-coef 10 --recon-visual-target 1 --goal-bonus 10 \
  --coord-alpha 1 --latent-dim 256 --hidden-dim 256 \
  --adam-eps 1e-4 --grad-clip-norm 0.5 \
  --require-games-with-events 10 --require-total-events 12 \
  --require-max-level 3
```

Passing that prospective single-seed gate would justify a multi-seed
learner-versus-random comparison; it is not currently a closed milestone.

For cloneable local ARC environments, the object-state search harness offers
a separate planning path. It probes visible components, removes actions with
duplicate outcomes, detects commuting binary transitions, and searches the
resulting visual state graph using `levels_completed` as the external goal.
General graphs regenerate actions and object coordinates at every state;
initial no-ops may become useful after another action. Invalid cloned-engine
branches are skipped, and both expanded states and attempted transitions are
bounded. Search itself is simulator-backed planning, not learned-policy
evidence:

```bash
python python/examples/arc_agi3_object_search.py \
  --game-prefixes sp80,vc33,lp85,ft09,r11l,lf52,cd82,su15 \
  --max-levels 1 --max-states 500 --max-transitions 20000 \
  --max-depth 10 --max-proposals 64 --commutativity-limit 24 \
  --require-levels 1

python python/examples/arc_agi3_object_search.py \
  --game-prefixes ft09 --max-levels 2 --max-states 10000 \
  --require-levels 2

python python/examples/arc_agi3_object_search.py \
  --game-prefixes vc33 --max-levels 3 --max-states 50000 \
  --max-transitions 500000 --require-max-level 3
```

The first gate reaches L1 on all eight selected games in 246 expanded states.
Four are new relative to the frozen-root search: SP80, CD82, LF52, and SU15.
Across all 19 games exposing complex actions, the same dynamic abstraction
reaches 8 at L1 under 200–500-state caps; this broader corpus has no matched
random comparison. The original six ACTION6-only comparison remains 4/6 versus
random 2/6. The second command deterministically reaches FT09 L2 in 11 solution
actions after 2,978 expanded states; VC33 reaches L2 in 10 actions/45 states.
The third command reaches VC33 L3 in 33 total solution actions and 1,169 total
expansions (the L2→L3 segment is 23 actions/1,124 expansions). Deeper FT09
search remains combinatorial.

Successful search paths can be distilled into Kindle's persisted categorical
and coordinate policies. The search exports pre-action hybrid object/pixel
tokens, availability masks, stable action identities, and normalized click
targets. A masked supervised policy update trains positive cross-entropy labels
directly, without synthetic rewards or an untrained value baseline. The CPU
coordinate head consumes the raw 64-value token plus Kindle's fixed task
embedding. This prevents cross-game click interference without restoring the
redundant GPU latent, so checkpoints remain portable across training and
evaluation batch sizes. Multiple compatible corpora can be merged; exact
duplicate states are removed and contradictory model-visible labels are
rejected. Evaluation reloads a fresh one-lane agent and executes deterministic
heads with no cloning or search:

```bash
python python/examples/arc_agi3_object_search.py \
  --game-prefixes sp80,vc33,lp85,ft09,r11l,lf52,cd82,su15 \
  --max-levels 1 --max-states 500 --max-transitions 20000 \
  --max-depth 10 --max-proposals 64 --commutativity-limit 24 \
  --require-levels 1 --save-demonstrations /tmp/kindle-arc-l1.json

python python/examples/arc_agi3_object_search.py \
  --game-prefixes ft09 --max-levels 2 --max-states 10000 \
  --require-levels 2 --save-demonstrations /tmp/kindle-arc-ft09-l2.json

python python/examples/arc_agi3_object_search.py \
  --game-prefixes vc33 --max-levels 3 --max-states 50000 \
  --max-transitions 500000 --require-max-level 3 \
  --save-demonstrations /tmp/kindle-arc-vc33-l3.json

python python/examples/arc_agi3_distill.py \
  /tmp/kindle-arc-ft09-l2.json /tmp/kindle-arc-vc33-l3.json \
  /tmp/kindle-arc-l1.json \
  --epochs 35000 --policy-lr 0.001 --policy-check-interval 25 \
  --pointer-epochs 30000 --pointer-lr 0.0005 \
  --require-pointer-accuracy 1.0 \
  --weights-dir /tmp/kindle-arc-policy \
  --evaluate-game-prefixes sp80,vc33,lp85,ft09,r11l,lf52,cd82,su15 \
  --eval-max-steps 120 --require-eval-level 1 \
  --require-eval-levels ft09:2,vc33:3 \
  --require-eval-games-reached 8
```

The merged corpus contains 76 unique rows. Kindle's fixed action policy reaches
100% label accuracy, while a persisted variable-set pointer scores the 12
features of each current object and fits all 67 complex-action labels. A
separate `--epochs 0` process reloads `policy.safetensors` and
`candidate_pointer.npz`, then reaches L1 on 8/8 games, FT09 L2, and VC33 L3.
No clone or coordinate-motion heuristic is used during this evaluation.

The pointer is deliberately separate from the 18-way universal action head:
Kindle selects stable ACTION6, then the pointer compares however many objects
exist in the current frame (65 in the largest demonstrated set). This replaces
failed pixel regression and fixed-rank classification without adding padded
logits to native and Atari policies. Coordinate regression remains available
for older demonstrations lacking candidate features. This result proves
retention through demonstrated L3 paths, not unseen-level generalization or
autonomous task discovery.

Native environments return a `StepResult` containing the post-action
observation, reward, positive `task_event`, and `terminated`/`truncated` flags.
The task event is deliberately independent of reward sign and episode end.
Native loops should pass the result to `Agent::observe_step_results`; this
forwards reward, event, and homeostatic state and marks episode boundaries in
one call. The runnable
examples under `kindle-gym/examples/` use this path. Environments with
state-dependent legal actions can also implement `Environment::action_mask`
and call `Agent::set_action_masks_from_envs` immediately before `act`.

-----

## Python Bindings

Kindle ships an optional pyo3 extension under `python/`, built via [maturin](https://www.maturin.rs/). `maturin develop` installs into the active environment, so run it inside a virtualenv (it fails outside one):

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin 'gymnasium>=1.1,<2'
cd python && maturin develop --release
```

For the Atari harness, install the packaged optional dependencies with
`pip install -e './python[atari]'` from the repository root (or install the
equivalent Gymnasium Atari and headless OpenCV packages into the same venv).

Then in Python:

```python
import gymnasium as gym
import kindle

env = gym.make("CartPole-v1")
agent = kindle.Agent(obs_dim=4, num_actions=2, env_id=0, seed=0)

episode_returns = agent.run(env, steps=10_000)
print(agent.diagnostics())
```

`Agent.run(env, steps, homeo_fn=None)` drives any gymnasium-style env directly — it calls `reset()`/`step(action)` itself, handles both the 5-tuple and legacy 4-tuple step shapes, and auto-resets on episode end. It returns the completed episodes' extrinsic returns for monitoring; training remains purely intrinsic.

The Python extension is a thin wrapper — training still runs on GPU through meganeura. Python is for scripting and for dropping kindle into a gymnasium loop; it is not a second implementation.

-----

## Repository Structure

This is a Cargo workspace.

```
kindle/
├── Cargo.toml                 # [workspace]
├── kindle/                    # core library crate (git dependency; not on crates.io)
│   ├── Cargo.toml
│   ├── src/                   # agent, encoder, world_model, reward, policy, …
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
