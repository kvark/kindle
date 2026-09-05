# Kindle

Kindle is an experimental Rust agent that learns while acting. It combines
Dreamer's recurrent world model and imagined actor/critic with frozen DINOv3
perception, using [Meganeura](https://github.com/kvark/meganeura) and
[Blade](https://github.com/kvark/blade) for native GPU computation.

The current pivot adds action-conditioned prediction of unseen visual features.
The reconstruction agent remains the measured control; a predictive objective
is not yet a demonstrated replacement. Games are our first testbed, with
intrinsic motivation, continual learning and experience sharing as longer-term
goals.

Read the [single project plan](docs/kindle_single_life_dreamer_plan.md) for the
architecture, evidence and research gates, and [AGENTS.md](AGENTS.md) for working
directions. Superseded planners and training stacks remain in git history.

## Architecture

```text
previous belief + action ──► deterministic state ──► forecast frozen features
                                     │
RGB ──► frozen DINOv3 ──► 7×7×64 ────┤
                                     ▼
                            categorical posterior
                                     │
                     reward / continuation / reconstruction
                                     │
                       imagined actor + two-hot critic
```

The forecast reads the deterministic state *before* the current observation
enters the posterior. The feature-reconstruction control reads the posterior.
Both retain balanced KL, categorical sampling, replay-value learning, 15-step
imagination and DreamerV3's LaProp/AGC optimizer ordering.

Three objective settings are exposed in both native and Atari runners:

| Experiment | `--reconstruction-loss-scale` | `--future-prediction-loss-scale` |
| --- | ---: | ---: |
| Calibrated reconstruction control | 0.25 | 0 |
| Predictive auxiliary | 0.25 | 0.25 |
| Prediction only | 0 | 0.25 |

Zero disables and removes the corresponding head's parameters. Future targets
are stop-gradient frozen features; reset observations do not contribute to that
loss. Both heads use the same spatial structure and matched downstream
initialization. These coefficients are experimental settings, not universally
tuned values.

Intentional differences from the pinned upstream DreamerV3 include frozen DINO
instead of learned RGB perception, f32 computation, 100k-entry replay, and
separate world/behavior optimizer sessions. Library defaults retain BPTT 8 and
train ratio 32 for compatibility; measured Atari controls explicitly use full
BPTT 64, B16×T64, row microbatch 4 and ratio 256. Microbatching accumulates
gradients before one update; it does not truncate recurrence.

## Build and weights

Use Rust 1.88 or newer and a Blade-supported GPU backend. Linux experiments use
Vulkan. Practical learning needs a hardware accelerator.

DINO weights are supplied separately:
[`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32`.
The model is externally licensed and is not committed here.

On multi-adapter hosts, `MEGANEURA_DEVICE_ID` selects a backend-reported numeric
device ID, not an adapter ordinal. Our RTX 5080 uses `0x2c02`; omit the variable
to retain Blade's default selection.

```bash
cargo build --release --workspace --examples

cd python
python -m venv .venv
. .venv/bin/activate
pip install maturin
maturin develop --release --extras test
```

## Native use

```rust,ignore
use kindle::{ActionMode, DreamerAgent, DreamerConfig};

let mut config = DreamerConfig::new(environment.action_count());
config.loss_scales.reconstruction = 0.25;
let mut agent = DreamerAgent::new(config, "/models/dino/model.safetensors", None)?;
agent.begin_episode(&environment.reset());

let action = agent.act(ActionMode::Sample, None);
let transition = environment.step(action);
let reward_channels = agent.observe(&transition);
let reports = agent.learn_scheduled(1);
if transition.terminated || transition.truncated {
    agent.begin_episode(&environment.reset());
}
```

The first reset and executed transitions have different replay semantics.
Learning starts after enough complete replay sequences exist. State and learning
persist across natural episodes; `begin_episode` resets recurrence only.
Validity masks affect live actions, not imagination: give invalid actions a
benign meaning or use the same fixed vocabulary throughout.

GridWorld is a small visual integration task, not a general benchmark:

```bash
cargo run --release -p kindle-gym --example grid_world -- \
  /models/dino/model.safetensors --steps 10000 --model-size 1m \
  --all-actions --world-backprop-length 64 --train-ratio 256 \
  --random-action-steps 10000 --learning-rate 0.001 \
  --behavior-learning-rate 0.0001 --learning-rate-warmup 0 \
  --reconstruction-loss-scale 0.25 \
  --checkpoint checkpoints/grid --output runs/grid.jsonl

cargo run --release -p kindle-gym --example grid_world -- \
  --random --all-actions --persistent --steps 10000

cargo run --release -p kindle-gym --example grid_world -- \
  /models/dino/model.safetensors --all-actions --steps 10000 \
  --restore checkpoints/grid --evaluate --output runs/grid-eval.jsonl
```

This native gate uses diagnostic learning rates and a fixed random training
trajectory before frozen evaluation. Omit `--random-action-steps` to let the
learning agent collect its own data; the persistent reward comparison does so.

`--persistent` disables the artificial episode limit. Exhaustion respawns the
agent through normal dynamics, costs one game reward, and preserves food
progress. In this mode positive/negative reward events count food/deaths.
Keep this flag and action vocabulary matched across training and evaluation.

`--visitation-bonus --intrinsic-reward-scale 0.1` enables an optional bounded
visual-novelty experiment. A fixed 16-bit random-hyperplane hash counts visits;
features are centered on a checkpointed, fixed first-observation reference.
The bonus is `1/sqrt(previous_count + 1)`. Counts survive episode resets and
checkpoints, and saturate instead of wrapping. Collisions reduce novelty.
This is not epistemic uncertainty. Intrinsic and extrinsic channels are logged
separately; neither intrinsic generation nor its learning scale is enabled by
default. `--train-ratio 0` disables scheduled updates for coverage/noise checks;
the native `--fixed-action N --all-actions` supplies a stationary-action control.

## Atari and analysis

From an activated Python environment at the repository root:

```bash
python python/examples/atari.py /models/dino/model.safetensors ALE/Pong-v5 \
  --steps 100000 --seed 0 --atari-protocol published \
  --model-size 12m --world-backprop-length 64 --world-microbatch-size 4 \
  --checkpoint checkpoints/pong --output runs/pong-seed0.jsonl

python python/examples/atari.py /unused ALE/Pong-v5 \
  --steps 100000 --seed 0 --atari-protocol published --random-policy \
  --output runs/pong-random-seed0.jsonl

python python/examples/summarize_atari_scores.py \
  runs/pong-seed0.jsonl runs/pong-seed1.jsonl runs/pong-seed2.jsonl \
  --minimum-kindle-seeds 3 --require-uninterrupted-kindle

python python/examples/summarize_atari_training.py runs/pong-seed0.jsonl
```

`published` uses all 18 actions, no reset no-ops, repeat four, max-pooling,
non-sticky actions, a 100k-frame episode cap and Pillow 64×64 RGB resizing.
`current` follows the pinned upstream source's newer wrapper.
`published-minimal` changes only the action vocabulary. Do not mix protocols.

The conventional score window is 350k–400k nominal frames, with episode scores
averaged within seed before averaging seeds. These are training-window scores,
not frozen endpoint evaluations. `--restore ... --evaluate` runs a frozen
sampled policy; `--greedy` is a separate diagnostic. Restores refill replay and
must not be presented as uninterrupted runs.

The recovered 12M Pong runs scored −4, +6 and −2 (seed mean 0), versus matched
random −20.623. Local pinned upstream 1M and 12M controls also exist. See the plan for
the small episode counts, runtime/perception differences and artifact locations.

Read-only native and Atari probes measure reward calibration, representation
separability, open-loop feature error against persistence/unrelated actions,
actor/model agreement and critic calibration. Each example's `--help` describes
the exact protocol.

## Checkpoints and verification

Checkpoints contain world/behavior parameters and optimizer moments, the slow
critic, configuration, backend/vision provenance, counters, return normalization
and optional visitation counts. Replay, scheduler credit, exact RNG state and
the in-flight environment are absent. Restore is model recovery into a fresh
data segment, not exact lifetime continuation. Backend revisions are validated;
old artifacts require their original checkout. Experimental head and hash
revisions are also checked. Run headers expose model provenance; Atari headers
add the executable extension and runner hashes, preventing mixed implementations
from silently becoming a multi-seed score.

New pixel-agent checkpoints also fingerprint the actual DINO weight file.
Restore checks its SHA-256 before GPU construction. Legacy checkpoints without
that field require the pinned reference file; equal tensor shapes do not make
different perceptual representations compatible.

`LearnReport.timing` separates replay, posterior, imagination, training and
synchronization wall time. The synthetic canary avoids perception/environment
costs and can export inference and gradient traces. Gradient captures exclude
optimizer/clipping/accumulation and use the final row microbatch. Graphs exceeding
Blade's timestamp capacity retain ordinary wall time and an explicit missing-trace
reason; they are not silently reported as complete GPU profiles:

```bash
MEGANEURA_GPU_TIMING=1 cargo run --release -p kindle --example dreamer_canary -- \
  12m 64 4 --learn --updates 8 --profile-dir runs/profile

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo test --release -p kindle tiny_ --lib -- --ignored --test-threads=1
KINDLE_DINOV3_WEIGHTS=/models/dino/model.safetensors \
  cargo test --release -p kindle vits16_checkpoint_matches_hugging_face \
  --lib -- --ignored
PYTHONPATH=python python -m pytest python/tests
```

Serialize GPU-heavy tests and experiments on one device. Trace instrumentation
changes launch overhead; report ordinary wall time alongside GPU timestamps.
