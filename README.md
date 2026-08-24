# Kindle

Kindle is an experimental DreamerV3 agent implemented on
[Meganeura](https://github.com/kvark/meganeura) and
[Blade](https://github.com/kvark/blade). It learns a categorical recurrent
world model from one stream of game frames, trains its actor and critic in
latent imagination, and uses a frozen DINOv3 ViT-S/16 as its visual frontend.

This repository has deliberately pivoted to a small, legible baseline. Earlier
Kindle policies, planners, reward circuits, pretraining scripts, and experiment
harnesses remain in git history; they are not part of the current architecture.

The native crates require Rust 1.88 or newer and a Blade-supported graphics
backend (Vulkan on Linux in CI, Metal on macOS). A software Vulkan driver such
as lavapipe is sufficient for tests; practical DINO inference and learning need
a real GPU.

## Baseline

```text
RGB game frame
    │  aspect-preserving resize
    ▼
frozen DINOv3 ViT-S/16 patch tokens
    │  fixed 384→64 projection + 2×2 spatial pooling
    ▼
7×7×64 observation ──► categorical RSSM ──► reward / continuation / reconstruction
                              │
                              └── posterior replay states
                                      │
                                      ▼
                               latent imagination
                                      │
                                      ▼
                          categorical actor + two-hot critic
```

The learning algorithm follows DreamerV3 at pinned upstream revision
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`:

- uniform online sequence replay with batch 16 and length 64;
- block-recurrent deterministic state plus 32 categorical stochastic variables;
- posterior/prior KL balancing, 1 free nat, and 1% uniform mixing;
- feature reconstruction, two-hot reward prediction, and learned continuation;
- 15-step imagination from every posterior sequence state;
- REINFORCE actor loss, entropy regularization, lambda returns, replay-value
  learning, and a 2% EMA value model;
- per-parameter adaptive gradient clipping followed by RMS normalization and
  momentum (LaProp-style ordering), at `4e-5` with 1,000-step warmup.

Intentional differences are explicit:

- Perception is frozen DINOv3 rather than DreamerV3's learned pixel encoder.
- Replay reconstructs a fixed compressed DINO feature map rather than pixels.
- World-model gradients are truncated every 8 recurrent steps to keep
  Meganeura's static training graph practical. Each update still samples the
  full 16×64 replay batch, averages gradients across all eight chunks, and
  starts imagination from all 1,024 posterior states.
- Replay holds 100,000 compressed observations rather than D3's five million;
  the current f32 observation and recurrent context consume roughly 2.3 GB at
  that capacity. Larger/f16 lifetime storage is a measured follow-up.
- Dreamer computation is f32 on the current backend rather than D3's bfloat16.
- World and behavior parameter groups use separate optimizer sessions;
  objectives, per-parameter clipping, and representation-gradient paths are
  preserved.
- The first baseline supports categorical actions only.

Extrinsic and intrinsic reward channels are separate in replay and independently
scaled. Intrinsic reward defaults to zero so externally rewarded Dreamer remains
the control baseline.

## DINOv3 weights

Kindle expects `model.safetensors` from
[`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32`.

The weights are externally licensed and are never committed here. Review Meta's
model license before downloading or redistributing them. The native
implementation is numerically checked against Transformers/PyTorch golden
outputs by an ignored local parity test.

## Rust use

```rust,ignore
use kindle::{
    ActionMode, DreamerAgent, DreamerConfig, Reward, RgbFrame, Transition,
};

let config = DreamerConfig::new(18);
let mut agent = DreamerAgent::new(config, "/models/dinov3/model.safetensors", None)?;

let first = RgbFrame::new(width, height, first_rgb);
agent.begin_episode(&first);

let action = agent.act(ActionMode::Sample, action_mask.as_deref());
let transition = Transition {
    frame: RgbFrame::new(width, height, next_rgb),
    reward: Reward { extrinsic: score, intrinsic: 0.0 },
    terminated,
    truncated,
};
agent.observe(&transition);
let reports = agent.learn_scheduled(1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`kindle-gym` contains one rendered GridWorld for native integration and the
first measured baseline. It writes JSONL learner, episode, interval, and final
summary events to stdout:

```bash
cargo run -p kindle-gym --example grid_world --release -- \
  /models/dinov3/model.safetensors --steps 100000 --seed 0 \
  --checkpoint checkpoints/gridworld-seed-0 \
  --output runs/gridworld-seed-0.jsonl

# Matched random valid-action control (does not need DINO weights):
cargo run -p kindle-gym --example grid_world --release -- \
  --random --steps 100000 --seed 0

# Resume model/optimizer state; replay is deliberately refilled from scratch:
cargo run -p kindle-gym --example grid_world --release -- \
  /models/dinov3/model.safetensors --steps 100000 \
  --restore checkpoints/gridworld-seed-0 \
  --checkpoint checkpoints/gridworld-seed-0-resumed

# Frozen greedy evaluation (updates recurrent state, but not parameters):
cargo run -p kindle-gym --example grid_world --release -- \
  /models/dinov3/model.safetensors --steps 10000 \
  --restore checkpoints/gridworld-seed-0 --evaluate

# Short compute-heavy diagnostic, explicitly not the default baseline:
cargo run -p kindle-gym --example grid_world --release -- \
  /models/dinov3/model.safetensors --steps 2000 --model-size tiny \
  --learning-rate-warmup 0 --train-ratio 256

# Frozen-perception probe for GridWorld position and food-state separability:
cargo run -p kindle-gym --example probe_grid_world --release -- \
  /models/dinov3/model.safetensors --samples 500 --seed 0
```

## Python use

Build the extension with Maturin:

```bash
cd python
python -m venv .venv
. .venv/bin/activate
pip install maturin
maturin develop --release --extras test
```

The Python agent consumes `H×W×3` NumPy-compatible `uint8` frames:

```python
import kindle

agent = kindle.Agent("/models/dinov3/model.safetensors", num_actions=18)
agent.begin_episode(first_frame)
action = agent.act()
agent.observe(next_frame, extrinsic_reward=reward,
              terminated=terminated, truncated=truncated)
reports = agent.learn_scheduled()
```

See [`python/examples/atari.py`](python/examples/atari.py) for a Gymnasium loop.
ARC and other games should adapt their native controls into a stable categorical
vocabulary plus an optional validity mask; coordinate-parameterized actions are
a later extension, not hidden inside this baseline.

Validity masks constrain live action selection only. Latent imagination uses
the full fixed vocabulary, so adapters should give state-invalid actions a
defined benign/no-op meaning until mask prediction is added to the world model.

## Checkpoints

`DreamerAgent::save_checkpoint()` writes:

- world-model parameters and optimizer moments;
- actor/value parameters and optimizer moments;
- the slow value model;
- configuration, source revisions, learner counters, and return normalization.

The metadata also pins the Meganeura/Blade revisions and the fixed DINO
projection seed/shape. It does not embed or hash the separately supplied DINO
weights.

Replay and the in-flight episode are intentionally excluded. Restoring therefore
starts between episodes with empty replay while retaining learned parameters and
optimizer state.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib

# Serialized GPU act/learn/checkpoint and learning-wiring canaries:
cargo test -p kindle tiny_ --lib -- --ignored --test-threads=1

# Full DINO checkpoint parity:
KINDLE_DINOV3_WEIGHTS=/models/dinov3/model.safetensors \
  cargo test -p kindle vits16_checkpoint_matches_hugging_face \
  --lib -- --ignored
```

The design rationale and next research gates are in
[`docs/kindle_single_life_dreamer_plan.md`](docs/kindle_single_life_dreamer_plan.md).

## Repository

```text
kindle/src/dreamer/   DreamerV3 model, replay, learner, and checkpointing
kindle/src/vision/    native frozen DINOv3 ViT-S/16
kindle/src/env.rs     minimal visual environment boundary
kindle-gym/           native rendered smoke environment
python/               compact PyO3 binding and Atari example
```

Kindle is experimental research software, not a production-safe autonomous
control system.
