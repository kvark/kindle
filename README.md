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

- online-first sequence replay that consumes fresh non-overlapping sequences
  FIFO before uniform fallback, with batch 16, length 64, and a 1,024-item
  learner prefill gate;
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
- The decoder keeps 64 hidden channels per patch before its 64-channel DINO
  output. Dreamer's small pixel presets end at only 4–16 channels because they
  predict RGB; reusing that width imposed a measurable low-rank bottleneck.
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

Scheduled learning deliberately returns no reports until replay exposes 1,024
complete sequence starts (1,088 frames with the default length and context).
`DreamerCore::learn()` remains available for isolated learner diagnostics.
Live policy sampling, live and training posterior sampling, replay selection,
imagination, and read-only probes use independent deterministic RNG streams.
Changing learner throughput or imagination length therefore does not silently
advance the environment policy or replay RNG.

Learner reports expose both D3's floor-clipped dynamics/representation losses
and `raw_kl`. `DreamerConfig::dynamics_free_nats` can optionally override only
the prior-training floor while `free_nats` continues to protect posterior
information; leaving it unset preserves D3's shared one-nat default. The native
and Python runners also expose the existing dynamics loss coefficient for
controlled prior-alignment diagnostics.
`DreamerConfig::actor_learning_starts` can delay policy parameter updates by a
fixed number of learner updates without delaying either critic objective. Its
default is zero, actor optimizer moments continue tracking gradients, and
learner reports expose `behavior.actor_update_scale` so gated and active
updates are unambiguous in experiment logs.

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

# Atari-like fixed action vocabulary: border moves become benign no-ops in
# both the random control and Dreamer run instead of being masked live:
cargo run -p kindle-gym --example grid_world --release -- \
  --random --all-actions --steps 100000 --seed 0

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

# Strict dense visual-representation diagnostic, with action history removed:
cargo run -p kindle-gym --example grid_world --release -- \
  /models/dinov3/model.safetensors --steps 3000 --model-size 1m \
  --learning-rate 1e-3 --learning-rate-warmup 0 --train-ratio 256 \
  --free-nats 16 --reconstruction-loss-scale 3.1887755e-4 \
  --stop-replay-value-gradient --reward-mode dense-right \
  --randomize-position --random-action-steps 3000 \
  --checkpoint checkpoints/gridworld-dense-probe-seed-0

# Frozen-perception probe for GridWorld state separability:
cargo run -p kindle-gym --example probe_grid_world --release -- \
  /models/dinov3/model.safetensors --samples 500 --seed 1 \
  --randomize-position

# The same held-out probes after the RSSM, including the posterior policy's
# invalid-action mass, agreement with true/model-predicted rewarding actions,
# action-conditioned prior reward, and decoded-DINO prediction error against a
# no-change persistence baseline through horizon 15. The output also reports
# affine rank floors for the 64-channel DINO patches and the checkpoint's
# configured decoder depth, exposing widths that cannot represent the
# observation data even with a perfect latent:
cargo run -p kindle-gym --example probe_grid_world --release -- \
  /models/dinov3/model.safetensors --samples 500 --seed 1 \
  --randomize-position \
  --checkpoint checkpoints/gridworld-dense-probe-seed-0
```

The D3-compatible replay-value representation gradient remains enabled in
`DreamerConfig` by default. The strict frozen-DINO diagnostic stops that
gradient because matched ablations found that it prevented visual reward
learning at this budget; the behavior critic still receives its normal
replay-value updates.

Short optimization diagnostics can also set `--behavior-learning-rate`
independently from the world-model `--learning-rate`. Omitting it preserves
D3's shared rate; the split exists to test whether a fast world learner causes
the actor/critic to collapse before a sparse reward model is grounded.
`--imagination-length` isolates rollout-horizon failures. `--all-actions`
exposes GridWorld's border moves as no-ops, matching the full action vocabulary
used inside imagination; pass the same switch to the random control, frozen
evaluation, and representation probe.
`--actor-learning-starts N` is a causal diagnostic for premature actor
specialization: it holds only actor parameter updates through the first `N`
learner updates while the actor optimizer moments and behavior critic learn
from the same imagined and replayed states.

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
It reproduces D3's Atari-100k preprocessing (minimal actions, repeat 4,
two-frame max pooling, non-sticky actions, 0--30 reset no-ops, and Pillow
bilinear 64×64 RGB resizing) and train-ratio setting. Thus 100,000 runner steps
correspond to the conventional 400,000-frame Atari budget. The model uses the
mean feature reconstruction and isolated replay-value gradient supported by the
frozen-DINO diagnostics. Both adaptations remain explicit command-line and
Python constructor options. The same runner provides deterministic random
controls, resumable training, and frozen sampled evaluation matching D3;
`--greedy` is available as a separate policy diagnostic:

```bash
# Environment-only random control; the DINO path is not opened.
python python/examples/atari.py /unused ALE/Pong-v5 \
  --steps 100000 --random-policy --output runs/pong-random-seed0.jsonl

# Train and save, then evaluate without learner updates.
python python/examples/atari.py /models/dinov3/model.safetensors ALE/Pong-v5 \
  --steps 100000 --checkpoint checkpoints/pong-seed0 \
  --output runs/pong-seed0.jsonl
python python/examples/atari.py /models/dinov3/model.safetensors ALE/Pong-v5 \
  --steps 10000 --restore checkpoints/pong-seed0 --evaluate \
  --output runs/pong-seed0-eval.jsonl
```

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
