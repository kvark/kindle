# Kindle

**[Project status: done, in progress, next, results and videos](STATUS.md)**

Kindle is an experimental Rust agent that learns while acting. It combines a
Dreamer recurrent world model and imagined actor/critic with frozen causal
LeVJEPA perception, using [Meganeura](https://github.com/kvark/meganeura) and
[Blade](https://github.com/kvark/blade) for native GPU computation. DINOv3 remains
an explicit control. Python supplies environment adapters and analysis, not a
second learner.

Six Atari streams share batched inference and one learner, with independent
causal histories. This is not yet a general gameplay policy; the
[status dashboard](STATUS.md) tracks completed game gates and current work.

See the [single project plan](docs/kindle_single_life_dreamer_plan.md) for current
scores, whole-rollout **videos**, world-model reports and next experiments;
[current evidence/archive](docs/experiments/README.md) for retained results; and
[AGENTS.md](AGENTS.md) for working directions. Swarms follow strong single-actor
learning and cross-game transfer, not the other way around.

## Architecture

The deterministic prior predicts frozen visual features **before** the current
frame enters the posterior. The optional reconstruction control reads the
posterior instead. Both retain categorical RSSM state, balanced KL, sequence
replay, 15-step imagination, a two-hot critic and LaProp/AGC optimizer ordering.

LeVJEPA consumes each arrival through bounded 16-frame causal chunk prefixes,
projected to 7×7×64 features. Chunk boundaries reset perception only; episode
boundaries also reset recurrent belief. Its 303M pretrained encoder is frozen.
The selected 12M learner uses F32, full BPTT64 and replay ratio 256. Library
defaults still use BPTT 8 / ratio 32; pass experiment settings explicitly.

| Objective control | Reconstruction scale | Future-prediction scale |
| --- | ---: | ---: |
| Reconstruction | .25 | 0 |
| Auxiliary prediction | .25 | .25 |
| Prediction only | 0 | .25 |

A zero scale removes that head. Future targets are stop-gradient frozen features;
reset observations are not predictable transitions. Row microbatching accumulates
gradients before one update, without truncating recurrence. A bounded visitation
bonus is available behind a separate intrinsic-reward channel and is off by
default; it is not a demonstrated solution to sparse-reward exploration.

## Build and weights

Use Rust 1.92 or newer and a Blade-supported GPU backend. Learning needs hardware
acceleration; CI also exercises small synthetic canaries on lavapipe.

```sh
cargo build --release --workspace
python -m venv .venv
. .venv/bin/activate
pip install maturin
cd python
maturin develop --release --extras test,atari
```

Weights are separately licensed and not committed:

- LeVJEPA: [galilai-group/LeVJEPA-VideoMix-Large](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large),
  snapshot `e831a0347737fcaa660b39c57d41c109de399845`, CC-BY-NC-4.0.
- DINO: [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
  snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32`; see its model license.

Select the single-agent frontend with `encoder="levjepa"` in Python,
`--encoder levjepa` in `atari.py`, or `DreamerAgent::with_perception` in Rust.
The vector runner uses LeVJEPA. On multi-adapter hosts, set `MEGANEURA_DEVICE_ID`
and check the reported executing device. Current experiments require the exact
declared source-matched package; a checkout is not the identity of an old binary.

The compact causal-video encoder is **ViT-Tiny/16, 5.49M parameters**. Opt in with
`encoder="levjepa-tiny"` in Python agents or `--encoder levjepa-tiny` in the vector
runner, using an exported Tiny checkpoint—not the Large weights. Restore selects
the recorded architecture and verifies the weight hash. Large remains the fresh
default. Native pretraining and the first candidate's results are linked from
[STATUS.md](STATUS.md); mixed motion-probe results still require downstream testing.

## Acting, learning and evaluation

Native agents expose `begin_episode`, `act`, `observe` and `learn_scheduled`.
Observe the **executed** action and both termination/truncation flags. Learning
starts once replay contains enough complete sequences; natural episode resets
preserve learned weights. Validity masks affect live actions, not imagination.
`kindle-gym` provides a small visual GridWorld for integration checks.

For Atari, the vector runner steps without wall-clock pacing. Example recipe
from the repository root (use fresh outputs; experiment jobs must use the
[bounded host guard](docs/gpu_incident_response.md)):

```sh
python python/examples/atari_vector.py /models/levjepa/model.safetensors ALE/Pong-v5 \
  --atari-protocol published --num-envs 6 --seed 2017 --steps 400008 \
  --model-size 12m --batch-size 16 --batch-length 64 \
  --world-microbatch-size 16 --train-ratio 256 --learning-rate 0.00004 \
  --checkpoint checkpoints/pong --output runs/pong-train.jsonl

python python/examples/atari_vector.py /models/levjepa/model.safetensors ALE/Pong-v5 \
  --atari-protocol published --num-envs 6 --seed 100000 --steps 600000 \
  --restore checkpoints/pong --evaluate --episodes-per-env 4 \
  --output runs/pong-eval.jsonl
```

`published` uses 18 actions, no reset no-ops, repeat 4, max-pooling, non-sticky
actions, a 100k-frame episode cap and Pillow 64×64 RGB. Do not mix wrapper protocols.
Frozen evaluation samples actions with **zero updates**; greedy evaluation is a
different diagnostic. Retain every completed episode, faster-stream extra and
unfinished tail. Training-window returns are not final-policy competence.

Useful tools in `python/examples/`: `summarize_atari_training.py` for complete
accounting, `audit_atari.py` / `audit_atari_tasks.py` for frozen gates,
`replay_atari.py` for independent action/reward/boundary replay and whole videos,
and `run_upstream_control.py` for the pinned upstream comparison. Use `--help`.

## Checkpoints and world diagnostics

Checkpoints contain world/behavior parameters, optimizer moments, slow critic,
configuration, counters, normalizers and backend/frontend identity. Format 3
restore checks actual encoder bytes and complete tensors. Old checkpoints need
their original executable; never edit provenance to make a restore pass.

Replay, exact RNG, scheduler credit and live environment/belief are absent.
Restore is recovery into a fresh data segment, **not equivalent interrupted
continuation**. Hashed tensor files detect torn/mixed saves, but saving a generation
is not atomic. Keep a known-good checkpoint separately.

`probe_atari_dynamics.py` compares prior feature/reward/continuation forecasts
against recorded observations. Keep posterior estimates separate and use feature
persistence, unrelated actions and zero-reward baselines. Report sparse event
counts and both MAE/MSE; these are feature forecasts, not rendered imagined RGB.
See the plan's linked reports. Pretrained encoder loading is supported; offline
action-conditioned world pretraining and policy-skill transfer are not adopted.

## Verification and performance

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo fmt --manifest-path python/Cargo.toml -- --check
python -m pytest python/tests -q
```

Native GPU tests are explicitly ignored in ordinary unit testing; run declared
hardware checks serially. Actual learning stays on the GPU. **NVML is temporarily
disabled** on the development host; see the incident runbook before GPU work.

`LearnReport.timing` separates replay, posterior, imagination, training and
synchronization wall time. `dreamer_canary` isolates learner work and exposes
opt-in GPU profiles. Profile instrumentation changes overhead; judge throughput
with untraced matched runs. The qualified block optimization gives 27.3% higher
throughput, but the fixed R256 recipe still runs below aggregate real time.
Lower replay ratios and smaller frontends require learning-quality comparisons.
