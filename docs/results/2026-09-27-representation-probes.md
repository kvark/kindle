# Offline representation probes — partial controls

These are held-out supervised state probes, **not RL results or Phase 2 completion**.
[All per-target R², errors, counts, trajectory splits and selections](2026-09-27-representation-probes.json).

The table uses native phase15/JL64/mean features; raw RGB56 uses two frames.
Values are unweighted means over position or motion targets and probe-head seeds.
Undefined constant-target R² is excluded, not turned into zero. All variants remain in JSON.

| Model | Probe | Game | Position R² | Motion R² |
| --- | --- | --- | ---: | ---: |
| pretrained_tiny | ridge | Breakout | 0.827 | 0.372 |
| initial_tiny | ridge | Breakout | 0.805 | 0.332 |
| large | ridge | Breakout | 0.984 | 0.703 |
| reconstruction_cnn | ridge | Breakout | 0.830 | 0.205 |
| raw_rgb56 | ridge | Breakout | 0.716 | -0.025 |
| pretrained_tiny | ridge | Pong | 0.923 | 0.246 |
| initial_tiny | ridge | Pong | 0.927 | 0.340 |
| large | ridge | Pong | 0.981 | 0.719 |
| reconstruction_cnn | ridge | Pong | 0.943 | 0.311 |
| raw_rgb56 | ridge | Pong | 0.780 | 0.110 |
| pretrained_tiny | ridge | Seaquest | 0.529 | 0.006 |
| initial_tiny | ridge | Seaquest | 0.526 | 0.032 |
| large | ridge | Seaquest | 0.751 | 0.045 |
| reconstruction_cnn | ridge | Seaquest | 0.578 | 0.028 |
| raw_rgb56 | ridge | Seaquest | 0.274 | -0.089 |

Whole trajectories are held out: four train, two validation and two test seeds/game.
RAM is an offline target only. Hyperparameters and checkpoints are selected on validation;
test labels never enter fitting. Secondary visibility scores retain the disclosed Seaquest
sprite/mapping limitations; neither they nor test scores select examples or variants.

The frozen encoders expose 7×7×64 values. The stateless reconstruction CNN has no history;
identical phase0/15 fits may be reused only after byte equality of all three feature splits.
Raw RGB56 controls have 3×/6× as many values and are not size-matched encoder baselines.

Tiny has 250k prior RGB64 frames from Boxing/Pong/Freeway/Breakout/Qbert. Large uses VideoMix.
The reconstruction CNN trains on 12,288 native frames from this corpus's training split,
including Seaquest. These different experiences must not be attributed solely to architecture.

Matched three-seed learning curves remain required before deciding the 2D frontend.
See the [comparison protocol](../experiments/2026-09-27-representation-comparison.md).
