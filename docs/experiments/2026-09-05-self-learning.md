# First self-learning results with causal prediction

Started 2026-09-05, after kickoff commit `53edbbc`. This records experiments,
not a second roadmap; see [the project plan](../kindle_single_life_dreamer_plan.md).

## Question and acceptance

Can the causal architecture learn useful game behavior from its own actions?
Earlier predictive native runs used forced-random trajectories. Earlier successful
Atari runs used reconstruction. Neither answers this question by itself.

The first candidate is prediction-only: future-feature coefficient 0.25,
posterior reconstruction 0, unchanged categorical RSSM and imagined actor/critic.
The world/behavior models start fresh; frozen pretrained DINO remains perception.
Game rewards guide learning. No action overrides, offline trajectories, intrinsic
bonus, privileged game state, cloned training environment or checkpoint selection
enter these runs. This is reward-guided online self-learning, not intrinsic-only
or real-time lifelong learning.

Positive evidence must include both a native game and Atari, improving actual
behavior over random/untrained controls and retaining it under frozen evaluation.
A falling loss, a single lucky reward or the old reconstruction scores is not
completion. Preserve every seed and failure. A matched reconstruction control
checks regressions; the candidate need not beat it to establish initial learning.

## Declared first experiments

- Persistent GridWorld: size1m, full BPTT 64, B16×T64, row batch 16, train ratio
  256, world rate 1e-3, behavior rate 1e-4, no warmup, all four actions, 10k
  agent-selected actions. Evaluate the final checkpoint for 10k frozen greedy
  actions in a fresh world, with zero updates. Start seed 0; then replicate
  seeds 1 and 2. Rerun reconstruction seed 0 on the same production build.
  Random seeds 0/1/2 already collect 168/145/125 food in 10k actions under these
  unchanged dynamics. Require at least 450 training food, at least 100 food per
  1k actions in the final two windows, and at least 1,500 frozen food in each
  successful predictive seed before treating the native gate as positive.
- Pong: size12m, full BPTT 64, B16×T64, row batch 16, ratio 256, published
  wrapper, standard Atari rate/warmup, 100k agent-selected actions. The larger
  row batch is an explicit experiment, not a reinterpretation of the old
  microbatch-4 control. Pair prediction-only and reconstruction on the same
  build, initially seed 0, then replicate the successful causal setting.
  Report all complete episodes in the conventional final training window and
  50k actions of frozen sampled evaluation, including actual frame counts.
  Seek final-window and frozen means above −10, with at least five complete
  frozen episodes and a naturally terminated game with positive return (a
  time-limit cutoff does not count as a win), then confirm on another training
  seed. Keep random and zero-update baselines separate from learned behavior.

Inspect the seed-0 native result before starting long Atari work. If pure
prediction fails, diagnose it and test the existing auxiliary rather than
silently relabeling reconstruction as the new architecture.

## Status

Native prediction-only seed 0 passes the declared per-seed gate:

| Measure | Result |
| --- | ---: |
| Agent-selected training actions | 10,000 |
| Learner updates | 2,229 |
| Training food / deaths / game return | 1,137 / 38 / 1,099 |
| Final two 1k-action windows, food / deaths | 246 / 0; 245 / 0 |
| Frozen 10k-action food / deaths | 2,495 / 0 |
| Frozen learner updates | 0 |

The learning curve changes from 3 food / 9 deaths in actions 4,001–5,000 to
113 / 3, then 242 / 0, 245 / 0, 246 / 0 and 245 / 0 in successive windows.
All 2,229 world/behavior/timing reports are finite. Its header verifies
reconstruction 0, future prediction 0.25, zero forced actions, zero starting
updates and 828,707 parameters. The saved world contains future-predictor tensors
and no decoder tensors; its encoder and three tensor-file hashes validate.
Training takes 645.55 s excluding construction; frozen execution takes 54.01 s.
Neither run has a harness reset. This is the first positive agent-collected native
result for the causal architecture, not yet a multi-seed result.
Artifacts: `runs/selflearn-20260905-grid-predictive-seed0{,-eval}.{jsonl,log}`
and `checkpoints/selflearn-20260905-grid-predictive-seed0`.

The same-build native reconstruction control does **not** retain its late
training performance under the declared frozen greedy evaluation. Training
collects 1,100 food / 37 deaths; the final two 1k-action windows each collect
247 food with no deaths. Frozen evaluation collects only 2 food / 98 deaths,
with no food after its first 1k actions. Both runs have zero harness resets;
evaluation makes zero updates. The 2,229 training reports, saved tensors,
configuration, encoder, counters, hashes and reward/action accounting validate.
Training samples actions; evaluation is greedy and starts fresh recurrent state.
Their separate effects need diagnosis, not an assumed checkpoint-corruption
explanation or replacement of the failed endpoint. A sampled frozen diagnostic
would be additional evidence, not a substitute for this predeclared test.
Artifacts: `runs/selflearn-20260905-grid-control-seed0{,-eval}.{jsonl,log}`,
`...-audit.json`, `...-final-tensors.json` and the matching checkpoint directory.

The Pong zero-update baseline completes 50k sampled actions and 55 episodes at
mean return −20.3455, with zero learner updates and 199,948 actual action frames.
Its execution time is 302.24 s, excluding construction. This is an untrained
behavior baseline, not an Atari-100k training-window score.
Pong prediction-only seed 0 completes training and passes every declared
per-seed gate using the final 100k checkpoint:

| Measure | Result |
| --- | ---: |
| Agent-selected training actions / learner updates | 100,000 / 24,729 |
| Final 350k–400k nominal-frame training-window mean / games | +18.1429 / 7 |
| Whole-training complete-game mean / games | −3.8286 / 70 |
| Frozen sampled actions / learner updates | 50,000 / 0 |
| Frozen complete-game mean / naturally terminated wins | +19.0714 / 28 of 28 |
| Actual training / frozen action frames | 399,910 / 199,947 |
| Training / frozen execution time, excluding construction | 13,770.78 s / 306.66 s |

Its first win is **+13 at action 44,075**: episode 41 naturally terminates
after 1,962 actions, without truncation. The final training-window returns are
15/18/14/20/20/20/20; the frozen returns range from +8 to +20. No evaluation
game is truncated. The unfinished final evaluation game is +16 after 1,369
actions and is excluded from complete-game scores. Early training losses remain
in the whole-run mean; neither checkpoint nor episodes were selected by score.

It has 10,281,233 trainable parameters. Startup checks verify the same
executable, runner, encoder, ALE, GPU, vocabulary and initial counters as the
zero-update baseline; only train ratio differs (0 versus 256). First-1,000-action
counts, rewards and frame accounting match before the learner starts. Rechecking
the win confirms zero forced actions, reconstruction 0, future prediction 0.25,
extrinsic scale 1, intrinsic scale 0 and the frozen executable hashes. All 24,729
world/behavior/timing reports and all 241 saved F32 tensors are finite. The final
checkpoint has the predictive head and no reconstruction decoder; all three
tensor-file fingerprints validate. Frozen evaluation verifies the identical
configuration, model/encoder/backend identity and executable, starts at counters
100k/24,729, and makes no updates. Training averages 7.26 actions/s and the emulator
waits for learning. This demonstrates online learning, not real-time 15 Hz control.

Artifacts: `runs/selflearn-20260905-pong-predictive-seed0{,-eval}.{jsonl,log}`,
`...-score.json`, `...-eval-summary.json`, `...-audit.json` and
`...-final-tensors.json`. The immutable final copy is
`checkpoints/selflearn-20260905-pong-predictive-seed0-step100000`; earlier
diagnostic copies are retained, not used to select the evaluation endpoint.

The matched reconstruction seed-0 run is complete. Its configuration differs
only in the two objective weights, with the corresponding head identity and
131,072 additional posterior-input parameters (10,412,305 total). The executable,
encoder, backend, wrapper, full BPTT, row batch, schedule and behavior settings
match. Both runs complete 100k training actions and 24,729 updates, then evaluate
their final checkpoints for 50k sampled actions with zero updates:

| Seed-0 objective | Final training-window mean / games | Frozen mean / wins |
| --- | ---: | ---: |
| Prediction only | +18.1429 / 7 | +19.0714 / 28 of 28 |
| Reconstruction | +17.6667 / 6 | +19.2143 / 28 of 28 |

The control's final-window returns are 18/21/14/21/16/16. All 28 frozen games
terminate naturally without truncation; the unfinished +1 game after 152 actions
is excluded. Actual training/frozen action frames are 399,940/199,943; execution
takes 13,795.83/304.72 s excluding construction. All 24,729 training reports and
241 saved tensors are finite. Configuration, encoder, executable, checkpoint
hashes, the full ordered learner schedule and reward/action accounting validate.
Its saved world has reconstruction tensors and no future predictor.

Reconstruction's first win is +6 at action 74,366 (episode 49, length 3,626,
no truncation), versus prediction's +13 at 44,075. These first-win times are
diagnostics, not substitute endpoints or multi-seed evidence. The endpoint
results support viable causal learning with no observed seed-0 Pong regression,
not superiority or multi-seed reliability. Independent native/Pong training
seeds remain required.

Control artifacts: `runs/selflearn-20260905-pong-control-seed0{,-eval}.{jsonl,log}`,
`...-score.json`, `...-eval-summary.json`, `...-audit.json` and
`...-final-tensors.json`; immutable final copy:
`checkpoints/selflearn-20260905-pong-control-seed0-step100000`.

Fresh CPU-only Pong random controls are complete on the current wrapper/ALE:

| Seed | Final-window mean | Complete window episodes | Actual action frames |
| --- | ---: | ---: | ---: |
| 0 | −20.6154 | 13 | 399,903 |
| 1 | −20.7143 | 14 | 399,899 |
| 2 | −20.5385 | 13 | 399,901 |

Each executes 100k actions, with no reset no-ops and zero learner updates.
Artifacts: `runs/selflearn-20260905-pong-random-seed{0,1,2}.{jsonl,log}`.
They ran on the CPU while the native learner occupied the single GPU; native
wall times are descriptive, not isolated performance comparisons.

## Evaluation-analysis correction

Preparing frozen evaluation exposed an absolute/local-step mismatch in the
window summarizer. It now bounds restored windows by starting environment step
plus local run length, clips the default window to that segment, and reports the
run mode/offset. Seven regression cases cover restored training/evaluation and
malformed offsets; all 83 Python tests pass. A completed historical evaluation
successfully summarizes steps `(100000, 105000]` with zero learner updates;
artifact: `runs/selflearn-20260905-historical-evaluation-summary.json`.
Only analysis code changed. The running experiment's runner, native binary and
extension hashes remain unchanged and verified.

## Gameplay evidence preparation

`python/examples/record_atari.py` records raw emulator frames during frozen
evaluation or a random control, without changing the training runner or adding
policy calls. Its movie and raw-frame hashes link to the ordinary evaluation
log. Playback is nominal 60 Hz, not measured acting latency; reset no-ops count
as frames, but reset images do not. Existing outputs are refused and an encoder
failure cannot produce a completion manifest.

A current-wrapper Pong random seed-17 check executes 2k actions with and without
recording. All six JSONL records agree exactly except wall-time fields, including
the two episode returns (−20, −21), action counts, reset accounting and zero
updates. The movie decodes to exactly 7,998 frames at 160×210, matching actual
emulator steps. Nine recorder tests cover transition/RNG parity, no-op/reset
handling, output preservation and failure cleanup; all 92 Python tests pass.
Artifacts: `runs/selflearn-20260905-video-parity-plain.jsonl`,
`runs/selflearn-20260905-video-parity-recorded-v2.jsonl` and
`runs/selflearn-20260905-video-parity-v2.mp4{,.json}`. This is recording validation,
not a trained-policy result. Learned-policy recording waits for the single GPU.
The experiment's native binary, extension and Atari runner still match their
frozen executable manifest.
