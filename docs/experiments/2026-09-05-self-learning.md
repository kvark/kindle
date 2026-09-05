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
result for the causal architecture, not yet a multi-seed or Atari result.
Artifacts: `runs/selflearn-20260905-grid-predictive-seed0{,-eval}.{jsonl,log}`
and `checkpoints/selflearn-20260905-grid-predictive-seed0`.

The Pong zero-update baseline completes 50k sampled actions and 55 episodes at
mean return −20.3455, with zero learner updates and 199,948 actual action frames.
Its execution time is 302.24 s, excluding construction. This is an untrained
behavior baseline, not an Atari-100k training-window score.
Pong prediction-only seed-0 training is underway, followed automatically by frozen
evaluation. It has 10,281,233 trainable parameters. Startup checks verify the same
executable, runner, encoder, ALE, GPU, vocabulary and initial counters as the
zero-update baseline; only train ratio differs (0 versus 256). First-1,000-action
counts, rewards and frame accounting match before the learner starts. Early
updates are finite and take about 0.52 s. This is runtime evidence, not a learned
Pong result yet. The matched reconstruction run and independent training-seed
confirmations remain required.

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
