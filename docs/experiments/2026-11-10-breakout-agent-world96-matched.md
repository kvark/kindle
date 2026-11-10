# Matched Breakout agent: retained recipe with the pixel world objective

The [first agent run](2026-11-10-breakout-agent-world96.md) with the
state-only pixel world scored 4.76/4.79 frozen against the retained CDP
learner's 70.5/94.6, but it changed model size, resolution, batch geometry,
guidance labels, actor unimix, learning rates and the exploration bonus along
with the world objective. This protocol swaps only the objective.

## Recipe

The retained Breakout learner's exact command
(`/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009/queue.json`:
seed 1009, 200,000 aggregate actions, eight environments, published protocol,
sticky .25, native frames with one GPU resize, 12M, Auto, B16/T16 full-batch
world training, replay ratio 16, replay 100k, 20% guided episodes with
corrective labels, imitation scale 100 and the world gradient, actor unimix
.01, behavior rate 8e-5, replay-value scale 1, action-effects bonus scale 1,
checkpoint at 100k) with these changes only:

- `--cdp --cdp-centered --encoder-learning-rate 4e-5` removed; the world is
  the plain learned-RGB decoder with reconstruction 100 and
  `--world-pixel-loss motion_cross_entropy --world-prediction-length 8` (the
  rollout must be shorter than the 16-step window; the standalone recipe used
  16 of 32). Without CDP every parameter group trains at the base 4e-5, so the
  encoder rate is unchanged and the dynamics group loses CDP's 4e-4 default;
  that rate belongs to the objective being replaced and is not restored.
- The runner now accepts `--disagreement-scale` with a fresh pixel-world
  objective (the ensemble predicts detached encoder outputs, which exist in
  any learned-RGB world; the Rust graph never required CDP). A CPU graph test
  and a GPU tiny-agent test cover the combination.
- Two resolutions: `train96` (the standalone world's 96x96) and then `train64`
  (the retained learner's geometry, where the ball is sub-pixel after the
  resize). Same seed, same everything else.

A 768-action smoke (`smoke/`) reported the prior-rollout loss, the exploration
loss and non-zero intrinsic rewards; 220 ms per update at 96, VRAM peak 5.5
GiB. Root `/x/Temp/kindle-breakout-world96-matched-20261110`, outputs
`w96/` and `w64/`, `queue.json`, unit
`kindle-breakout-world96-matched-20261110.service`.

## Evaluation

Frozen evaluations (three natural episodes per environment, zero updates,
25k-action truncation) on cohort 4000022003 in sampled and greedy modes for
the actual-initial and 200k checkpoints of each run. References on that
cohort: retained CDP 1009 94.625 greedy / 70.500 sampled; fresh CDP 22003
75.250 / 44.708; actual-initial controls about 2.

## Reading

One development seed on a shared cohort. If either pixel-world learner
reaches the CDP band, the world objective is not what held the first run
back and the other differences are. If both stay near the first run's 5, the
pixel objective itself (as a replacement for CDP) is the cause at this budget,
and the next step is CDP plus the pixel rollout as an auxiliary loss rather
than a replacement. Report aggregate actions, emulator frames, updates, wall
time and two-wall counts; do not read guided online returns as competence.
