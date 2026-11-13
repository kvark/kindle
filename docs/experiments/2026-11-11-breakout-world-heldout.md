# Held-out qualification of the two best standalone Breakout worlds

Every world-model number so far was scored on the November 7 development
origins, which were reused across dozens of runs and screens. This protocol
collects a fresh held-out corpus with a seed never used before and scores
the two best checkpoints on it once, with no selection.

## Corpus

`breakout_world_scale.py collect` with the 4M context config (96x96, eight
lanes), `--ticks 1024` per lane (8,192 actions, the development corpus's
size), `--seed 777001`, the same teacher-plus-random-block policy and sticky
.25 as every corpus in this series. Output
`/x/Temp/kindle-world-scale-20261110/corpus/heldout96`.

## Checkpoints

- 4M context frame at 65,536 updates:
  `train-96-context-4m-65k-v4/checkpoint-65536`.
- State-only 1M at 65,536 updates: `train-96-plain-65k-v2/checkpoint-65536`.

`probe` mode restores each checkpoint, forecasts 128 dense origins of the
fitting corpus and 128 of the held-out corpus (origin draws seeded 182003 and
182009 as in training), writes the ball-tracking metrics and the linear
state probe. Outputs `heldout/context-4m-65k` and `heldout/state-only-1m-65k`.
The 64x64 control cannot be scored on the same episodes because the
collector stores resized frames only and the pixel teacher's actions depend
on the frame side.

## Reading

Held-out counts within the development counts' range (plus or minus about
ten of 112 at each horizon, the spread seen between adjacent evaluation
points) would say the development screens did not overfit the reused
origins; a systematic drop would quantify the optimism of every earlier
number. One corpus, one seed: this is the first held-out point, not a
reliability claim.
