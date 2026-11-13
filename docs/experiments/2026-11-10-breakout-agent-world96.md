# Breakout agent with the state-only 96x96 world objective

The standalone series showed that a learned-RGB world whose decoder renders
from the recurrent state alone, trained with motion-weighted pixel
cross-entropy and a 16-step prior image rollout on 96x96 frames, carries the
ball in its state and beats the previous best world model at every horizon.
This run asks whether that world objective helps the whole agent play.

## Recipe

`python/examples/atari_vector.py` (copied into the root, hash recorded) with
the retained Breakout development recipe except for the world objective:
seed 1009, 200,000 aggregate actions, eight environments, published protocol,
sticky .25, native frames with one GPU resize to 96x96, 4M preset, Auto
precision, B8/T32 with full-sequence BPTT, replay ratio 16, replay capacity
100k, 20% guided episodes with imitation scale 100 and the world gradient, as
in the retained CDP learner. The retained learner's action-effects exploration
bonus (scale 1) is absent: the runner ties it to CDP, so this run differs from
the retained recipe in the world objective and in the missing bonus. `--world-pixel-loss
motion_cross_entropy --world-prediction-length 16` select the new objective;
CDP is off (plain learned RGB; the runner sets reconstruction scale 1, not
the 100 this paragraph first claimed; corrected November 10). Root `/x/Temp/kindle-breakout-world96-agent-20261110`, output
`candidate/`, `queue.json`.

## Evaluation

Frozen 24-episode evaluations (three per environment, zero updates) on
cohort 4000022003 in sampled and greedy modes, for the actual-initial checkpoint
and the 200k checkpoint. References on that cohort: retained CDP 1009
94.625 greedy / 70.500 sampled; fresh CDP 22003 75.250 / 44.708; their
actual-initial controls score about 2.

## Reading

This is one development seed on a shared cohort, not a reliability claim. A
clear gain over the retained CDP points would support moving the agent to the
state-only RGB world; parity or a loss would say the world's forecast quality
is not the limiting factor for play at this budget, or that 4M at 96x96 costs
too much imagination throughput. Report aggregate actions, emulator frames,
updates, wall time and the usual two-wall count.

## Correction (November 10, after the run)

The recipe paragraph above is wrong about what it matched. The retained
Breakout learner (`/x/Temp/kindle-breakout-replay-value1-20261102/full/
breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009/queue.json`)
trains 12M at 64x64 with B16/T16, centered CDP, the action-effects bonus
(scale 1), corrective guidance labels, actor unimix .01, encoder rate 4e-5,
dynamics rate 4e-4 (the CDP default), behavior rate 8e-5 and replay-value
scale 1. This run used 4M at 96x96 with B8/T32, no CDP, no bonus, plain
guidance labels, actor unimix 0, one 4e-5 rate for every group and replay-value
scale .3. Its result therefore does not isolate the world objective; the
[matched protocol](2026-11-10-breakout-agent-world96-matched.md) does.
