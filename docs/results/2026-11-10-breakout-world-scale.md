# Sixteen times more Breakout play does not lift long-horizon tracking

[Protocol](../experiments/2026-11-10-breakout-world-scale.md). Root
`/x/Temp/kindle-world-scale-20261110`; unit `kindle-world-scale-20261110.service`
(invocation `8f0483d510844f458ce09b29a9ecc4ef`, PID 2326563), guard directory
`guards-v4`. Source `9579ea49` on Meganeura main `a0eff600`.

## Data

The fresh fitting corpus holds 525,019 arrivals from 524,288 aggregate actions
(2,096,090 emulator frames, 723 game-overs, every one of the 18 commands at
least 4,679 times) collected in 278 s by eight lanes with the fixed pixel
teacher plus random 4–12 command blocks, sticky .25, seed base 251001. Its
distribution matches the November 7 corpus that every earlier learner reused:
on sampled frames the ball is visible in 86.3% versus 86.1%, episodes last
710 versus 684 arrivals, and reward per arrival is .0348 versus .0344. Only the
quantity differs: 16x more distinct arrivals, so 16,384 B8/T32 updates see each
arrival about eight times instead of 128.

Development scoring reuses the November 7 `standard-development` corpus and the
same selection seeds, so the natural (32 origins), contact (8) and ending (10)
groups and the 128 dense origins are the same as the control's.

## Learning

Unchanged early physical-feature recipe (1M/RGB64/B8/T32/H16/Auto/41001,
`motion_event_rollout`, RGB100, KL1/.1, rates 4e-4, spatial decoder/fusion/
action, LEFT/RIGHT/FIRE table). Native calls take about 127 ms per update on
Meganeura main. Ball within one resized pixel, counted over visible targets,
first latent draw; the control is the closed early-feature learner on the
33k-arrival corpus.

| Learner, cohort, updates | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Scale, dense development, 2048 | 84/112 | 86/109 | 8/109 | 0/107 | 0/114 | 0/95 |
| Scale, dense development, 4096 | 94/112 | 94/109 | 39/109 | 19/107 | 2/114 | 1/95 |
| Scale, dense development, 6144 | 98/112 | 93/109 | 42/109 | 17/107 | 7/114 | 1/95 |
| Scale, dense development, 8192 | 100/112 | 100/109 | 32/109 | 17/107 | 10/114 | 1/95 |
| Scale, dense development, 10240 | 103/112 | 101/109 | 51/109 | 22/107 | 10/114 | 1/95 |
| Scale, dense development, 12288 | 104/112 | 104/109 | 61/109 | 41/107 | 18/114 | 2/95 |
| Scale, dense development, 14336 | 101/112 | 102/109 | 60/109 | 40/107 | 15/114 | 2/95 |
| Scale, dense development, 16384 | 101/112 | 104/109 | 59/109 | 38/107 | 18/114 | 0/95 |
| **Control, dense development, 16,384** | 106/112 | 101/109 | 66/109 | 53/107 | 25/114 | 11/95 |
| Scale, dense fitting (new corpus), 16,384 | 98/111 | 95/118 | 47/117 | 28/116 | 6/105 | 2/116 |
| Control, dense fitting (old corpus), 16,384 | 147/163 | 159/177 | 114/180 | 90/176 | 43/171 | 24/179 |
| Scale, natural development (32 origins), 16,384 | 25/26 | 26/27 | 15/26 | 8/27 | 3/27 | 1/25 |
| Control, natural development (32 origins), 16,384 | 25/26 | 23/27 | 17/26 | 11/27 | 5/27 | 0/25 |

## Reading

Sixteen times more distinct play, with the recipe and the 16,384-update budget
unchanged, leaves development ball tracking where the control had it: equal
within a few targets at one and four steps, and lower at eight, sixteen and
thirty-two steps (38/107, 18/114, 0/95 against 53/107, 25/114, 11/95). The
scale learner's fitting cohort, drawn from frames it saw about eight times,
scores like its development cohort at every horizon, and the control's fitting
cohort, replayed 128 times, also matches its own development scores. Neither
learner memorizes trajectories; the long-horizon loss is a limit of the model,
objective or optimization, not of the corpus.

The scale curve climbs steeply between 8,192 and 12,288 updates (eight-step
17→41/107) and is flat afterwards, so the remaining budget at this learning
rate was not simply too short either. Observed reconstruction (101/112) and
one-step prediction (104/109) are essentially solved; the ball is lost while
being carried through the prior for more than a few transitions. The paddle,
brick, reward and ending metrics for every point remain in
`train/evaluations/*/*.json`.

Native calls took 2119.9 s for 16,384 updates (129.4 ms each) and the
whole loop 2248.1 s, against the control's 1,841 s/1,999 s on the fork,
so Meganeura main is about 15% slower on this recipe before any tuning.

Two follow-ups run serially in `kindle-world-scale-followup-20261110.service`
on the same corpus: the same 1M recipe for 65,536 updates (checks whether a
longer schedule moves the plateau) and the 4M preset for 16,384 updates (checks
capacity). Model-side changes to how the prior carries the ball come after
those two numbers. This is a development screen on reused seed 41001 and the
reused November 7 development origins; nothing here qualifies the world model.

## Failures retained

Two earlier queue attempts are retained: `guards-failed-v1` failed on the
collector constructing the learner without the recipe's objective, and
`guards-v2` completed the whole collection but was killed by the host guard's
3 s kernel-log probe during the 6.4 GB lane-file write; `guards-v3` failed the
same probe at startup while the page cache was still flushing. Collection
therefore ran as a plain memory-capped unit; training kept the guard.
