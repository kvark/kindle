# Breakout ball-tracking ceiling under sticky actions

Post hoc analysis, not declared in advance; CPU only. Script
`python/examples/breakout_sticky_oracle.py` (seed 931001, 512 origins, 32
replays each, a few minutes); output
`/x/Temp/kindle-world-scale-20261110/heldout/sticky-oracle-512-v2.json`.
Model-side split: `python/examples/breakout_world_event_split.py` on the
held-out probe files.

Every world report scores a forecast against the one recorded continuation
of a game played with sticky actions (probability .25 per emulator frame,
four frames per action). Given the recorded actions the executed ones are
random, so the future is not determined and even the emulator cannot match
the recorded continuation every time. The oracle plays the corpus teacher
(FeedbackPixelPolicy with random command blocks, block probability .025),
snapshots the emulator at random origins and replays the same 32 recorded
actions 32 times with fresh sticky draws. A replay tracks the continuation
when its RAM ball position is within the dense evaluation's 1.5 px at 96x96;
balls outside the detector window below the bricks are not scored, and
origins whose ball leaves play within 32 actions are dropped.

Checks: with stickiness 0 every replay reproduces the recorded continuation
at every horizon (64 origins, 50 with a paddle bounce at thirty-two steps).
The per-frame emulation matches ALE's own sticky actions in distribution
(same start state and action sequence, 2,000 runs each: paddle mean and
spread agree, total variation .001 to .06 after one to twelve actions). RAM
ball positions map onto the image detector's ball centre within 1.5 px for
99.5% of 572 held-out truth frames (median .53 px), so the ball ceiling is
directly comparable with the image metric. The paddle is not: the image
paddle detector on max-pooled frames agrees with the RAM paddle within
1.5 px for only 80% of truth frames, so paddle ceilings are not reported.

## Ceiling against the best world

| Horizon | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Oracle origins scored | 476 | 478 | 461 | 467 | 464 |
| Perfect simulator, one draw | .996 | .966 | .903 | .700 | .406 |
| Best point forecast (replay with most neighbours) | .998 | .983 | .941 | .771 | .504 |
| [Best world](2026-11-13-breakout-world-context-h32-131k.md), held-out, one draw | .957 | .778 | .628 | .505 | .328 |
| Best world as a share of the one-draw ceiling | 96% | 81% | 70% | 72% | 81% |

## By what the ball does inside the horizon

Oracle one-draw / point ceilings (origins) against the best world's held-out
hits; a paddle bounce is any vertical reversal below native y 150, a brick or
ceiling bounce one above it.

| Horizon | Paddle bounce | Brick or ceiling bounce only | No vertical bounce |
|---|---|---|---|
| 4 | .680 / .837 (43) vs 3/12 | 1.000 / 1.000 (35) vs 6/9 | .994 / .998 (400) vs 82/90 |
| 8 | .554 / .734 (94) vs 3/23 | 1.000 / 1.000 (82) vs 10/20 | .990 / .993 (285) vs 56/62 |
| 16 | .412 / .547 (232) vs 11/47 | .995 / 1.000 (159) vs 30/42 | .964 / .974 (76) vs 11/12 |
| 32 | .301 / .418 (390) vs 17/77 | .964 / .959 (74) vs 15/23 | none |

The world's scored set also holds origins whose ball is lost and served
again inside the horizon (the oracle drops them): 0/1, 0/6, 2/8, 4/10 and
6/16 at one to thirty-two steps. On the same held-out origins the 16-step
4M world scores paddle 3/23 and 10/47, bricks 11/20 and 22/42 at eight and
sixteen steps, and the 1M state-only world 5/23 and 8/47, 14/20 and 22/42.

## Reading

Sticky actions cap thirty-two-step tracking of the recorded continuation at
41% for a simulator scored with one draw and about 50% for the best possible
point forecast, because most thirty-two-step windows contain a paddle bounce
whose angle depends on where randomly delayed paddle moves left the paddle.
The best world's 33% held-out is 81% of the one-draw ceiling. Reading
thirty-two-step tracking as a fraction of perfect overstates the remaining
gap, and tracking against a single recorded continuation cannot show much
further progress there. Distribution-level scoring (several draws against
the oracle's replay distribution) would be needed for that.

The learnable gap sits in contacts at four to sixteen steps. Brick and
ceiling bounces are deterministic (oracle 96% to 100%), yet the world tracks
50% to 71% of them; paddle bounces reach 13% to 25% against a one-draw ceiling
of 41% to 68%; straight flights are near the ceiling (90% to 92% against 96%
to 99%). At sixteen steps the world stands at 72% of the ceiling, the
largest relative shortfall. The oracle's origins come from fresh play with
the corpus teacher, not from the held-out corpus states (which were not
saved), so the comparison is between matched distributions, not matched
origins.
