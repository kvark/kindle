# Breakout agents without hand-made relevance priors

Declared in [the clean-representation screen](../experiments/2026-11-15-breakout-clean-representation.md):
plain DreamerV3 learned-RGB agents on the published protocol (18 actions,
sticky .25), 12M, 64x64 RGB, B16/T16, train ratio 16, replay 100k, 400k
actions, seed 1009, with no motion weighting, hand action features, context
frames, exploration bonus or teacher guidance. Five arms: A clean, B
`--actor-critic-gradient`, C six teacher-played demonstration episodes per
lane cloned at scale 100 into the world state, D both, E
`--no-replay-value-gradient`. Root `/x/Temp/kindle-breakout-clean-20261115`,
units `kindle-breakout-clean-20261115` (stopped by the queue's evaluation
check at C's first greedy evaluation, see below) and
`kindle-breakout-clean-2-20261115` (the remaining 20 jobs). Each arm trained
24,977 updates in 43 to 44 minutes (97 ms per update).

**No arm puts the ball into the model state.** A linear probe of the
posterior state on held-out teacher play reads the ball with a mean error of
19.5 to 24.8 source pixels in every arm, against 27.3 / 26.3 for an untrained
network; under 6% of ball rows are read within 8 px. The paddle is learned
everywhere (2.4 to 7.7 px). Value gradients from imagination (B) give the
best frozen play of the screen, about 7 points against 4, and the sharpest
paddle, without a better ball. Demonstrations sharpen the paddle but leave
greedy policies that never press FIRE, through a return-scale shock described
below. Removing D3's replay-value gradient (E) changes nothing in play.

## Frozen play

Three natural episodes per lane, 24 per cell, mean (best). An episode is cut
at 100,000 frames (25,000 actions); cells list cut episodes and are scored on
the natural ones; "stalled" means no natural episode.

| Arm | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy |
|---|---:|---:|---:|---:|
| A clean | 4.3 (10) | 4.0 (9) | 4.9 (12) | 4.1 (10) |
| B ac_grads | 7.2 (15) | 7.4 (13), 9 cut | 6.2 (13) | 5.8 (12), 13 cut |
| C demonstrations | 4.0 (7) | stalled, 24 cut | 3.9 (7) | stalled, 24 cut |
| D demonstrations + ac_grads | 6.4 (16) | stalled, 24 cut | 5.6 (11) | stalled, 24 cut |
| E no value gradient | 4.4 (13) | 4.6 (10) | 5.0 (12) | 4.2 (11) |

Context only: the guided learners scored greedy / sampled 223.0 / 89.8
(pixel world) and 192.0 / 90.6 (CDP) on cohort 4000777001 at 400k.

## State probe

`breakout_agent_probe.py` (seed 5550001; eight lanes of the edge-correcting
teacher, 1,536 actions each; ridge on lanes 0-3, penalty on 4-5, scored on
6-7; 3,050 test rows with a ball). Source-pixel mean absolute error and R².

| Checkpoint | Ball x | Ball y | Within 8 px | Paddle x | Ball dx R² | Ball dy R² |
|---|---:|---:|---:|---:|---:|---:|
| A initial | 27.3 (.35) | 26.3 (.07) | .021 | 23.8 (.51) | .16 | .00 |
| A final | 19.5 (.64) | 19.9 (.42) | .059 | 6.7 (.96) | .28 | .19 |
| B final | 23.0 (.51) | 23.0 (.27) | .043 | 2.4 (.99) | .24 | .11 |
| C final | 22.1 (.53) | 23.5 (.25) | .041 | 3.3 (.99) | .24 | .06 |
| D final | 21.1 (.59) | 22.2 (.30) | .042 | 3.7 (.99) | .25 | .01 |
| E final | 24.8 (.45) | 24.8 (.17) | .032 | 7.7 (.95) | .23 | .02 |

The initial checkpoints of B to E, probed after the screen (unit
`kindle-breakout-clean-3-20261115`, `queue-3.json`), give metrics identical
to A's at full precision, so every arm starts from the same probed
representation and A's row is the common floor.

Reference: the guided learners read the ball at 9 to 13 source pixels with
an earlier linear probe (different trajectories and protocol, so not a
matched comparison). Everything the clean arms learned about the
ball is a 3 to 8 px gain over an untrained network.

## Training curves

Online episodes and learner means per 50k actions
(post hoc tabulation of `train.jsonl`; columns: episodes, mean
return, policy entropy, return scale).

| Actions | A | B | C | D | E |
|---|---|---|---|---|---|
| 0-50k | 268, 1.6, 2.69, 1.0 | 267, 1.5, 2.77, 1.0 | 25, 207.8, 1.39, 7.1 | 25, 207.8, 1.39, 7.0 | 262, 1.6, 2.72, 1.0 |
| 50-100k | 251, 1.8, 2.07, 1.0 | 254, 2.2, 1.91, 1.0 | 57, 90.5, 1.05, 29.6 | 54, 95.3, 1.05, 29.6 | 219, 2.3, 1.86, 1.1 |
| 100-150k | 201, 3.1, 1.14, 1.6 | 180, 4.5, .72, 1.9 | 112, 3.9, .92, 48.2 | 113, 3.7, .92, 46.9 | 204, 2.7, 1.36, 1.2 |
| 150-200k | 193, 3.5, .84, 1.7 | 174, 3.2, .81, 1.8 | 126, 2.7, .92, 24.0 | 146, 2.1, .97, 24.7 | 195, 3.2, 1.51, 1.5 |
| 200-250k | 196, 3.1, .95, 1.9 | 220, 2.7, 1.03, 1.4 | 70, 4.1, .72, 1.4 | 80, 2.9, .78, 1.6 | 185, 3.7, 1.01, 2.0 |
| 250-300k | 188, 3.7, .75, 2.1 | 206, 3.2, 1.03, 1.7 | 53, 4.2, .54, 1.2 | 138, 3.2, .60, 1.0 | 182, 4.0, .52, 2.4 |
| 300-350k | 180, 3.9, .74, 2.4 | 175, 4.1, .70, 2.0 | 52, 2.9, .48, 1.1 | 129, 3.6, .40, 1.0 | 191, 3.6, .72, 2.4 |
| 350-400k | 169, 4.4, .61, 2.7 | 178, 4.8, .81, 2.6 | 44, 2.7, .43, 1.0 | 123, 4.0, .37, 1.1 | 179, 4.1, .76, 2.9 |

The demonstration rows of the first 100k actions are the teacher's (48
episodes, mean 186.9).

## Reading

- **Value gradients into the representation (B over A).** Frozen play
  improves by about 3 points on both cohorts and both modes, and the paddle
  error falls from 6.7 to 2.4 px; online returns end level (4.8 against
  4.4). The ball is not better read (23 against 19.5 px), so imagined actor
  and value losses sharpen what the policy already uses rather than
  discovering the ball. D over C repeats the sampled gain (6.4 / 5.6 against
  4.0 / 3.9).
- **Demonstrations (C over A, D over B).** Cloning the teacher sharpens the
  paddle (3.3 px against 6.7) and nothing else measured: the ball is no
  better read, sampled play is no better, and both demonstration arms
  produce greedy policies that never launch the ball. The mechanism visible
  in the curves is the return normalizer: imagined returns on demonstration
  replay push the return scale from 1 to 48, which shrinks normalized
  advantages (mean |advantage| .02 against .05 in A) for about 100k actions
  after the teacher stops, while the actor is still cloning a fading target.
  When the scale recovers near 200k, C's online play has settled into long
  low-fire episodes (44 episodes in the last 50k actions against A's 169);
  D's online episodes are of normal length, but its greedy policy stalls
  just the same. Greedy stalls are not specific to demonstrations: B's greedy
  policy also fails to launch in 9 and 13 of 24 episodes. With 18 actions of
  which 6 fire, a greedy policy that prefers a non-fire action after a lost
  life can stay unlaunched until the episode cap. The return-scale shock
  is one mechanism consistent with the curves, not an isolated cause. The
  copycat risk of cloning with the previous action as an input remains
  untested here; the probe was taken after the demonstrations had left the
  replay and does not show whether the ball was present at the end of the
  demonstration phase.
- **No value gradient (E against A).** Play is unchanged (4.4 / 4.6 and
  5.0 / 4.2 against 4.3 / 4.0 and 4.9 / 4.1) and the probe is a few pixels
  worse (ball 24.8 against 19.5, paddle 7.7 against 6.7), inside the
  declared noise. At this budget the default replay-value path into the
  representation does nothing measurable; the imagination path of B is the
  value gradient that moved play.

## Compute context

The clean recipe replays each action 16 times. DreamerV3's current
configuration (`danijar/dreamerv3`, `configs.yaml`) uses train ratio 32 for
the full-length Atari setting (96x96 grey, sticky, all actions, B16/T64) and
256 for Atari100k (64x64 RGB, no sticky actions, the game's needed actions).
The guided learners reached 100 to 220 at the same ratio 16, with the
teacher supplying states and labels that the representation now has to find
alone. The clean arms train half to a sixteenth as much per action as their
reference, so this screen cannot tell a representation that cannot find the
ball from one that was not trained enough to; it says only that the
imagination value path, not demonstrations, is the first lever that moved
play.

## Protocol notes

- The first queue stopped at `eval-C-demos-c777001-greedy`: the runner
  expected a natural-episode evaluation end, and the greedy C policy never
  launched the ball, so all 24 episodes hit the 25,000-action cap
  (`action_cap_reached`). The remaining 20 jobs ran as `queue-2.json` with
  the evaluation end check relaxed to `{event: run_end, learner_updates: 0}`;
  no job was rerun and no result changed.
- Cells are scored on the first three natural episodes per lane; a cut
  episode is counted and excluded, which moved A by at most .1.
- The declaration said each arm's initial checkpoint is probed as its own
  floor; the queue probed only A's, on the assumption that seed 1009 gives
  every arm the same initial tensors. It does not: world and behavior tensor
  hashes differ between arms. The four missing
  initial probes were run afterwards as a repair; they match A's exactly,
  so the tensor hash differences lie outside what the probe reads.
- One seed per arm, development cohorts, linear probe only, one probe point.

## Next

[Arm B at train ratio 64](../experiments/2026-11-15-breakout-clean-ratio64.md)
separates the two: four times the updates per action, nothing else changed,
with the probe at every 100k actions. If the ball stays out of the state,
the next clean change is resolution (96x96). Two further changes need a
decision first: the game's four needed actions (would likely remove the greedy FIRE
stalls but leaves the published protocol), and demonstrations kept out of
the return normalizer or used as separate pretraining.
