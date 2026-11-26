# Breakout under the minimal action set: does the agent stop moving at contact?

Prepared and committed 11:08 UTC November 18; **launched by hand at 11:08
UTC**, unchanged. One training run and its readings; nothing is chained
after it.
[Result](../results/2026-11-18-breakout-minimal-actions.md).

## Why

[The open-loop run](2026-11-18-breakout-open-loop-contact.md) (`O22`)
calibrated imagination and left play where it was. A CPU replay of the
frozen evaluation logs of three learners (`L22` and `O22` at 1.6M, held-out
`H50` at 1.2M; 96 games each) then sorted every lost ball:

- More than half of all lives are lost on the **wide** paddle (251 of 470,
  340 of 480, 352 of 480), where the perfect-knowledge planner returns .98
  and the agents .950, .929 and .926.
- No lost ball was out of reach, on a short descent, or (beyond the 9% to
  14% also seen on returned balls) redirected late. The paddle is a median
  33 to 40 px from the landing point when the ball turns down, with nine
  to twelve actions to cover it.
- The paddle arrives late and moving. One action before the end it is
  within 8 px of the landing point in .41 to .57 of wide descents; the
  agent moves in one of its last two actions in .73 to .80 of them; a
  lost ball misses the paddle's centre by a median 11 to 12 px, 3 to 4 px
  past its edge, with the paddle moving toward it in three quarters of
  the cases and past it in half.
- A paddle within 5 px of the landing point (4 px on the narrow paddle)
  three actions early that
  **holds still** returns .99 to 1.00 on the wide paddle and .96 to 1.00
  on the narrow one. From the same position the agent moves in about
  half the cases or more and then returns .91 to .96 (wide) and .70 to .79
  (narrow).
- The state knows where the ball lands early enough: a held-out linear
  readout gives the landing x to a median 3.1 to 4.2 px from one to five
  actions before contact on `O22` (2.7 to 6.2 on `L22`), against a 3 px
  floor of the label itself, while the paddle is still a median 27 to
  28 px away at five actions.
- One move action carries the paddle 23 px. Under sticky actions drawn per
  emulator frame, a planned move ends off-plan in 43% to 51% of trials
  (spread 5.5 px; 12 px or more off in about one in ten).

So the lost lives are a habit of the policy, not missing knowledge: it
keeps moving into contact and the sticky draw decides the rest. Twelve of
the 18 actions move the paddle and six leave it still; any exploration
or indecision is two to one a 23 px move. Every published agent that
plays Breakout well from little data uses the game's four actions. The
one earlier arm with four actions also removed sticky actions and so
tested something else ([protocol arms](../results/2026-11-16-breakout-ratio16-protocol.md):
at 400k it had 16 of 96 games above 300 against none for its 18-action
non-sticky sibling). `--atari-protocol published-minimal` changes only
the action vocabulary. It has not been run with sticky actions.

## Run

`M22`: the life-terminal recipe `K` exactly as `L22` ran it (published
wrapper, sticky .25, eight environments, 12M, 96x96, B16/T16, train
ratio 16, replay 100k, centered CDP, encoder rate 4e-5, behaviour rate
8e-5, replay value 1, actor unimix .01, actor-critic gradient, life-loss
terminal; no open-loop term) with `--atari-protocol published-minimal`
(NOOP, FIRE, RIGHT, LEFT), seed 22003, 1.6M actions, checkpoints every
400k. No native change.

Root `/x/Temp/kindle-breakout-minimal-20261118`, `queue-18.json` (23 jobs,
`make_minimal_queue.py`; pinned copies in `sources.sha256`), guards
`guards-18`, unit `kindle-breakout-minimal-20261118`, about 4.2 hours:
training (about 200 minutes), 18 matched cells with the belief reset
under the same four-action sticky protocol (1.6M on the two reused
cohorts and the fresh one; 1.2M, 800k and 400k on the reused cohorts),
reset probes and own-play dumps at 1.2M and 1.6M. Launched by hand under
the reviewed host guard (review file of November 18, this boot only).

Whether four actions become the Breakout default is the user's decision;
this run only measures it.

## Declared tests

At 1.6M on the two reused cohorts (96 games), read from the cell logs
with `descent_rows.py` and `descent_in_place.py --minimal`. Baselines are
`L22` at 1.6M (same recipe and seed, 18 actions), then `O22` and `H50`.

1. **Behaviour.** The agent moves in one of its last two actions in .60
   or less of wide-paddle descents (`L22` .73, `O22` .77, `H50` .80), and
   the paddle is within 8 px of the landing point one action before the
   end in .65 or more (.57, .47, .41).
2. **Wide paddle.** Returned .965 or more (.950, .929, .926; planner .98).
3. **Narrow paddle.** Returned .87 or more (.830, .803, .775), fastest
   balls .85 or more (.765, .730, .721; planner .90).
4. **Score.** Pooled median of 96 games 400 or more, or 20 or more games
   past 432 (`L22` 380.5 and 8).

Readings:

- 2 or 3 passes together with 4: the action vocabulary was a limit.
  Confirm on held-out seed 5011 and put the default to the user.
- 1 passes, 2 and 3 fail: the agent holds still more and loses as many
  balls; the reading above is wrong.
- 1 fails: the habit does not come from the vocabulary. The actor's
  training signal is next (what imagination pays for arriving early),
  on the open-loop recipe whose imagination is calibrated.

Checkpoint medians of one run have differed by 50 to 100 between
neighbouring checkpoints on this recipe, so tests 1 to 3, which count
thousands of descents, carry more weight than test 4.

## Predictions

Moves in the last two actions .55 to .70; in place one action early .50
to .65; wide .94 to .97; narrow .80 to .87; median 320 to 420. I expect
test 1 to be close, a small gain on the wide paddle, and no clear pass
of test 4; a pass of 2 and 4 I put at about one in four.

## Risks

- One seed, development cohorts; the baseline's 1.6M checkpoint is its
  best of nine.
- Fewer actions also change exploration early in training; a difference
  at 400k is not the contact habit.
- A third kernel GPU fault would stop the queue; it needs a new review by
  the user and nothing is reset.
