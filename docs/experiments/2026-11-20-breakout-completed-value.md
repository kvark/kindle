# The actor's training signal: a critic whose replayed targets are the real return of the life

Prepared 02:43 UTC November 20; **launched by hand at 02:44 UTC**. Follows
[the critic-calibration reading](../results/2026-11-20-breakout-critic-calibration.md):
on every trained checkpoint the critic's mean value is 1.8 to 6.1 times
the discounted reward the life goes on to earn (87.8 against 30.5 on the
best one), and the actor is trained on advantages measured against that
critic inside imagination. This is the run the user chose on November 18
(the actor's training signal), now with a concrete design.

## Change

Runner only: `atari_vector.py` no longer refuses `--replay-value-completed`
together with `--life-loss-terminal`. The native option is unchanged
(`replay_value_completed`, November 6): for each replayed row whose
episode has ended in the replay, the critic's replay target is the
discounted sum of the stored rewards to that end, with no bootstrap; rows
whose episode has not ended keep the default target. With the life-loss
terminal an episode is a life, so the target is the quantity the critic
is already defined to predict. The imagined value loss, the actor loss
and every scale are unchanged. No rebuild; no new tensor.

## Run

`V22`: the life-terminal recipe exactly as the 50k-replay control
`C22r50` ran it (clean centered CDP, 12m, 96x96, B16/T16, train ratio 16,
replay 50k, sticky .25, 18 actions, seed 22003) with
`--replay-value-completed`, for 1.6M actions. Root
`/x/Temp/kindle-breakout-realvalue-20261120`, `queue-24.json` (23 jobs:
training, cells at 400k / 800k / 1.2M / 1.6M on the two reused cohorts,
the fresh cohort at 1.6M, own-play dumps and reset probes at 1.2M and
1.6M), guards `guards-24`, unit `kindle-breakout-realvalue-20261120`,
reviewed runner with this boot's review file. Pinned `atari_vector.py`
(bab85e21).

Smoke (02:40 to 02:42 UTC November 20, 16,384 actions, no fault): 154 to
236 of the 240 replayed rows in a batch carry a completed return, .114 s
per update (`C22r50` .121 at the same stage), peak 5.22 GiB at 16.7k
frames, so memory is `C22r50`'s (9.9 GiB projected at replay 50k).
The frozen tools (evaluation cell, own-play dump, reset probe) ran on
the smoke checkpoint. Expected: 3.4 hours of training, 3.9 hours in all.

## Tests

At 1.6M. Baselines: `L22` at 1.6M (same recipe without the option,
replay 100k) and `C22r50` at 1.2M (replay 50k).

1. **Calibration** (`value_td.py`, own-play dump): mean value divided
   by mean real return of the life 1.5 or less (`L22` 2.9; `C22r50` at
   1.2M 2.2; all eighteen existing dumps 1.8 to 6.1). This is the check
   that the option does what it says in this recipe.
2. **The critic prices a loss before contact**
   (`value_drop_at_contact.py`, all descents, one action before
   contact): AUC .85 or more (existing learners .65 to .78; a linear
   readout of the state .83 to .89).
3. **Ball return** (96 games, `descent_rows.py`): wide .96 or more and
   narrow .86 or more (`L22` .950 and .830).
4. **Score:** pooled median 410 or more, or 20 or more games past 432
   (`L22` 380.5 and 8).

Observations, no bar: the share of descents with the paddle within 8 px
of the landing two actions before the end (`L22` .364 wide, .360
narrow); how often a paddle already in place holds still; the 1.2M
checkpoint against `C22r50`'s 322; the mean imagined return in the
training log against the value the replayed reward and continuation
rates imply.

Readings:

- **Test 1 fails:** at scale 1 the real targets do not hold the critic's
  level against the imagined loss. The run then says nothing about
  whether a calibrated critic plays better; a stronger anchor (a larger
  replay-value scale, or the posterior bootstrap) would be declared
  separately.
- **1 passes, 2 fails:** the level is fixed and the timing is not; the
  critic still learns of a loss at contact. Then what it lacks is
  resolution at the paddle, not a target, and the completed-return
  actor is next only as a way round the critic.
- **1 and 2 pass, 3 and 4 fail:** the critic knows in time and the
  actor does not use it. The actor objective itself is next
  (`--replay-actor-scale`, which a calibrated critic makes usable).
- **3 or 4 passes:** confirm on held-out seed 5011 before any claim.

A median below 250 at 1.6M means the option costs play at this budget
and it is not retained, whatever the other tests say.

Predictions: ratio 1.2 to 2.0 (test 1 passes with three chances in
five); AUC .70 to .85 (one in three); wide .93 to .96, narrow .78 to
.87 (one in six); median 250 to 400 (one in eight); time per action
within 5% of `C22r50`.

## Limits known in advance

- One seed, one checkpoint per test. The reference is noisy: `L22`
  scores 334, 380.5 and 331.5 at 1.2M, 1.6M and 2.0M, and the 25m
  learner's checkpoints 400k apart moved by a factor of two. The first
  two tests are aimed at quantities the change should move by more than
  that; the score test is not protected.
- Completed returns are single outcomes of the replayed policy: the
  real return minus the value has a standard deviation of 42 points on
  `L22` at 1.6M, and the replay holds the last 50,000 actions.
- The imagined value loss is unchanged, so the critic is pulled two
  ways on replayed states and may settle between.
- The replay-value loss is also a term of the world model's objective
  (replay-value scale 1.0), so the representation's gradient changes
  with the critic's target; the run does not separate the two.
- The one earlier trial of this option (November 6, guided whole-game
  recipe, 200k actions) scored above its control with intervals that
  spanned zero.
