# A wider model: the life-terminal recipe at model size 25m

Prepared 19:51 UTC November 19; **launched by hand at 19:51 UTC**. Follows [the 128x128 result](../results/2026-11-19-breakout-wide-input.md)
and [the encoder-precision readings](../results/2026-11-19-breakout-encoder-precision.md):
the encoder's output for one frame holds the paddle to 2.1 px and the
ball to about 5 px, the state holds the paddle worse than the encoder
does (3.5 px), and at 128x128 the encoder got sharper while the state
did not. Both losses sit in layers whose width the model-size preset
sets: 64 channels on the encoder's last grid, one 256-unit layer between
the encoder and a 32x16 categorical latent.

## Change

None in code. `--model-size 25m` is an existing preset that has not been
run on Breakout: recurrent state 3,072 (12m: 2,048), hidden 384 (256),
latent 32x24 (32x16), MLP units 384 (256), encoder channels 48 / 72 / 96
/ 96 (32 / 48 / 64 / 64). The world model has 40.3M parameters (17.9M)
and the actor and critic 3.6M (1.6M). Learning rates, batch, train ratio
and every loss scale are unchanged.

## Run

`S25`: the life-terminal recipe exactly as the 50k-replay control
`C22r50` ran it (clean centered CDP, 96x96, B16/T16, train ratio 16,
replay 50k, sticky .25, 18 actions, seed 22003) with `--model-size 25m`,
for 1.6M actions. Root `/x/Temp/kindle-breakout-size-20261119`,
`queue-23.json` (23 jobs: training, cells at 400k / 800k / 1.2M / 1.6M
on the two reused cohorts, the fresh cohort at 1.6M, own-play dumps and
reset probes at 1.2M and 1.6M), guards `guards-23`, unit
`kindle-breakout-size-20261119`, reviewed runner with this boot's review
file. Pinned `atari_vector.py` (a8c8016a, the tree's).

Smoke (19:45 to 19:48 UTC November 19, 16,384 actions, no fault): .181 s
per update against .121 s for 12m at the same stage (1.50x); peak 7.43
GiB with 16.8k frames in the replay, so about 5.5 GiB without the replay
(12m: 4.3) and about 11.3 GiB projected at replay 50k, of a budget near
19.4 GiB. Replay 100k would come to about 17.2 GiB, inside half a GiB of
the 2 GiB headroom check, so the replay stays at 50k as in `C22r50`. The
frozen tools (evaluation cell, own-play dump, reset probe) ran on the
smoke checkpoint (state width 3,840). Expected: about 5.2 hours of
training, 5.5 hours in all.

## Tests

At 1.6M, the bars of the 128x128 declaration. Baselines: `L22` at 1.6M
(same recipe, 12m, replay 100k; median 380.5, 8 of 96 games past 432,
wide .950, narrow .830, ball x 5.5 px, landing minus paddle 6.7 / 6.8 /
7.3 px) and `C22r50` at 1.2M (12m, replay 50k; median 322) as an
observation.

1. **Precision** (`state_precision.py`, own-play dump, 64,000 rows,
   linear): ball x 4.1 px or less, and landing minus paddle 5.5 px or
   less in each of the three bands up to 4 actions out.
2. **Ball return** (96 games, `descent_rows.py`): wide paddle .96 or
   more and narrow paddle .86 or more.
3. **Score**: pooled median of 96 games 410 or more, or 20 games past
   432.

Also read, as observations: the encoder-output readouts of
`encoder_precision.py` on the 1.6M checkpoint (does a wider encoder hold
the ball better per frame, and does the state now keep the paddle the
encoder has), and the state's paddle x.

Readings:

- **2 or 3 passes:** width buys play. Confirm on held-out seed 5011
  before any claim.
- **1 passes, 2 and 3 fail:** the state is sharper and the paddle does
  not use it; the actor's training signal is next.
- **All fail, and the learner is level with or ahead of `C22r50` at
  1.2M:** width is not the limit at this budget; 12m stays the default
  and the next candidate is an objective term that makes the state carry
  the encoder's output for the current frame.
- **All fail, and the learner is behind `C22r50` at 1.2M** (as the
  128x128 learner was): more parameters at these learning rates learn
  slower here, which is a statement about the recipe's optimization, and
  is examined before another capacity change.

Predictions: time per action 1.4 to 1.6 times `L22`; ball x 4.5 to 5.5
px; landing minus paddle 6.0 to 7.5 px; wide .93 to .96; narrow .76 to
.86; median at 1.6M 280 to 400; median at 1.2M 200 to 340. Each test
passes with about one chance in five. DreamerV3 reports that larger
models learn faster per step; the 128x128 learner, whose world model was
also about twice the size, learned slower. I do not know which applies.

## Limits known in advance

- One seed, one budget, a reused development seed and reused cohorts.
- The preset changes the encoder, the recurrent state, the latent and
  the actor and critic together. A pass does not say which mattered.
- 1.5 times the time per action: a pass at equal actions is not a pass
  at equal GPU time (`L22` reaches 380.5 in 3.6 hours).
- This is not the run the user chose on November 18 (the actor's
  training signal), which still has no concrete design.
