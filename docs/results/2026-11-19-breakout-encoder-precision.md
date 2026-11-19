# Where position precision is lost: the encoder's output holds the paddle to 2 px and the ball to about 5 px, and the latent path loses the paddle

Post hoc, CPU only, November 19, after the
[128x128 run](2026-11-19-breakout-wide-input.md). No training and no GPU
job. Root `/x/Temp/kindle-breakout-wide-20261119`: features in
`encoder-precision/`, readings in `analysis/encoder-*` and
`analysis/state-precision-mlp-L22-1600000.txt`, readers in
`scratch-scripts/` (`encoder_precision.py`, `encoder_readout.py`,
`encoder_readout_mlp.py`, `encoder_sensitivity.py`, batches
`encoder_batch.sh` and `encoder_batch2.sh`).

**Question.** Every precision reading so far was taken on the model
state (the actor's input). The state reads each object to about 3.5 px
and the offset between them to 6 to 7 px, and a planner needs 3 px or
better. Is that precision already missing in the encoder's output, or is
it lost between the encoder and the state?

**Answer.** Both, and differently for the two objects. The encoder's
output for a single frame holds the paddle to 2.1 px, better than the
state (3.5 px): the latent path loses paddle precision the encoder has.
The same output holds the ball only to about 5 px (median) in open play
and loses it among the bricks; the state reads the ball better than any
single frame because it integrates frames. At 128x128 the encoder is
sharper on every object and the state is not.

## Method

The encoder is the four-stage CNN whose flattened 6x6 grid (8x8 at
128x128) the posterior reads and the latent objective predicts. Its
forward pass was ported to NumPy from the scalar reference the GPU
encoder is tested against (`networks/rgb/tests.rs`), with weights read
from each checkpoint's `world.safetensors`. Frames come from CPU
emulator replays of two sampled frozen logs per learner at 1.6M actions
(cohorts 4000777001 and 4000022003, 4,000 ticks each, 64,000 frames),
resized with Pillow as the runner's GPU resize does. Readouts use the
ridge and lane splits of `state_precision.py`. State numbers are the
same reader on each learner's own-play dump at the same checkpoint
(64,000 other rows of the same policy).

## Readouts

Mean absolute error (median) in source pixels, linear readout.

| | `L22` encoder output | `L22` state | `W22` encoder output | `W22` state |
|---|---:|---:|---:|---:|
| Input | 96x96 | | 128x128 | |
| Features | 2,304 | 2,560 | 4,096 | 2,560 |
| Paddle x | 2.1 (1.8) | 3.5 (2.5) | 1.6 (1.3) | 5.5 (4.2) |
| Ball x | 10.4 (5.5) | 5.5 (3.8) | 6.9 (3.8) | 5.5 (3.8) |
| Ball y | 9.4 (6.4) | 4.6 (3.1) | 6.0 (3.7) | 3.7 (2.6) |

A nonlinear readout (a 512-512 MLP, lanes 0 to 4 for fitting) finds no
more in the encoder's output: `L22` ball x 12.1 (6.2), `W22` 6.3 (3.9).
Its error by where the ball is, ball x, mean (median):

| Ball | `L22` rows | `L22` | `W22` rows | `W22` |
|---|---:|---:|---:|---:|
| Above the bricks | 745 | 43.8 (37.9) | 217 | 21.5 (17.9) |
| In the brick rows | 1,498 | 33.0 (22.4) | 1,005 | 22.0 (15.4) |
| Open field | 8,084 | 7.8 (5.3) | 8,593 | 5.0 (3.7) |
| Near the paddle row | 5,527 | 8.6 (5.5) | 5,993 | 4.9 (3.5) |

The same MLP on `L22`'s state reads the paddle to 4.4 (3.5), ball x to
6.8 (4.7), and landing minus paddle one action out to 6.6; on the
emulator's exact coordinates it reads landing minus paddle to 3.0 (1.7).

## What the objective pays for a misplaced ball

The latent objective is a cosine distance between the predicted and the
actual encoder output after subtracting the batch mean. At 600 own-play
states per learner the emulator state was copied, the ball's RAM
position shifted by d pixels, and the frame rendered again; pairs were
kept when only that coordinate changed (about 1,150 of 1,200 per row).
Centered cosine distance between the two encoder outputs, mean:

| Ball shifted by | `L22`, x | `L22`, y | `W22`, x | `W22`, y |
|---:|---:|---:|---:|---:|
| 1 px | .0009 | .0005 | .0016 | .0006 |
| 2 px | .0029 | .0019 | .0038 | .0022 |
| 4 px | .0075 | .0060 | .0105 | .0060 |
| 8 px | .0250 | .0178 | .0310 | .0183 |
| 16 px | .0699 | .0539 | .0786 | .0493 |
| One action of play, for scale | .0521 | | .0331 | |
| Training loss, last 200k actions | .036 | | .057 | |

The encoder's output does change when the ball moves by a pixel, so the
position is in there in some form; the readers above do not recover it
as a coordinate below about 5 px. For the objective, a state that has
the ball 4 px off in x pays .0075, a fifth of `L22`'s remaining loss; 2
px pays .003 and 1 px .001. The last pixels are worth little to it.

## Reading

- **Paddle.** The encoder has it to 2 px and the state to 3.5 px
  (`W22`: 1.6 and 5.5). A likely reason, not measured: next-step
  prediction under sticky actions cannot reward a precise paddle much,
  because a move's end point is scattered by about 5.5 px whatever the
  state knows. The pixel
  auxiliary of [`X22b`](2026-11-19-breakout-pixel-auxiliary-precision.md)
  brought the state's paddle to 2.3 px with the same architecture, so
  the path can carry it when the objective asks.
- **Ball.** One frame's encoder output places the ball to about 5 px in
  open play at 96x96 and 3.5 to 3.7 px at 128x128, and not at all among
  the bricks. The state's 3.8 px median comes from integrating frames.
- **128x128.** More pixels per ball did sharpen the encoder's output, as
  the wide-input declaration assumed. The state did not follow, and the
  learner was far behind at equal actions.

## Limits

- The NumPy encoder follows the tested reference line by line but was
  not compared with native output on these frames.
- "Not recovered by a ridge or a small MLP" is not "absent".
- Encoder rows come from frozen logs and state rows from own-play dumps:
  the same policy and checkpoint, different games.
- One seed per learner, and `W22` is the less trained of the two.
- The paddle's share of the objective was not measured (its RAM position
  cannot be shifted the way the ball's can).

## Next

Both losses sit in layers whose width the model-size preset sets (64
channels on the encoder's last grid; one 256-unit layer into a 32x16
latent). [The same recipe at model size 25m](../experiments/2026-11-19-breakout-model-size.md)
is declared separately. If width does not help, the candidate this
reading points to is an objective term that makes the state carry the
encoder's output for the current frame.
