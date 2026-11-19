# Breakout: does a pixel-reconstruction auxiliary sharpen the state's positions, and does play follow?

Prepared 20:33 UTC November 18; smoke run and **training launched by hand
at 20:41 UTC**, unchanged. One training run and its readings, after a
short memory smoke; nothing is chained after it. The user chose this
run ("Position precision", question of November 18) over the actor's
training signal after [the sticky check](../results/2026-11-18-breakout-sticky-check.md).

## Why

The sticky check and its post hoc readings: without the sticky draw the
agents still lose half to three quarters of the balls they lose with it.
One to two actions before contact the right action flips at about 8 px
of paddle offset (staying loses .03 to .06 inside it and .33 just
outside), the policy's action mass only changes beyond 12 px, and the
state's readable precision for that offset is 6 to 7 px: a linear
readout with 48,000 rows gives the paddle to 3.5 px and the ball's x to
3.8 px on `O22`, an MLP readout no better, against 2 to 3 px from the
emulator's coordinates. Whether the limit is the representation or the
actor's use of it is not settled; this run tests the representation.

The world objective (centered CDP on latents, reward, continuation,
eight-step open-loop supervision) never asks for a position to the
pixel. A reconstruction loss does, at least for the 16 px paddle, and
needs no label or relevance prior. `world_pixel_scale` adds an RGB
decoder to a CDP world and trains it on posterior states; the CDP
objective and its scales are unchanged. It has been run only in the
guided era and with motion weighting ([auxiliary result](../results/2026-11-10-breakout-agent-cdp-pixel-aux.md)).
Here it is plain squared error, no motion weighting, no prior-rollout
images.

## Run

`X22`: the open-loop recipe exactly as `O22` ran it (life-terminal
recipe `K`, published wrapper, 18 actions, sticky .25, eight
environments, 12M, 96x96, B16/T16, train ratio 16, replay 100k, centered
CDP, encoder rate 4e-5, behaviour rate 8e-5, replay value 1, actor unimix
.01, actor-critic gradient, life-loss terminal, `--open-loop-length 8
--open-loop-scale 1`) plus `--world-pixel-scale 1`. Seed 22003, 1.6M
actions, checkpoints every 400k. No native change.

Root `/x/Temp/kindle-breakout-pixelaux-20261118` (`make_pixelaux_queue.py`;
pinned copies in `sources.sha256`), under the reviewed host guard with
the review file of 16:55 UTC November 18 (this boot only).

1. **Smoke** (`queue-20-smoke.json`, guards `guards-20-smoke`, unit
   `kindle-breakout-pixelaux-smoke-20261118`): the training command for
   16,384 actions. It must finish, keep the 2 GiB budget headroom the
   runner enforces (`O22`'s recipe uses 16.5 of 20.9 GB without the
   decoder) and show the time per action. If it fails on memory, the run
   is not launched and goes back to the user.
2. **Run** (`queue-20.json`, 23 jobs, guards `guards-20`, unit
   `kindle-breakout-pixelaux-20261118`): training, then the 1.6M cells
   on the two reused cohorts, the 1.6M own-play dump and reset probe,
   then the remaining cells (1.6M fresh cohort; 1.2M, 800k and 400k on
   the reused cohorts) and the 1.2M dump and probe. Matched cells use the
   belief reset.

**Smoke (20:36 to 20:40 UTC):** finished, 16,384 actions and 1,001
updates, no fault. The decoder adds 1.1 GiB at the same stage (6.5 GiB
against `O22`'s 5.4), which projects to about 16.7 GiB at a full replay
against a 19.1 GiB budget, 0.4 GiB above the enforced headroom. An
update takes .194 s against `O22`'s .118 (1.6 times), so training should
take about 5.4 hours and the queue about 6. Reconstruction loss 22.2
after 1,001 updates.

**First attempt stopped (recorded 21:57 UTC November 18); not relaunched.**
The training job stopped at 333,128 actions (21:54 UTC, 73 minutes in)
on the runner's own memory check, not on a fault: usage was a steady
16.68 GiB, and another project's GPU job on the same card lowered the
budget from 19.05 to 18.66 GiB, 20 MB under the declared 2 GiB headroom.
No kernel message, no checkpoint yet (the first is at 400k), 0 of 23
jobs complete; the partial log stays in `X22-…` (online returns 3.8,
7.3, 14.5 and 21.3 per 100k actions, the last bin partial; `O22` had 6,
16, 24 and 39). A relaunch needs either a lower
training headroom, a smaller footprint or the card to itself; that
choice is the user's. `queue-20b.json` (`make_pixelaux_queue_b.py`) is
the same run from scratch as `X22b` with `--min-gpu-budget-headroom-mib
1024` on the training job only; it is prepared and **not launched**.

**Relaunch (amendment, 06:27 UTC November 19).** The user chose the lower
headroom ("Lower headroom to 1 GiB", question of November 18).
`queue-20b.json` is launched by hand unchanged: `X22b`, the same command
from scratch with `--min-gpu-budget-headroom-mib 1024` on the training
job (cells and probes keep 2048), guards `guards-20b`, unit
`kindle-breakout-pixelaux-b-20261119`. With 1 GiB the run tolerates up to
2.3 GiB of other use on the card; the most seen so far is 1.3 GiB. Tests,
baselines and readings are unchanged and are read on `X22b`. The kernel
logged a GPU resume at 00:55 UTC November 19 (power management, no fault
record); the review file still covers every fault record of the boot.

## Declared tests

At 1.6M. Baseline `O22` at 1.6M (same recipe and seed without the
decoder); `L22` is the best checkpoint on record.

1. **Precision** (own-play dump, 64,000 rows, `state_precision.py`,
   linear readout held out by lane): paddle x mean error 2.5 px or less
   (`O22` 3.5) and landing minus paddle one to four actions out 5.0 px
   or less in each of the three distance bands (`O22` 6.6 / 6.4 / 6.2).
2. **Decision** (`policy_at_contact.py`, same dump): one action before
   contact on the wide paddle, the policy's mass on still actions with
   the paddle within 5 px exceeds its mass at 5 to 12 px by .15 or more
   (`O22` .48 against .43).
3. **Ball return** (96 games, `descent_rows.py`): wide paddle .955 or
   more and narrow paddle .85 or more (`O22` .929 and .803; `L22` .950
   and .830).
4. **Score**: pooled median 400 or more, or 20 or more games past 432
   (`L22` 380.5 and 8; `O22` 317 and 0).

Readings:

- **1 fails:** squared-error reconstruction does not sharpen the state.
  Precision then needs the input or the architecture (a wider native
  input), or it is not available to a readout at all; the auxiliary is
  not retained.
- **1 passes, 2 and 3 fail:** the state holds the positions and the
  policy still does not use them. Precision was not the limit; the
  actor's training signal is next.
- **1 and 3 pass:** precision was a limit. Confirm on held-out seed 5011
  before any claim; 4 says how far it carries.

Predictions (mine, before the run): paddle 2.2 to 3.2 px; landing minus
paddle 5.0 to 6.5 px; wide .93 to .96; narrow .78 to .86; median 280 to
400; time per action 1.3 to 1.6 times `O22`'s. Test 1 passes about one
time in two, test 3 about one in four.
