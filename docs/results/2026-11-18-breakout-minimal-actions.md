# Four actions do not change the habit; the minimal set learns slower and plays worse at 1.6M

Result of [the minimal-action declaration](../experiments/2026-11-18-breakout-minimal-actions.md).
Root `/x/Temp/kindle-breakout-minimal-20261118`, `queue-18.json`, guards
`guards-18`, unit `kindle-breakout-minimal-20261118`, 11:08 to 14:54 UTC
November 18. `M22` is the life-terminal recipe exactly as `L22` ran it with
`--atari-protocol published-minimal` (NOOP, FIRE, RIGHT, LEFT), sticky .25,
seed 22003, 1.6M actions: 200.8 minutes, 133 actions per second, 99,977
updates. CPU readings in `analysis/`, readers in `scratch-scripts/`.

**22 of 23 jobs completed.** A third GPU kernel fault (14:54:20 UTC, same
shape as the two before) stopped the last job, the own-play dump at 1.2M.
Training, all 18 cells, both probes and the 1.6M dump finished before it,
so every declared test is read on complete data. See [GPU fault](#gpu-fault).

**Result.** All four declared tests fail, and none is close. With four
actions the agent moves into contact as often as with 18 (a move among
the last two actions in .755 of wide descents; `L22` .73), returns fewer
balls (wide .919, narrow .774) and scores a pooled median of 241.5 with no
game past the first wall (`L22` 380.5 and 8). The habit does not come
from the action vocabulary. One seed, one run.

## Declared tests

At 1.6M on the two reused cohorts (96 games). Baselines at 1.6M are `L22`
(same recipe and seed, 18 actions) and `O22`; `H50` is the held-out seed
at 1.2M.

| Test | Bar | `M22` | `L22` / `O22` / `H50` | Predicted | |
|---|---|---:|---|---|---|
| 1. Wide descents with a move among the last two actions | .60 or less | .755 | .73 / .77 / .80 | .55 to .70 | fails |
| 1. Wide descents with the paddle within 8 px of the landing one action before the end | .65 or more | .528 | .57 / .47 / .41 | .50 to .65 | fails |
| 2. Wide paddle, balls returned | .965 or more | .919 (4,728) | .950 / .929 / .926 | .94 to .97 | fails |
| 3. Narrow paddle, balls returned | .87 or more | .774 (434) | .830 / .803 / .775 | .80 to .87 | fails |
| 3. Narrow paddle, fastest balls | .85 or more | .706 (252) | .765 / .730 / .721 | | fails |
| 4. Pooled median; games past 432 | 400 or 20 | 241.5; 0 | 380.5; 8 (`L22`) | 320 to 420 | fails |

Declared reading for a failed test 1: the habit does not come from the
vocabulary, and the actor's training signal is next, on the open-loop
recipe whose imagination is calibrated. I follow it. My predictions were
too favourable on every row but one.

## Play

Matched cells (belief reset at a lost life, same four-action sticky
protocol), 24 games each; pooled over the two reused cohorts and both
modes.

| Actions | `M22` cell means | `M22` pooled median | `L22` pooled median | `M22` games at 100 or more / above 300 / at 400 or more |
|---:|---|---:|---:|---|
| 400k | 34.8 to 41.4 | 35 | 73 | 0 / 0 / 0 of 96 |
| 800k | 124.0 to 158.1 | 94 | 246 | 44 / 12 / 1 of 93 |
| 1.2M | 144.7 to 208.8 | 133 | 334 | 63 / 23 / 1 of 96 |
| 1.6M | 182.9 to 253.0 | 241.5 | 380.5 | 73 / 30 / 8 of 96 |
| 1.6M, fresh cohort 4000313131 | 228.7 and 231.0 | 257 | | 36 / 17 / 2 of 48 |

- **Slower, and still rising.** Online returns per 100k actions reach 102
  at 800k, 188 at 1.1M and 235 to 250 from 1.3M (`L22`: 152, 245 and 258
  to 297). Wide-paddle returns rise from .895 at 800k to .906 and .919.
  The run is behind `L22` by several hundred thousand actions; whether it
  would catch up is not measured.
- **Greedy stalls.** At 800k, 23 of 278 greedy games ran to the
  25,000-action limit (scores from 15 up), and 2 of 221 at 1.2M;
  none at 400k or 1.6M and none in sampled play. The cell reader counts
  each stream's first three natural games, which leaves 93 games at 800k
  and passes over those stalled games, so the 800k and 1.2M medians
  flatter greedy play. In the 18-action set nine actions include FIRE;
  here one does.
- **Fewer games reach the narrow paddle:** 50 of 96 (`L22` 83, `O22` 67).
  Lives lost on the wide / narrow paddle: 382 / 98 (`L22` 251 / 219).

## The habit is unchanged

Replays of the 1.6M cell logs (`descent_rows.py`, `descent_in_place.py
--minimal`, `descent_read.py`).

| | `M22` | `L22` |
|---|---:|---:|
| Wide: move among the last two actions | .755 | .73 |
| Wide: within 8 px of the landing one / two / three actions before the end | .528 / .367 / .251 | .567 / .364 / .243 |
| Wide, paddle within 5 px three actions early: holds still, returned | .978 (224) | 1.000 (444) |
| Wide, same position: moves again, returned | .939 (651) | .938 (421) |
| Wide: share that moves again from that position | .74 | .49 |
| Wide lost balls: paddle moving toward the landing / crossed under it | .82 / .46 | .76 / .45 |
| Wide lost balls: median miss from the paddle's centre | 12 px | 11 px |
| Lost balls on a short descent or needing more than 18 px per action | 0 | 0 |
| Critic, AUC of the value's drop one action before contact | .66 | .71 |
| State probe: ball within 8 px; paddle error | .50; 7.2 px | .62 (1.2M); 4.5 to 5.8 px |

With two of four actions moving the paddle instead of twelve of 18, the
agent moves again from a paddle already under the ball more often, not
less. The lost balls have the same shape as before.

## Post hoc: two other explanations checked on the own-play dumps

Both use the 1.6M dumps of `M22` and `O22` (about 1,940 descents each)
and were not declared.

- **Not the paddle's step size** (`paddle_lattice.py`). One move action
  carries the paddle 23 px, more than the paddle's hit window, so a
  resting paddle might be unable to hold some landing points. If that
  cost balls, returns would fall where the needed displacement sits
  between multiples of 23 px. They do not: by distance to the nearest
  multiple (0 to 3, 3 to 6, 6 to 8, 8 to 10, over 10 px) the wide paddle
  returns .937, .949, .919, .929, .944 on `M22` and .980, .927, .935,
  .925, .944 on `O22`. In own play the paddle's travel between rests is
  spread over 2 to 30 px, not stacked at 23.
- **Not the state's position error** (`belief_vs_truth.py`). The linear
  readout of landing minus paddle errs by 5.7 to 8.5 px (mean absolute)
  one to four actions early, close to the spread of the agent's final
  miss. If the agent steered by what that readout holds, its paddle
  movement would follow the readout's belief rather than the truth.
  Regressing the movement on both gives the truth .63 to .89 and the
  belief .09 to .28 on the wide paddle, and a readout from the
  emulator's exact coordinates, which has no perception error, gets the
  same .08 to .26. The policy reads the position better than the linear
  probe does; the probe's error is not the agent's.

Both are observational, on one checkpoint each of two learners.

## GPU fault

At 14:54:20 UTC the kernel logged `ring gfx_0.0.0 timeout`, `Illegal
opcode in command stream`, failed MES resets, a mode-1 reset with `VRAM is
lost`, and two further resets by 14:54:38 (reset counter 4). The host
guard stopped the job ("GPU kernel fault; this boot requires explicit
recovery review") and the queue ended. Nothing was done to the device: no
reset, reload or reboot, no retry. Evidence is in
`gpu-fault-20261118-1454/` under the root (kernel records, unit journal,
the job's guard directory, and `unreviewed-fault-records.json` with the
17 fault records that no review covers).

This is the third fault on boot `a8648671`, after 12:28 UTC November 16
and 04:02 UTC November 18. All three hit a frozen-inference job (an
evaluation cell, the imagined-contact probe, an own-play dump); none hit
a training run, which hold most of the GPU time. Three events, cause not
established. **The review file of November 18 does not cover this fault;
no GPU job runs until the user reviews it.**

## Limits

- One seed, one run; development seed on reused cohorts. Checkpoint
  medians of one run differ by 50 to 100 between neighbouring
  checkpoints; the ball-return rates and the per-descent shares carry
  more weight than the medians.
- `M22` was still improving at 1.6M; "worse at 1.6M" is not "worse at
  convergence".
- The own-play dump at 1.2M is missing (the faulted job).
- Whether sticky actions are the direct cause of the misses is still not
  shown; the same checkpoints have not been evaluated without them.

## Next

Not launched. The declared next step is the actor's training signal: what
imagination pays for arriving early and holding still, on the open-loop
recipe, whose imagination is calibrated. It needs its own declaration and
the user's review of the 14:54 fault. Four actions are not proposed as
the Breakout default; that remains the user's decision.
