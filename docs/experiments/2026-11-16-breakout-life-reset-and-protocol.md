# Matched life-reset evaluation and the Atari-100k protocol on the clean CDP Breakout agent

The first arm of the [life-loss terminal experiment](2026-11-15-breakout-life-loss-terminal.md)
(`M22`, seed 22003, train ratio 64) finished on November 16 and misses its
declared criteria. Frozen whole-game play at 400k is 33.4 / 36.1 / 43.1 /
26.5 (pooled median 28; the criterion was 54, the control's median is 27),
and the ball probe reads within 8 px .29 and .25 at 300k and 400k (criterion
.4, control .08 to .10). Online, where the boundary was active, it earned
57, 53 and 51 per game over the last 150k actions against the control's 28,
30 and 40. That verdict stands as declared. This declaration follows two
things the frozen logs show when their recorded actions are replayed in the
emulator (`breakout_life_replay.py`, CPU only, every reward checked against
the log).

**1. The declared evaluation left the agent's training distribution after
the first lost life.** Training restarted the agent's belief at every lost
life; evaluation deliberately did not. Pooled over the 96 frozen games at
400k:

| Life | M22 return | M22 actions | M22 balls returned | Seed-1009 control returned | Seed-22003 control returned |
|---|---:|---:|---:|---:|---:|
| 1 | 10.1 | 334 | .94 | .59 | .46 |
| 2 | 5.7 | 161 | .89 | .79 | .66 |
| 3 | 5.9 | 158 | .86 | .84 | .81 |
| 4 | 5.4 | 141 | .85 | .87 | .81 |
| 5 | 7.7 | 123 | .72 | .85 | .78 |

"Returned" is the share of ball descents that the paddle sends back. M22
plays its first life far better than either control plays any life, then
drops to their level once its belief carries on through a loss it never saw
continue; a third of its later lives score nothing. The controls fail the
other way round: they lose the first life of every game in 49 to 74 actions,
with the paddle standing still at 71 and 75 of 96 misses, and after their
first return of the game they send back 8% and 34% of balls.

**2. In its first life M22 is limited by the ball speeding up, not by the
slow ball.** By paddle hits so far in the life, M22's first-life return rate
is .97 at the serve, .99 after one to three hits, .93 after four to eleven
and .81 from twelve on; Breakout speeds the ball up after the fourth and
twelfth hit. Descents faster than 13 source pixels per action are returned
65% of the time over all lives. When it misses, the paddle is a median 25 px
from the ball and is moving away from it in half the cases.

Every run so far used sticky actions (probability .25) and all 18 actions.
The published scores the goal is measured against did not: BBF's
configuration states that Atari 100k "doesn't use sticky actions"
(`sticky_actions = False`), IRIS and DIAMOND build `NoFrameskip`
environments with up to 30 no-op starts, and DreamerV3's released Atari-100k
scores came from a non-sticky wrapper (this repository's `atari.py` records
both of its profiles). Those environment IDs also default to each game's
minimal action set, four actions in Breakout; that default was not re-read
today. A sticky action repeats the previous command for a random number of
frames, which costs most exactly where M22 fails: placing the paddle under a
fast ball. Kindle's sticky .25 protocol stays the retained protocol for the
five-game results; the arms below are a declared diagnostic of how much of
the gap to the published numbers it explains, and adopting it is the user's
decision.

## Part A: matched evaluation of existing checkpoints

**Change.** `--life-loss-reset`, frozen evaluation only (the runner rejects
it without `--restore --evaluate`). At a lost life the agent's belief
restarts from the same frame through the same two calls training used
(`observe` with the terminal, then `begin_episodes`). The emulator, rewards
and logged episodes stay whole games, and the policy still never sees the
lives counter. The header records `raw-frame-life-loss-belief-reset-v1` and
the audit accepts it only on a frozen restore. This is the deployment that
matches how the agent was trained; IRIS and DIAMOND instead carry their
recurrent policy state through the loss at test time, so both evaluations
are reported.

**Cells.** Sampled and greedy, three natural episodes per lane on cohorts
4000777001 and 4000022003, published protocol with sticky .25 as before,
with the reset added:

| Checkpoint | Points | Purpose |
|---|---|---|
| `M22` (life terminal, seed 22003) | 100k, 400k | matched play |
| `M10` (life terminal, seed 1009) | 100k, 400k | matched play, second seed |
| `K` (life terminal, ratio 16) | 400k | matched play at the cheap ratio |
| `L` (no life terminal, ratio 16), seed-22003 and seed-1009 controls | 400k | the reset applied to agents never trained with it |

32 evaluations, `queue-10-eval.json`, in the order M22 400k, the two
controls, then the rest (the M10, K and L checkpoints are written by the
still-running `queue-9.json`).

**Reading.** This evaluation was chosen after seeing M22's result, so it
does not rescue the declared verdict; it is a second measurement.

- The life boundary helps play on a seed, under matched evaluation, if the
  M arm's pooled 400k median is at least twice its control's whole-game
  median (54 for seed 22003, 60 for seed 1009) and every cell mean is above
  the control's. Predicted before running, from M22's first-life play:
  greedy means 60 to 90, sampled 40 to 60.
- The controls should get worse under the reset (every life then starts
  like their first). If they improve instead, the explanation in point 1 is
  wrong and the belief's age matters in some other way.
- K against L keeps its declared criterion, read on K's matched cells.

## Part B: the Atari-100k protocol

**Withdrawn before launch on November 16.** Part A showed the ratio-16
life arm playing at least as well as the ratio-64 one on this seed, so the
protocol arms run at ratio 16 instead:
[ratio-16 protocol declaration](2026-11-16-breakout-ratio16-protocol.md).
`queue-10-train.json` was never started. The text below is kept as
declared.

Both arms are M22's recipe and seed (22003, the seed that does not pick up
the ball by itself) with the life terminal: 12M, 96x96 RGB, B16/T16,
replay 100k, train ratio 64, actor unimix .01, behavior rate 8e-5, replay
value 1.0, `--actor-critic-gradient`, `--cdp --cdp-centered
--encoder-learning-rate 4e-5`, 400k actions, 8 environments, checkpoints
every 100k. No guidance, labels, demonstrations, motion weighting, context
frame or bonus.

| Arm | Sticky | Actions | No-op starts | Changes against M22 |
|---|---:|---:|---:|---|
| `P22-cdp-life-atari100k-96-s22003` | 0 | 4 | up to 30 | the Atari-100k wrapper (`--atari-protocol current`) |
| `S22-cdp-life-nosticky-96-s22003` | 0 | 18 | none | sticky actions only |

P22 runs first: it is the protocol the published agents report, so it is
the direct test of the goal. S22 separates sticky actions from the action
set and no-op starts. About 3.3 hours each.

**Measures.** For each arm: matched frozen play (life reset) under its own
protocol at 100k and 400k on both cohorts, sampled and greedy; at 400k also
without the reset and, with the reset, under sticky .25 (robustness to the
retained protocol). Linear state probe at every kept checkpoint; P22 uses
`breakout_agent_probe_actions.py --minimal-actions`, which feeds the same
teacher trajectories through the four-action vocabulary (checked on CPU:
identical frames and rewards over 6,000 actions and 34 games). Per-life and
per-descent statistics from the frozen logs. Online returns, entropy and
return scale per 50k actions.

**Reading.**

- Atari-100k budget: the 100k checkpoint is compared with the published
  scores at that budget (DreamerV3 31, IRIS 84, DIAMOND 133) as cell means
  under the arm's own protocol. P22 above 31 in all four cells is parity
  with the Dreamer baseline; above 84 in all four is the first Kindle
  result that plays Breakout at the level of the published agents that
  play it well.
- Against M22 at 400k (matched cells from Part A): the protocol helps if
  the arm's pooled median is at least 1.5 times M22's and every cell mean
  is above M22's. S22 passing means sticky actions alone cost that much;
  P22 passing with S22 failing puts it on the action set and no-op starts.
- Mechanism: the first-life return rate from twelve hits on (M22 .81) and
  for descents faster than 13 px per action (M22 .65). If the protocol
  arms play better without these rising, the explanation in point 2 is
  wrong.
- Cells under sticky .25 show what the arm keeps when the retained
  protocol is applied to it at test time. They do not replace training
  under that protocol.
- If neither arm passes, the fast ball is the limit under either
  protocol, and the next step is what the state knows about the ball's
  motion (two-frame encoder input, `--rgb-frame-pair`) or acting on a
  shorter action repeat, declared separately.

**Run.** Root `/x/Temp/kindle-breakout-protocol-20261116`; runner, audit,
probe variant and replay analysis from this commit, copied to the root with
`sources.sha256`. Kindle on Meganeura main `e3215fa2`, unchanged since the
life-loss experiment. Nothing is chained to the running queue: after
`kindle-breakout-life-20261115` finishes (about 06:45 UTC on November 16),
`queue-10-eval.json` (guards `guards-10-eval`, about half an hour) is
launched as a 15 GiB memory-capped unit and read, then
`queue-10-train.json` (guards `guards-10-train`, about seven hours).

One run per arm on one seed and reused development cohorts: a screen. A
passing arm needs a second seed before any reliability claim, and the
original 864 / 90% two-wall gate is not addressed by any of this.

## Amendment before launch: the state probe carries the belief through lost lives too

Added on November 16 after the second life-loss arm (`M10`, seed 1009)
finished and before either queue above was started. No declared job
changes; jobs are appended.

M10 misses its declared criteria the same way M22 did: frozen whole-game
play 28.5 / 29.7 / 30.9 / 33.8 at 400k (pooled median 25.5 against the
criterion's 60; its control plays 71 to 91), ball within 8 px .26 and .18
at 300k and 400k against its control's .67 and .51. Its replayed frozen
games repeat M22's pattern more sharply: the first life returns 95% of
balls and scores 17.8 in 449 actions, the second life scores nothing in
82 of 96 games, and in later lives the serve is returned 48% of the time
against 100% in the first life. Online, with the boundary active, it
earned 67 and 94 per game over its last two 50k bins and never fell back
(its control peaked at 124 near 150k and fell to 59).

The pinned state probe (`breakout_agent_probe.py`) has the same mismatch as
the whole-game evaluation. Its teacher loses three or four lives per lane
in 1,536 actions (counted on CPU on the probe's own seeds), so 71% of the
probe's rows come after a lost life with the belief carried through it. The
probe criterion of the life-loss declaration was therefore read mostly
outside the M arms' training distribution, and M10's apparent loss of the
ball may be that and not a cost of the boundary.

**Change.** `breakout_agent_probe_life.py`, a copy of
`breakout_agent_probe_actions.py` with `--life-loss-reset`: when the teacher
loses a life, the agent's belief restarts from the same frame (`observe`
with the terminal, then `begin_episodes`). Trajectories, labels, splits and
the regression are unchanged; lives lost per lane are recorded. Checked on
CPU with a stub agent: with and without the flag, and through the
four-action vocabulary, the teacher's returns and the 29 lost lives are
identical, and every ended lane is restarted before the next action.

**Added jobs** (`make_probe_additions.py` in the root; 44 and 50 jobs):

- `queue-10-eval.json`, after its 32 evaluations: reset probes of M22 and
  M10 at 100k to 400k, of K and L at 400k, and of the two ratio-64
  controls at 400k (agents never trained with the reset), into
  `reset-probe/`.
- `queue-10-train.json`, after all 42 declared jobs: each arm's four
  checkpoints probed with the reset (`--minimal-actions` for P22).

**Reading.** Post hoc like Part A: it does not change the declared probe
verdicts. The boundary costs representation on seed 1009 only if M10's
reset probe stays under .4 within 8 px at 300k and 400k; it repairs seed
22003 if M22's reset probe passes .4 there. Predicted before running: M10's
vertical ball error falls under 10 px (from 16.5 and 17.8) and both M arms
read within 8 px above their carried-belief probes; the controls change
little or get worse. For M10's matched play, predicted from its first life:
cell means 70 to 100 and a pooled median above 60.
