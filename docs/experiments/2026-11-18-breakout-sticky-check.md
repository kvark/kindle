# Breakout: are the lost balls the sticky draw? The same checkpoints evaluated without sticky actions

Prepared 16:56 UTC November 18. Evaluation only; nothing is trained and
nothing is chained after it. The user chose this check before the next
training run (question of November 18, "Sticky check, then actor signal").

## Why

[The lost-ball replays](../results/2026-11-18-breakout-open-loop-contact.md)
and [the minimal-action run](../results/2026-11-18-breakout-minimal-actions.md)
leave one question open. The agents lose balls with a paddle that is
still moving at contact, one move carries the paddle 23 px, and sticky
actions drawn per emulator frame scatter where a move ends (spread
5.5 px). Whether that scatter is the direct cause of the misses has not
been measured: no checkpoint trained under sticky actions has been
evaluated without them. A planner with the emulator's exact state never
loses a ball without sticky actions and returns .98 (wide) and .90
(narrow, fastest) with them.

## Run

Four existing checkpoints, each evaluated exactly as its sticky cells were
(matched belief reset, same wrapper and action set, cohorts 4000777001
and 4000022003, sampled and greedy, 24 games per cell, 96 games per
checkpoint) with one change: `--sticky-actions 0`.

| Learner | Checkpoint | Action set |
|---|---|---|
| `L22` (best on record) | 1.6M | 18 |
| `O22` (open-loop) | 1.6M | 18 |
| `H50` (held-out seed 5011) | 1.2M | 18 |
| `M22` (minimal) | 1.6M | 4 |

Root `/x/Temp/kindle-breakout-nosticky-20261118`, `queue-19.json` (16
cells, `make_nosticky_queue.py`; pinned copies in `sources.sha256`),
guards `guards-19`, unit `kindle-breakout-nosticky-20261118`, under the
reviewed host guard with the review file of 16:55 UTC November 18
(`gpu-fault-20261118-1454/reviewed-kernel-faults.json` in root
`/x/Temp/kindle-breakout-minimal-20261118`: the 35 fault records of this
boot, which the user reviewed; this boot only). Then CPU replays of the
logs with the readers used for the sticky cells (`descent_rows.py`,
`descent_in_place.py`, `descent_read.py`), which take the sticky
probability from each log. A game ends at 100,000 frames, so a learner
that stops losing balls shows as truncated games; those are counted and
reported, not dropped.

These agents were trained under sticky actions; the evaluation changes
their dynamics as well as removing the draw. A gain is read as "the
draw costs this much to this policy", not as a policy that would have
been learned without sticky actions. The user's gate is under sticky
actions and is not changed by this check.

## Declared tests

Per learner, without sticky actions against its sticky cells on the same
cohorts. Sticky values for `L22` / `O22` / `H50` / `M22`.

1. **Wide paddle** returned (.950 / .929 / .926 / .919). *Direct cause*:
   .975 or more on at least three learners. *Not the cause*: a rise of
   less than .015 on at least three.
2. **Narrow paddle** returned (.830 / .803 / .775 / .774). *Direct
   cause*: .90 or more on at least three. *Not the cause*: a rise of
   less than .03 on at least three.
3. **Score**: pooled median and games past 432 or truncated at the frame
   limit (sticky: 380.5 and 8 for `L22`; 317 and 0; `H50` 276 over its
   six cells; 241.5 and 0). Reported, no bar.
4. **Habit**: share of wide descents with a move among the last two
   actions (.73 / .77 / .80 / .755). Reported, no bar; I expect no change.

Readings, which shape the next declaration but do not replace it (the
actor's training signal on the open-loop recipe is next either way):

- **1 and 2 direct cause:** the policy's aim is good enough and leaves no
  margin for the draw. The next run is judged on margin: arriving early
  and holding, under sticky actions.
- **1 and 2 not the cause:** the policy misses on its own, with no noise
  to blame. The next run is judged on precision of the final position,
  and the sticky scatter is dropped as an explanation.
- **Between or mixed:** both contribute; report the split by paddle width
  and say so.

Predictions (mine, before the run): wide .955 to .98, narrow .84 to .92,
medians higher by 30 to 100; the direct-cause bar on the wide paddle met
on three learners about one time in three.
