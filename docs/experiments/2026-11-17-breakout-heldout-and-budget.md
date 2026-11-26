# Declaration: a held-out seed, a fresh cohort and a 3.6M-action run of the clean life-terminal recipe

November 17, 02:25 UTC. Follows [the 1.2M-action budget result](../results/2026-11-16-breakout-ratio16-budget.md).

## Why

The 1.2M-action run meets its own median marker on two development
seeds (334 and 350), but that does not show Kindle plays Breakout well
by the project's standard:

- both seeds and both evaluation cohorts were reused from earlier
  screens; nothing held out has been run at this budget;
- only 3 of 192 games clear the first wall, and the project's Breakout
  gate (864 points in 90% of games) is unfinished;
- every curve was still rising at 1.2M, and the late-game return rate
  that limits the wall is rising with it. Returned descents by the
  game's score at the time, seeds 22003 / 1009:

  | Score so far | 800k | 1.2M |
  |---|---|---|
  | 300 to 350 | .78 / .74 | .84 / .86 |
  | 350 to 400 | .70 / .66 | .84 / .82 |
  | 400 to 432 | .76 / .77 (45 and 34 descents) | .81 / .78 |

This is one recipe at a longer budget, justified by those rising rates,
plus one held-out seed; it is not a seed campaign.

## What runs

Recipe unchanged from `K` / `B22` (12M, 96x96, B16/T16, train ratio 16,
replay 100k, centered CDP, `--actor-critic-gradient`,
`--life-loss-terminal`, sticky .25, 18 actions, no guidance or labels).
Evaluation is matched (`--life-loss-reset`), 24 games per cell, sampled
and greedy. Root `/x/Temp/kindle-breakout-heldout-20261117`,
`queue-13.json` (59 jobs, built by `make_heldout_queue.py` there),
guards `guards-13`, unit `kindle-breakout-heldout-20261117`, launched by
hand with `run_gpu_queue_reviewed.py` on the reviewed boot. About 10.6
hours. Nothing is chained after it.

1. **Fresh cohort.** `B22` and `B10` at 1.2M on cohort 4000313131,
   which no Breakout learner has been evaluated on (4 cells).
2. **Held-out seed.** `H50`: training seed 5011, not used by any
   recorded Breakout run, 1.2M actions (about 150 minutes), checkpoints
   every 400k; cells at 400k, 800k and 1.2M on the fresh cohort and the
   two reused ones (18 cells); reset probe at 1.2M.
3. **Longer budget.** `L22`: seed 22003 for 3.6M actions (about 450
   minutes), checkpoints every 400k; cells at 1.2M (identity check
   against `B22`), 1.6M, 2.0M, 2.4M, 2.8M, 3.2M and 3.6M on the two
   reused cohorts, plus the fresh cohort at 2.4M and 3.6M (32 cells);
   reset probes at 2.4M and 3.6M. Replays on the CPU afterwards.

## Reading

- **Fresh cohort.** The marker holds off the reused cohorts if each
  learner's 48 games have a median of at least 150. Predicted cell means
  250 to 350.
- **Held-out seed.** The marker holds on a held-out seed if `H50`'s
  pooled median at 1.2M is at least 150 over all six cells (144 games)
  and over the fresh cohort alone (48 games). Predicted: median 250 to
  380 at 1.2M and 50 to 80 at 400k. If it fails, the 1.2M result is
  limited to its two development seeds and says so.
- **Longer budget.** More budget still pays on the wall if at 3.6M the
  pooled median on the reused cohorts (96 games) is at least 400 **or**
  at least 24 of the 96 games pass 432. Mechanism target: descents
  returned in the 400 to 432 band at least .90 (1.2M: .81). If the
  median is under 380 and fewer than 10 games pass 432, more budget of
  this recipe is not the lever for the wall, and the next step is the
  end of the wall itself (what the agent is paid for the last bricks),
  declared separately. Predicted: median 380 to 420, 10 to 30 games past
  the wall, best game 500 to 700, band return .85 to .92, fast balls
  returned .92 to .95.
- **The project gate.** The number of games at 864 is reported for every
  cell. Nothing here is expected to meet 864 in 90% of games; the gate
  stays unfinished unless it does.

Named risks: a late fall in online returns as the 100k replay holds
only recent play; a second GPU fault on this boot, which stops the queue
and is not covered by the existing review. `L22`'s 1.2M checkpoint is
compared with `B22`'s by hash; the checkpoint interval differs (400k
against 200k) and is expected not to matter.
