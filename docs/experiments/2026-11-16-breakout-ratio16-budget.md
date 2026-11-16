# Ratio-16 life-terminal recipe at three times the budget, both seeds

Follows [the ratio-16 protocol result](../results/2026-11-16-breakout-ratio16-protocol.md).
Neither protocol change passed, the cheap recipe's play differs 1.6 times
between its two seeds at 400k actions (cell means 90 on seed 22003, 57 on
seed 1009), and every online return curve was still rising at the end
(50 to 84 per game over the last 50k actions). The fast ball separates
the arms, and nothing tried so far shows whether more of the same
training closes it. This run buys that answer directly.

## Arms

`K`'s recipe unchanged except the budget: clean centered CDP, 12M, 96x96
RGB, B16/T16, replay 100k, train ratio 16, actor unimix .01, behavior rate
8e-5, replay value 1.0, `--actor-critic-gradient`, `--cdp --cdp-centered
--encoder-learning-rate 4e-5`, `--life-loss-terminal`, sticky .25, 18
actions, 8 environments. **1.2M actions** (74,977 updates, about 150
minutes each), checkpoints every 200k. No guidance, labels,
demonstrations, motion weighting, context frame or bonus.

| Arm | Seed | Earlier 400k run of the same seed |
|---|---:|---|
| `B22-cdp-life-ratio16-96-1200k-s22003` | 22003 | `K`: 80.7 to 101.7, median 73 |
| `B10-cdp-life-ratio16-96-1200k` | 1009 | `K10`: 52.8 to 60.9, median 54 |

These are fresh lives, not extensions: checkpoints hold no replay, so the
400k runs cannot be continued. The first 400k actions of each arm repeat
`K` and `K10` in everything but the run, which also measures how much of
the seed spread is the run itself.

**Measures.** Matched frozen play (`--life-loss-reset`, three natural
episodes per lane, cohorts 4000777001 and 4000022003, sampled and greedy)
at 400k, 800k and 1.2M. Reset state probe at the same three checkpoints,
run after everything else. Per-life replay of the matched logs at each
point. Online returns, entropy and return scale per 50k actions.

**Reading.**

- The budget pays on a seed if its matched pooled median at 1.2M is at
  least 1.5 times its own 400k median and every 1.2M cell mean is above
  the same cell at 400k. It pays for the recipe if that holds on both
  seeds.
- Breakout is learned well by this recipe if both seeds reach a pooled
  median of at least 150 at 1.2M. Reported beside it: games above 300 and
  the first game above 432 (first wall cleared; best so far 415).
- Predicted: cell means 120 to 250 and pooled medians 100 to 180 at 1.2M;
  descents at 13 px per action or more returned .75 to .85 (`K` .76, `K10`
  .54); at least one game above 432.
- Repeat at 400k: each arm's 400k cells against the earlier run of the
  same seed. Reported, no threshold.
- Named risk: the seed-1009 no-boundary control at ratio 64 peaked near
  200k actions and fell as its return scale and entropy rose. A drop
  between 800k and 1.2M with a rising return scale would repeat that
  under the life boundary.
- If the fast-ball return rate stays under .7 on both seeds while slow
  balls stay above .95, more training does not buy precision and the next
  declared lever is control or input resolution for the fast ball
  (shorter action repeat, finer frames).

One run per seed on development cohorts and reused seeds.

**Run.** Root `/x/Temp/kindle-breakout-budget-20261116` (sources copied
from the protocol root and pinned in `sources.sha256`, unchanged),
`make_budget_queue.py`, `queue-12.json` (32 jobs: per arm one training
run and twelve evaluations, then six probes), guards `guards-12`, unit
`kindle-breakout-budget-20261116`, 15 GiB memory cap, about 5.5 hours.
Launched by hand; nothing is chained. Kindle on Meganeura main `e3215fa2`.

## Amendment, November 16 12:34 UTC: the open seed-22003 cells on a software device

A [GPU kernel fault](../results/2026-11-16-breakout-ratio16-budget.md)
stopped the queue after `B22`'s training and one 1.2M cell, and the host
guard blocks the device for this boot. Frozen evaluation also runs on
Mesa's software Vulkan device (llvmpipe, CPU only, the faulted card not
opened): a 100-tick smoke of the completed cell ran at about two ticks
per second and chose the same action as the GPU log in 718 of 800
lane-ticks, diverging first at tick 14. It is the same checkpoint and
runner on different floating-point code, so its games are another draw
of the same policy, not a replay of the GPU's.

Added, run by hand as memory-capped units outside the GPU queue, with
`VK_DRIVER_FILES` set to the llvmpipe ICD only, no device id and no
device-name expectation, otherwise the queue's command:

- `B22` at 1.2M, matched play, the three open cells (777001 greedy,
  022003 sampled, 022003 greedy), written as `…-reset-llvmpipe.jsonl`,
  about 50 to 90 minutes each.
- The completed cell (777001 sampled) once more on llvmpipe, as the
  check that the software device reproduces the GPU cell's distribution
  (GPU: mean 278.8, median 277).

Reading: the declared seed-22003 tests are read from these cells, marked
as software-device cells; `K`'s GPU cells stand for 400k because `B22`'s
400k world and policy weights are bit-identical to `K`'s. The GPU cells
in `queue-12b.json` still run when the device is cleared and supersede
them. Seed 1009 needs the GPU and stays open, so the both-seed tests
stay open too.
