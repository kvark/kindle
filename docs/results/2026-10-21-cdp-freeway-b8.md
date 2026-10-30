# Smaller batches recover Freeway learning on the fast backend

**Cohort update:** [the third seed fails](2026-10-21-cdp-freeway-reliability.md).
The completed three-seed mean is20.81 with48/72 crossing successes. The
successful single-seed batch comparison below is not a reliability result.

**Seed2017 scores30.58 frozen with B8, versus0 with B32. All24 natural rounds
reach25 crossings; the actual-initial control scores0.** B8 takes37m36s,
versus28m14s for B32. Use B8 for the next learning replicates, not the faster
but unsuccessful B32 recipe.

| Same500k-action budget | Historical B8 | Current B32 | Current B8 |
| --- | ---: | ---: | ---: |
| Final frozen mean |17.25 |0 |30.58 |
| Rounds with25 crossings |0/24 |0/24 |24/24 |
| Optimizer updates |124,939 |31,211 |124,939 |
| Training wall time |3h54m12s |28m14s |37m36s |
| Online last50-round mean |5.86 |0 |22.76 |

![Online learning against actions and wall time; diamonds are frozen scores](https://github.com/kvark/kindle/blob/archive/phase2-before-cleanup-20261025/docs/results/2026-10-21-cdp-freeway-b8.svg)

Every nonempty retained online report is plotted, with no smoothing or
extrapolation. Diamonds are final frozen means, not online training scores.
This is one learner seed, not a confidence band or seed-reliability result.

Current B8/B32 use the same qualified native binary, BF16, model, seed, rates,
reward/exploration, action protocol and interaction budget. **All346 actual
initial tensors match exactly.** Only batch and world microbatch change32→8.
This changes optimizer/EMA/warmup cadence, sampling trajectories and
shape-dependent execution together; it does not identify the causal component.
No kernel or model rewrite was made.

Historical B8 retains its old default precision/backend. Current B8 is6.23×
faster in whole training wall time for this seed and scores higher, but this
is not an isolated backend comparison or a three-seed learning-speedup claim.

## Completed checks and trajectory

- 500k actions,124,939 updates, zero debt; mean reported update16.183ms.
- Only56/240 training rounds are positive. Last50-round means are0 at200k
  and300k, .14 at400k, then22.76 at500k. Learning emerged late; early zeros
  alone would have been a poor stopping criterion for this fixed run.
- Final returns range27–32, mean30.5833; actual-initial24-round mean0.
  Both frozen cohorts have zero updates; all episodes/tails remain retained.
- Three GPU guards pass, zero new warnings. All saved346 tensors/model are
  finite. Both whole trajectories replay exactly for rewards, boundaries,
  resets and frame counts, producing unselected stream0 videos.
- Full frozen runs verify zero updates; only the preceding smoke exports
  before/after tensors for exact equality. Do not expand that claim.
- Controller and automatic CPU audit finish successfully at17:19:57UTC.

[Raw review](../../runs/cdp-freeway-b8-20261021.JEzySupH/learning-review.json)
· [Final whole video](../../runs/cdp-freeway-b8-20261021.JEzySupH/freeway-2017-final.mp4)
· [Untrained whole video](../../runs/cdp-freeway-b8-20261021.JEzySupH/freeway-2017-initial.mp4)
· [Failed B32 result](2026-10-21-cdp-training-return.md).

## Next: reliability, already allocated

Fresh original seeds1009 and3019 each receive the same B8/BF16 recipe and500k
actions, then24 final/24 actual-initial frozen rounds and whole videos.
Runs are serial; each is audited before proceeding. Execution/audit failure
stops the controller. Retain scores regardless of success. No automatic
budget extension or next-game allocation follows the cohort.

Service`kindle-cdp-freeway-reliability-20261021`; [artifacts](../../runs/cdp-freeway-reliability-20261021.osxXuUW7)
are in fresh `/mnt/data/kindle-cdp-freeway-reliability-20261021.osxXuUW7`,
linked from`runs/`, because local disk has only381MiB free. NFS artifact storage
differs from2017's local storage and can affect wall time; native sources,
binary and model recipe are unchanged. Earlier evidence stays in place.
Five-game supremacy and the33.40 long-run Freeway target remain open.
