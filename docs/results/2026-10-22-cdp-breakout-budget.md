# Breakout improves with experience, remains far from mastery

One fresh12M/B8/BF16 centered-CDP seed1009 improves from58.83 at500k actions
to98.21 at2M, versus1.42 actual-initial. No selected round reaches the historical
two-wall864-point gate. Published381.81 and multi-seed reliability remain unmet.

| Policy | Frozen mean,24 natural rounds | Two-wall successes | Training elapsed |
| --- | ---: | ---: | ---: |
| Actual initial |1.417 |0/24 |0 |
|500k prefix |58.833 |0/24 |37m35s |
|2M final |98.208 |0/24 |2h29m51s |

The fixed within-life improvement is39.375 points (66.9%). Unlike Pong's
near-flat budget comparison, more experience helps this seed. This is not a
replicated effect, a matched published control or proof that more budget alone
will close the gap. Historical1009 at500k scored48.17 on different backend/
precision; do not attribute that difference solely to optimization or BF16.

The online last50 returns are65.76/66.02/112.68/57.10 at500k/1M/1.5M/2M.
These are nonstationary training trajectories, not evaluations of the saved
checkpoints. Their unevenness motivates diagnosis, not selecting the1.5M
checkpoint after seeing the curve. It remains saved but unevaluated.

Same unassisted12M/B8/T16/H15/R32/microbatch8/replay100k/centered CDP/
action-effects1 recipe, eight environments/batched inference. No action hint,
override, RAM input or reward shaping. Native pixels, one GPU64 resize,
full18/sticky.25/repeat4. Qualified c8aeba28 implementation, no rebuild.

Training completes2M actions/499,939 updates with zero debt. All four guards,
three exact346-tensor frozen comparisons, finite checkpoints, complete action/
reward/frame replays and whole videos pass. All72 selected episodes are natural,
with zero evaluation updates/cutoffs. Training ends October22 at02:05:35UTC;
GPU collection ends02:06:49 and final audit02:07:31. Service exits successfully.

## Next: forecasts before another long allocation

Replay reward estimates have already seen the target observation. Their low
error does not prove useful open-loop dynamics. Late imagined intrinsic reward
is.00488 versus extrinsic.09208; this does not support assuming bonus domination.
The [completed frozen forecast check](2026-10-22-cdp-breakout-forecasts.md)
uses both predefined checkpoints and the same complete final-policy trajectory.
H15 cumulative reward MAE improves1.856→1.338 and beats all fixed controls at2M;
exact timing/false positives remain imperfect. A separately declared warm-start
life now tests more experience after its short live/frozen smoke passes. No
backend project or representation rewrite; all five quality targets stay open.

- [Complete audit](../../runs/cdp-breakout-budget-20261021.wqYChxy8/learning-review.json)
- [500k whole rollout](../../runs/cdp-breakout-budget-20261021.wqYChxy8/breakout-1009-prefix.mp4)
- [2M whole rollout](../../runs/cdp-breakout-budget-20261021.wqYChxy8/breakout-1009-final.mp4)
- [Actual-initial whole rollout](../../runs/cdp-breakout-budget-20261021.wqYChxy8/breakout-1009-initial.mp4)
