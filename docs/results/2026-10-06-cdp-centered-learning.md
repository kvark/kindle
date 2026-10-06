# Centered CDP improves early Pong learning in all three seeds

**The predeclared screen passes.** Six fresh 200k-action learners finish on
October 6 at 08:41 UTC, in 2h55m56s including frozen evaluation and audits.
Centering improves every paired seed and every actual initial control. It does
not yet produce reliable wins or DreamerV3 quality.

| Seed | Unchanged CDP frozen | Centered CDP frozen | Actual initial, both arms | Centered wins /24 |
| --- | ---: | ---: | ---: | ---: |
| 1009 | -20.6250 | -8.2500 | -20.3333 | 0 |
| 2017 | -14.5833 | -3.8750 | -20.0833 | 4 |
| 3019 | -20.7500 | -13.2500 | -20.5833 | 0 |
| Learner-seed mean | -18.6528 | -8.4583 | -20.3333 | 4/72 total |

Paired frozen improvement **+10.1944, 95% learner-seed bootstrap [7.5000,
12.3750]**. Centered mean interval [-13.2500,-3.8750]; unchanged CDP
[-20.7500,-14.5833]. Three seeds make these coarse intervals; the 72 episodes
per arm are not independent learner replicates. No seed, episode, or checkpoint
was selected for its score.

## What changed and what did not

The [declaration](../experiments/2026-10-06-cdp-centered.md) changes only the
cosine training objective: subtract the same detached full-batch target mean
from prediction and target. The CNN remains jointly learned, and actor inputs
are unchanged. Capacity, source gradients, rates, replay and action-effects
exploration are identical. This is a Kindle CDP ablation, not the unchanged
paper recipe. [Numerical qualification](2026-10-06-cdp-centered-qualification.md)
precedes all six learners.

Each arm uses Size1M/N8/B8/T16/H15/R32, microbatch8, full BPTT, 100000 replay,
full18/sticky.25/repeat4, native frames and one GPU RGB64 resize. Encoder6e-6,
dynamics4e-4/base4e-5, cosine500, AGC.3, warmup1000, ac_grads=false,
action-effects coefficient1. No action hint, pixel decoder, labels, video
pretraining or external reward shaping. Both arms use native `d32f3dc8`,
Meganeura `b684ffd9`, Blade `e349cddf`, RTX5080 /580.178.04.

Actual zero-experience checkpoints are saved before the first action/update;
all 346 tensors match exactly across arms at each seed. Frozen final and initial
policies use held-out base3,500,000,000 plus learner seed, first3 natural
episodes in each of8 streams, sampled actions and a200k-action cap. All cohorts
finish naturally; no cap is reached.

## Learning curves and reward separation

Last50-completed-episode online means, at the same recorded action counts:

| Actions | Unchanged CDP mean | Centered mean | Approximate training minutes /learner |
| ---: | ---: | ---: | ---: |
| 32,768 | -20.3226 | -20.3911 | 4.6 |
| 65,536 | -20.3933 | -19.9600 | 9.2 |
| 131,072 | -19.9000 | -16.8133 | 18.4 |
| 200,000 | -19.3333 | -14.1000 | 28.0 |

All action/time curves and per-seed confidence intervals remain in the
[full independent report](../../runs/cdp-centered-20261006.n6XuQXmE/independent-report.json).
The [compact JSON](2026-10-06-cdp-centered-learning.json) retains milestones,
time curves, all seed results, diagnostic means and the full report's hash.

Last100k-action replay diagnostics, unchanged / centered:

| Seed | Raw KL | Reward loss | Negative-event prediction | Zero-event prediction |
| --- | ---: | ---: | ---: | ---: |
| 1009 | .505 /6.142 | .1294 /.0224 | -.0226 /-.5398 | -.0222 /-.0020 |
| 2017 | 2.551 /8.241 | .0302 /.0281 | -.5517 /-.4801 | -.0039 /-.0030 |
| 3019 | .487 /5.907 | .1341 /.0245 | -.0242 /-.5990 | -.0237 /-.0026 |

The two formerly weak seeds now separate negative rewards from zero rewards
in replay, alongside improved gameplay. These are update-weighted minibatch
means, not pooled event calibration or held-out prior forecasts. Raw cosine
losses are different objectives and cannot be compared as prediction accuracy.
The earlier [frozen diagnosis](2026-10-06-cdp-pong-world.md) motivated this
change. The [paired frozen follow-up](2026-10-06-cdp-centered-world.md) now
confirms useful recurrent ball state and negative-event forecasts in all three
centered seeds, while preserving coordinate/persistence and sparse-event limits.

## Audit, cost and decision

All18 guards,288 selected natural frozen episodes, zero updates/cutoffs,
finite checkpoints, exact346 tensors per frozen model, full trajectory replays
and all12 whole-video hashes/frame counts pass. Two excess initial-control
episodes and all unfinished tails are retained. The independent CPU re-audit
takes10.9s with126.6MiB peak memory and zero swap. No new GPU warning, recovery
or NVML polling; historical allocation warnings remain in each guard baseline.

Six learners cost **1.2M actions /299,634 updates /4,799,025 executed frames /
2h48m15s training**. Frozen evaluation adds517,256 actions and zero updates.
Qualification's4096 diagnostic actions and all earlier experiments remain extra
cost. Mean training time1683.03s centered versus1682.09s control; mean updates
32.13ms versus32.11ms. This shows no material observed cost increase, not a new
speedup. GPU utilization remains unmeasured.

Promote centered CDP as the next learning candidate, not as a solved agent.
Four wins out of72, mean-8.46 and only one winning seed are far short of the
unchanged Pong target20.4455 and the full **Boxing/Pong/Freeway/Breakout/Qbert**
goal. The paired state/forecast check now passes with limitations; review the
measured runtime bottleneck before the next finite training allocation.
Do not repeat failed raw-cosine budgets or
automatically restart the suite. Cross-game transfer of this improvement and
same-hardware RGB compute savings remain untested.

## Whole rollout videos

These are complete stream-zero recordings, including retained unfinished tails,
not selected successful clips. Each frozen score above covers all eight streams.

| Seed | Unchanged CDP | Its actual initial | Centered CDP | Its actual initial |
| --- | --- | --- | --- | --- |
| 1009 | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-control-1009-candidate-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-control-1009-untrained-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-centered-1009-candidate-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-centered-1009-untrained-frozen.mp4) |
| 2017 | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-control-2017-candidate-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-control-2017-untrained-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-centered-2017-candidate-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-centered-2017-untrained-frozen.mp4) |
| 3019 | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-control-3019-candidate-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-control-3019-untrained-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-centered-3019-candidate-frozen.mp4) | [video](../../runs/cdp-centered-20261006.n6XuQXmE/pong-centered-3019-untrained-frozen.mp4) |

Artifacts: `runs/cdp-centered-20261006.n6XuQXmE`; no learning worker remains.
