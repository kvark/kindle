# Freeway exploration: six runs complete, no reward discovery

The first CDP disagreement screen **does not unlock Freeway**. All three
exploration seeds and all three extrinsic-only controls collect zero real rewards.
The guarded queue finished October4 at23:00:07 UTC, after22m32s. It stopped as
declared: no additional frozen evaluations, budget extensions or GPU jobs.

## Results

Each run uses32,768 actual aggregate actions,8,131 updates and16 completed natural
episodes. Seeds1009/2017/3019, N8/Size1M/B8/T16/H15/R32, full18 actions/sticky.25;
no action aid, pretraining or shaped external reward. Full settings are in the
[declaration](../experiments/2026-10-04-cdp-freeway-exploration.md).

| Metric | Extrinsic-only CDP | CDP + disagreement |
| --- | ---: | ---: |
| Positive real reward events, by seed | 0 / 0 / 0 | 0 / 0 / 0 |
| Completed-episode mean return, each seed | 0 | 0 |
| Mean run time | 213.96s | 220.95s |
| Mean full update | 24.52ms | 25.29ms |
| Final imagined policy entropy, by seed | 2.8904 / 2.8904 / 2.8904 | 2.8890 / 2.8868 / 2.8838 |

Paired wall time rises3.31% [1.63%,6.26% seed-bootstrap95% interval]. The score
difference is0 [0,0], a floor result, not evidence of equal learning capacity.
All196,608 actions,48,786 updates,96 completed episodes and unfinished tails
are retained. These are three learner replicates, not24 independent streams.

## What changed, and what did not

The intrinsic channel is active. Candidate mean absolute advantages are
0.112/0.110/0.121, versus exactly zero in all controls. First-window imagined
bonuses0.477/0.462/0.437 shrink to0.0094/0.0117/0.0105 in the final window.
The policy nevertheless stays close to the uniform maximum entropy ln(18)=2.8904.

This establishes that the bonus supplies learning signals, not that it drives
useful exploration. Next, diagnose whether it distinguishes actions that change
the player's reachable state, and whether those preferences persist long enough
to expand coverage. Current logs do not establish whether action contrast,
latent information, reward scale or temporal credit is the limiting factor.
Do not extend all six runs or change multiple factors without that evidence.

All six guards, complete trajectory/counter audits and finite checkpoints pass.
No new allocation warnings, NVML polling or recovery. CI295 passes Linux,
macOS and Python for implementation11059fb. Qualification failures and5,120
excluded smoke actions remain in the [qualification report](2026-10-04-cdp-exploration-qualification.md).

Accounting correction: earlier prose said8,135 updates. The launched controller
and audits required8,131, consistent with first training at action248:
1+(32,768−248)/4. This was a prose error, not a changed action budget or missing
learner work. No frozen competence claim or new successful rollout video.

[Compact results and sampled action/time curves](2026-10-05-cdp-freeway-exploration.json).
Full artifacts: `runs/cdp-exploration-20261004.mE26yV2V`.
