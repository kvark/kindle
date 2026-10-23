# Freeway exploration recipe passes all three intermediate gates

The two prospective replications succeed. Together with the retained selected
seed3019 rescue, the fixed recipe reaches **32.10 mean,72/72 rounds with at
least25 crossings**. The published33.40 reference remains unmet.

| Seed | Fixed exploration recipe | Unassisted control | Successes | Training |
| --- | ---: | ---: | ---: | ---: |
|1009 |32.667 |31.833 |24/24 |38m20s |
|2017 |30.667 |30.583 |24/24 |38m16s |
|3019, retained rescue |32.958 |0 |24/24 |38m19s |

All actual-initial scores are0 with zero successes. Unassisted mean20.81 and
48/72 successes stay separate; no failed result is replaced. The intervention
mainly rescues3019, with small changes on the previously successful seeds.
It was selected on3019;1009/2017 are prospective replications on reused
development seeds, not an untouched final test. Three seeds do not establish
universal reliability or five-game supremacy.

The unchanged12M/B8/T16/BF16/ratio32 centered-CDP/action-effects1 recipe uses
eight environments with batched inference. On5% of fixed16-action blocks it
holds a uniformly chosen action from all18, without directional hints or reward
shaping. Frozen evaluation disables the aid. Each run gets500k actions and
124,939 updates, with zero debt. The three aided runs total1h54m56s training;
the two new ones take1h16m36s, plus automated evaluation and audits.

Both new runs pass three guards each, exact346 same-seed initial tensors against
their unassisted controls, finite checkpoints, independent action/override
accounting, two exact346-tensor frozen exports and whole replay/video checks.
The retained3019 evidence has the same checks. All cohorts contain24 natural
rounds; no evaluation learning or best-mode/checkpoint selection. The combined
audit verifies identical configuration apart from seed and matching native,
runner and wrapper identities. GPU ends05:13:04UTC October23; final audit and
service end05:13:56. No worker from this queue remains.

Keep this fixed exploration recipe as the successful Freeway development
baseline. Do not allocate another exploration sweep. The next learning
experiment addresses Breakout's plateau with longer sequences at fixed sample
budget; the33.40 Freeway target and all other game targets stay open.

- [Combined audit](../../runs/cdp-freeway-persistent-replication-20261023.mr2oPmJq/cohort-review.json)
- [1009 whole rollout](../../runs/cdp-freeway-persistent-replication-20261023.mr2oPmJq/seed-1009/freeway-1009-final.mp4)
- [2017 whole rollout](../../runs/cdp-freeway-persistent-replication-20261023.mr2oPmJq/seed-2017/freeway-2017-final.mp4)
- [3019 whole rollout](../../runs/cdp-freeway-persistent-v3-20261021.QSzX1nuK/seed-3019/freeway-3019-final.mp4)
- [Fixed plan](../experiments/2026-10-23-cdp-freeway-replication.md)
