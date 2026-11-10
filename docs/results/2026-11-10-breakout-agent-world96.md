# Breakout agent with the state-only 96x96 world objective

[Protocol](../experiments/2026-11-10-breakout-agent-world96.md). Root
`/x/Temp/kindle-breakout-world96-agent-20261110`; outputs under `candidate/`,
guard directory `guards/`, unit `kindle-breakout-world96-agent-20261110.service`
(invocation `1a5e5fa00e2e43da90faf32d1aebc80f`, PID 2399809). Source
`af696cee` (runner copy hash in `launch.txt`) on Meganeura main `a0eff600`.
A 768-action smoke after the extension rebuild confirmed the prior-rollout
loss in the learner reports before launch (`smoke2/`); an earlier smoke with
a stale extension (`smoke/`) silently ran the squared-error world and is
retained as the reason for that check.

## Run

Training ran from 14:25 UTC on November 10 and completed 200,000
aggregate actions (799,321 emulator frames) in 1,865.9 s wall, 107.2
actions/s, with 12,469 learner updates (B8/T32, replay ratio 16); 1,750 s of
the wall was the learn stage (138 ms per update: 97 ms world, 17 ms
posterior, 9 ms imagination, 3 ms behavior). The retained CDP recipe trains
the same budget in 21m28s. 352 online episodes completed, mean return 34.5
with the 20% guided episodes included (25k-action buckets: 18.4, 32.5, 35.5,
22.0, 73.5, 42.6, 42.2, 23.5). The world losses fell from 19,011
reconstruction / 19,012 prior-rollout at update 60 to about 6,200 / 6,450
over the last 500 updates (minimum 5,791 / 4,505), reward loss .11,
continuation loss .02, KL 21.6; the actor's imagined return mean ended at
2.7 with return scale 9.5 and policy entropy .82. The frozen checkpoint is
`candidate/checkpoints/`; `candidate/final.jsonl` records a failed first
evaluation attempt that pointed at a non-existent `checkpoints/200000` path.

## Frozen evaluations (cohort 4000022003)

Three natural episodes per environment, zero updates, 25k-action truncation.
The greedy policy stalled fifteen times (episodes truncated at 25,000 actions
without a terminal), so the greedy run needed 528,464 actions and 93 completed
episodes before every stream had three natural ones; the 24-episode figure takes
the first three natural episodes of each stream.

| Checkpoint | Mode | Completed episodes | Mean return | Natural-24 mean | Max | Two-wall |
|---|---|---|---|---|---|---|
| Actual initial | sampled | 24 (24 natural) | 1.250 | 1.250 | 3 | 0/24 |
| Actual initial | greedy | 30 (30 natural) | 2.167 | 2.167 | 5 | 0/30 |
| 200k | sampled | 25 (25 natural) | 4.760 | 4.833 | 11 | 0/25 |
| 200k | greedy | 93 (78 natural, 15 truncated) | 4.785 | 5.208 | 15 | 0/93 |

References on cohort 4000022003: retained CDP 1009 94.625 greedy / 70.500
sampled; fresh CDP 22003 75.250 / 44.708. Outputs `candidate/final-v2.jsonl`
and `candidate/final-greedy-v2.jsonl`; guard directory `guards-final/`.

## Reading

The agent learned something (initial 1.25/2.17 to final 4.76/4.79) but
plays an order of magnitude below the retained CDP points on the same
cohort, and far below its own guided online returns (34.5 mean), so the
frozen policy does not carry what the teacher-aided play showed. This is one
development seed; it says the state-only pixel world, as wired into the
agent here, is not a drop-in replacement for the CDP objective at the 200k
budget. Two differences from the retained recipe are confounded and neither
is isolated: the world objective (pixel cross-entropy plus a 16-step image
rollout at 96x96 with the 4M preset instead of centered CDP) and the missing
action-effects exploration bonus. The update count matches (12,469 against
the retained learner's 12,477 at the same budget); only the wall time differs
(31 against 21 minutes). The world losses are dominated by pixels
(6,200 + 6,450 against reward .11 and continuation .02), and the imagined
return mean stayed under 7 throughout, so the critic saw little reward
signal in imagination; whether the latent that renders the ball is also the
latent the actor needs remains an open question that the standalone forecast
metrics cannot answer. The greedy stalls (15 truncations) match the earlier
Breakout greedy behaviour of not firing after a lost life.

Not established: that a better world forecast improves play, that the bonus
or the update count explains the gap, or anything about other seeds. The
cheapest next checks are a matched run with the exploration bonus decoupled
from CDP, and a run that keeps CDP and adds the pixel rollout as an
auxiliary loss, both on this seed and cohort before any wider claim.
