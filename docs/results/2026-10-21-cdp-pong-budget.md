# Pong: wins at500k; little gain from extending to2M

Fresh unassisted seed1009 wins all24 frozen matches at both predefined
checkpoints. The initial policy loses all24. The intermediate Pong gate passes;
the published20.4455 reference and training-seed reliability remain unmet.

| Policy | Mean return | Wins | Training elapsed |
| --- | ---: | ---: | ---: |
| Actual initial |−20.125 |0/24 |0 |
|500k prefix |16.750 |24/24 |37m13s |
|2M final |16.917 |24/24 |2h28m15s |

Extending this life costs another1h51m02s for only0.167 higher frozen mean.
Stream-bootstrap intervals overlap:15.875–17.542 at500k,16.250–17.625 at2M.
These are conditional on one trained seed, not training-reliability intervals.
This is evidence against automatically spending2M actions on every replicate,
not proof that more experience never helps. Last50 online returns at500k/1M/
1.5M/2M are15.76/16.02/16.48/16.56.

Same qualified c8aeba28 backend,12M/B8/T16/BF16/ratio32 centered CDP and
action-effects1, eight vector environments and batched inference. No persistent
action overrides or game-specific aid. Native pixels with one GPU64 resize;
full18 actions/sticky.25/repeat4. Training completes2M actions/499,939 updates
with zero debt. Historical Pong backend/precision differ; this is not an
isolated backend speedup or a same-protocol published baseline comparison.

All four guarded jobs and the automated CPU review pass. The three frozen
cohorts have24 natural rounds each, no cutoffs, zero updates, and all346 saved
tensors exactly unchanged. Finite checkpoint, full action/reward/frame replay
and whole stream-zero video checks pass. Training ends22:06:19UTC; all GPU
jobs finish22:08:20 and review22:09:44. The service exits successfully; no
worker from this allocation remains. Other saved snapshots are not evaluated
or selected as winners.

## Next: inexpensive reliability, then weaker games

Fresh original seeds2017/3019 run serially for500k actions each, retaining the
same recipe. Each has124,939 updates, final/actual-initial24-round evaluations,
exact frozen exports and whole videos; audit each before starting the next.
The1009 prefix is the predefined third500k observation, not an independently
retrained seed or a best-checkpoint selection. Report all failures. No automatic
2M extension; review this cohort before moving to weaker games. Backend
optimization remains parked and the full five-game objective remains open.

Service: `kindle-cdp-pong-reliability-20261021`.
[Replicate controller](../../runs/cdp-pong-reliability-20261021.R6HKgGOt/cohort.py).

- [Complete audit](../../runs/cdp-pong-budget-20261021.SNxyTFk7/learning-review.json)
- [500k whole rollout](../../runs/cdp-pong-budget-20261021.SNxyTFk7/pong-1009-prefix.mp4)
- [2M whole rollout](../../runs/cdp-pong-budget-20261021.SNxyTFk7/pong-1009-final.mp4)
- [Untrained whole rollout](../../runs/cdp-pong-budget-20261021.SNxyTFk7/pong-1009-initial.mp4)
