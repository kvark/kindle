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

## Completed500k reliability check

| Seed | Frozen mean | Wins | Actual initial | Training to500k |
| --- | ---: | ---: | ---: | ---: |
|1009 prefix |16.750 |24/24 |−20.125 |37m13s |
|2017 |15.083 |24/24 |−20.250 |37m13s |
|3019 |15.292 |24/24 |−20.250 |37m24s |

**Mean15.708,72/72 wins; all three intermediate gates pass.** All initial
controls lose24/24. These are three training seeds, not72 independent learners;
published20.4455 remains unmet. Seed1009's predefined prefix is not a second
independent learner or a best-checkpoint selection. Time to500k sums1h51m50s;
actual training also includes1009's extra1h51m02s, which is not discarded.

Fresh2017/3019 each complete500k actions/124,939 updates with zero debt, then
final/initial24-round controls. All six new guards, four346-tensor frozen
comparisons, finite checkpoints, whole action/frame replays and videos pass.
Each seed is audited before the next. Review finishes23:31:58UTC; service
`kindle-cdp-pong-reliability-20261021` exits successfully and has no worker.

The combined CPU review recomputes selected scores/wins and confirms the same
native binary and config except intentional learner seeds. An initial interactive
config-equality assertion omitted that seed field; the corrected reviewer passes
without repeating GPU work or changing any score. [Combined evidence](../../runs/cdp-pong-reliability-20261021.R6HKgGOt/cohort-review.json).

Move to [one fresh longer-budget Breakout learner](https://github.com/kvark/kindle/blob/archive/phase2-before-cleanup-20261025/docs/experiments/2026-10-21-cdp-breakout-budget.md),
not further Pong extension or backend optimization. Full five-game mastery
and matched RGB comparison remain open.

- [2017 whole rollout](../../runs/cdp-pong-reliability-20261021.R6HKgGOt/seed-2017/pong-2017-final.mp4)
- [3019 whole rollout](../../runs/cdp-pong-reliability-20261021.R6HKgGOt/seed-3019/pong-3019-final.mp4)

- [Complete audit](../../runs/cdp-pong-budget-20261021.SNxyTFk7/learning-review.json)
- [500k whole rollout](../../runs/cdp-pong-budget-20261021.SNxyTFk7/pong-1009-prefix.mp4)
- [2M whole rollout](../../runs/cdp-pong-budget-20261021.SNxyTFk7/pong-1009-final.mp4)
- [Untrained whole rollout](../../runs/cdp-pong-budget-20261021.SNxyTFk7/pong-1009-initial.mp4)
