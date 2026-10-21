# Fast B8 Freeway cohort: two successes, one failure

**Reliability remains unresolved.** Original seeds1009/2017/3019 score
31.83/30.58/0 after500k actions each. The cohort mean is20.81, with48/72
rounds reaching25 crossings. Do not generalize the successful2017 batch
control into a stable learner claim.

| Seed | Final frozen mean | Actual initial |25-crossing rounds | Training |
| --- | ---: | ---: | ---: | ---: |
|1009 |31.83 |0 |24/24 |38m19s |
|2017 |30.58 |0 |24/24 |37m36s |
|3019 |0 |0 |0/24 |38m15s |

Same12M/B8/T16/BF16/ratio32 centered-CDP/action-effects1 recipe, eight streams,
full18 actions, sticky.25, native frames→one GPU64 resize. Each learner has
124,939 updates, zero debt,24 natural final-policy and24 actual-initial rounds.
The144 frozen rounds have zero updates. Nine guards, finite checkpoint checks,
six whole trajectory replays/videos and all three CPU reviews pass, with zero
new warnings. The cohort completes18:42:16UTC; no worker from it remains.
Full frozen tensor equality was not re-exported for this cohort; that claim
remains limited to the earlier smoke, not inferred from zero-update counters.

Total training1h54m11s versus historical11h41m22s, about6.14× less wall time,
but mean score falls27.54→20.81. Both cohorts have48/72 crossing successes;
the failed seed changes from2017 to3019. Historical precision/backend and
current1009/3019 NFS storage differ. This is not a matched speedup with
preserved quality, and the33.40 reference remains unmet.

## Failure evidence and bounded next test

3019 records one crossing at action260,120. It enters25 learner updates,
from260,184 through359,312; some reward-bearing predictions become nonzero
(for example.8502 at352,296). Thus the reward is not wholly disconnected.
Yet the last50 online rounds and all24 frozen rounds score0. Intrinsic bonus
remains nonzero, policy entropy about2.25 nats, and cosine loss falls to.0201.
Those diagnostics do not establish useful exploration or representation.

Test the existing **persistent-uniform action control** on a fresh3019 life:
5% of fixed16-action blocks choose a uniform held action among all18 options.
This is a generic disclosed exploration aid, not a directional hint or reward
shaping. Keep all model/rate/precision/reward settings and500k action budget.
Frozen evaluation has no overrides. Compare against this exact failed seed;
one selected-seed rescue would still require broader replication.

The first declaration is rejected by the adapter's old mutual-exclusion rule
before environment construction/GPU initialization/actions. Preserve that
failed guard and source. A two-line adapter change permits the combination;
285 CPU vector/exploration/checkpoint tests pass, including combined-CDP
action forwarding, replay accounting and ignored-override rejection. Native
code already records the executed override as the pending/replayed action;
no native binary, kernel or model rebuild is involved.

A second attempt initializes the GPU, then fails before actions/updates because
the copied runner fingerprints a nonexistent adjacent wrapper file. The
reporter now hashes the actually imported wrapper module. The285 CPU tests
also cover a wrapper loaded from another path. Both failed guards/sources stay;
there was no native numerical/API fault or recovery.

A fresh separately declared run uses
`kindle-cdp-freeway-persistent-v3-3019-20261021`. It additionally exports both
frozen checkpoints for exact346-tensor comparisons. No automatic extension
or parameter sweep. Longer-budget Pong remains pending; backend work is parked.

- [Cohort](../../runs/cdp-freeway-reliability-20261021.osxXuUW7/cohort.json)
- [1009 whole video](../../runs/cdp-freeway-reliability-20261021.osxXuUW7/seed-1009/freeway-1009-final.mp4)
- [2017 whole video](../../runs/cdp-freeway-b8-20261021.JEzySupH/freeway-2017-final.mp4)
- [3019 whole video](../../runs/cdp-freeway-reliability-20261021.osxXuUW7/seed-3019/freeway-3019-final.mp4)
- [New exploration allocation](../../runs/cdp-freeway-persistent-v3-20261021.QSzX1nuK/seed-3019/declaration.json)
- [Retained CLI refusal](../../runs/cdp-freeway-persistent-20261021.Y7bKJQkQ/seed-3019/queue/freeway-3019-train/result.json)
- [Retained wrapper-path failure](../../runs/cdp-freeway-persistent-v2-20261021.RV7QoeJY/seed-3019/queue/freeway-3019-train/result.json)
