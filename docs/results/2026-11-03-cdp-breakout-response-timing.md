# Timing conversion modestly improves next-position forecasts

The fixed output conversion passes its declared prediction screen. On the
reused episode2 holdout, MAE falls **2.901→2.628 pixels** overall and
**2.722→2.577** near descending balls. RMSE improves in both groups, and all
eight lane MAEs improve. No response coefficient is fitted or changed.

| Subset | Rows | Raw-target MAE, original → converted | Raw-target RMSE, original → converted |
| --- | ---: | ---: | ---: |
| Episode2, all | 11,193 | 2.901154 → 2.627966 | 3.848765 → 3.705023 |
| Episode2, near descending | 2,552 | 2.722380 → 2.576993 | 3.697762 → 3.640855 |
| Episode2, contact within one action | 656 | 3.312736 → 3.233991 | 4.189021 → 4.189332 |
| All selected episodes | 31,817 | 2.874577 → 2.601959 | 3.836326 → 3.687885 |
| All selected, near descending | 7,327 | 2.705853 → 2.534426 | 3.694495 → 3.613268 |

The held-out overall95th-percentile absolute error worsens6.478→6.557 pixels,
and near-descending worsens6.344→6.491. Maximum held-out overall error rises
37.767→39.988. Within one action of contact, held-out RMSE slightly worsens
despite lower MAE. This is an aggregate prediction improvement, not a pointwise
improvement or an established solution to the larger contact errors.

The [fixed formula](../experiments/2026-11-03-cdp-breakout-response-timing.md) is:
`next_raw_prediction = next_pooled_prediction + (next_pooled_prediction - current_pooled_center) / 8`.
It uses the recorded pre-action model's first prediction and current pooled
center. Actual next observations, raw targets, future commands and later plan
positions do not enter the estimate. The teacher's coefficients, selected plans,
clipping and actions stay fixed. There is no extra output clipping or search.

This follows the [current-position timing result](2026-11-03-cdp-breakout-paddle-timing.md),
but neither result establishes contact-time control. Current and future ball/
paddle timestamps must agree; constant-velocity shifts can cancel in an
interception calculation. A separately specified contact-time assessment would
be needed before selecting a controller comparison. No teacher, native input,
architecture, loss, gameplay test or learner changes. The original225.083-point
teacher and zero-of24 two-wall result remain unchanged.

All52,072 source actions/208,254 frames and31,817 accepted forecast rows reconcile.
The preserved next-component exclusions are5,523 cropped,77 detector disagreements
and31 nonstandard widths. Both original raw-target and pooled-target metrics
reproduce to1e-10. Row joins verify pre-action debug/prediction identity, adjacent
decisions within the same episode and four-frame intervals. Targets remain raw
pixel-component proxies; this heavily reused development holdout is not an
independent cohort or a native representation measurement.

Four new CPU fixtures pass, reusing eight qualified parsing/statistics fixtures
without repetition. All21 pins pass. Independent JSON recount verifies31,817
rows and400 aggregate metrics. No failed stage, prior replay/analysis repeat,
native/emulator import,GPU work or learner update occurs.

Source `b7ed09f6`; preflight.133s. Analysis takes.945s and completes
**09:02:26.168 UTC November3**; controller completes **09:02:26.232** in1.051s.
Independent recount takes.260s and completes **09:02:52.834**. Root
`/x/Temp/kindle-breakout-response-timing-20261103` is closed; do not repeat stages.
No Kindle allocation is active at closure. No contact-time test, controller
integration or automatic successor is selected. The five-game goal remains open.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `9d9f9bc75e2c9801bb4b3d6eb2192088208eef6e14f9649203b4c380323e4083` |
| Review | `8952fec225e08d672eff5f1b910e47ae62240cc93bb3594bda34cd0f374134de` |
| Completion | `b5c49aa096ebf664bd69a31de92eb2a6d32afb190ec00466bc7d7f23ca516e15` |
| Independent recount | `a81c60cf610f4397afa4d8fcc9ba9406dc64d5a9cbf5518a1350907d55e57561` |
