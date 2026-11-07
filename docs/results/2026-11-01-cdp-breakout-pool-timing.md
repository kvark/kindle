# Breakout paddle observations exactly follow two-frame pooling on accepted rows

The retained teacher's observed paddle center equals the midpoint of the last
two raw-frame component centers on **all35,193 accepted frame pairs**. Against
the latest raw component alone,the observation has1.456-pixel MAE over35,610
accepted decisions. This identifies temporal averaging on these rows rather
than an unexplained detector discrepancy. It does not establish that pooling
causes failed control or select an observation/controller change.

| Selected phase | Latest-raw comparison rows | Pooled-versus-latest MAE | Stable-pair rows | Pooled-versus-midpoint MAE |
| --- | ---: | ---: | ---: | ---: |
| All |35,610|1.456|35,193|0.000|
| Near descending |7,509|1.446|7,438|0.000|
| Projected impact within one action |1,872|1.665|1,838|0.000|

The pooled-versus-latest95th-percentile absolute difference is3 pixels in all
three subsets. On stable pairs,the most recent raw-frame motion has mean
absolute magnitude2.920 pixels overall,2.899 near descending and3.349 near
impact. The midpoint relation holds on every accepted lane. The source remains
the same225.083-point teacher withzero of24 two-wall successes.

## Response error remains against the calibrated target

The unchanged model was calibrated to predict the next **pooled** observation.
On exact shared rows,its MAE is larger against the latest raw component:

| Phase | Shared rows | Prediction versus next pooled MAE / RMSE | Prediction versus latest raw MAE / RMSE |
| --- | ---: | ---: | ---: |
| All |31,817|1.965 /3.208|2.875 /3.836|
| Near descending |7,327|1.860 /3.102|2.706 /3.694|
| Projected impact within one action |1,769|2.170 /3.411|3.246 /4.111|

All rowwise residual decompositions reconcile:prediction-minus-raw equals
prediction-minus-pooled plus pooled-minus-raw. Their signs can cancel;these
MAEs cannot be added or interpreted as fractions of avoidable control error.
Even the pooled target retains substantial error and tails. The original
37,448 all/8,492 near-descending pooled pairs reproduce exactly:MAE1.953/1.894,
median1.108/1.089 and95th-percentile7.878/7.545,within the fixed1e-10 tolerance.

A current-frame correction would change both the response model's inputs and
targets. Ball observation timing and collision prediction also matter;this
paddle-only diagnostic does not quantify them. No correction coefficient,refit,
teacher change,gameplay test,loss or native architecture is selected. The earlier
rejected four- and six-feature mean-response fits remain unchanged.

## Coverage and verification

The join aligns all52,072 source actions/208,254 saved raw frames by lane,episode
and cumulative executed-frame count. Exactly42,303 decisions belong to the
fixed first three natural episodes per eight lanes. Observation exclusions are
24 unrecorded initial/reset images,6,545 cropped components,88 raw-detector
disagreements and36 nonstandard widths. Of35,610 accepted latest components,
417 preceding-frame pairs are excluded:361 cropped,22 nonstandard,31 detector
disagreements andthree width changes. Shared prediction rows exclude5,523
cropped,77 disagreed and31 nonstandard next components. All exclusions remain
recorded;edge exclusions limit generalization to the full trajectory.

Accepted components are12 or16 pixels wide,uncropped,and match the original raw
centroid within1e-9;stable pairs also match row/width. These are pixel proxies,
not exact emulator collision state. The trajectories,teacher and development
cohort are reused. No learned representation,independent seed or new gameplay
outcome is measured.

Six CPU fixtures and2.568s preflight pass. Analysis takes **1.516s**,controller
**4.475s**,completing **12:18:04 UTC November1**,source8c45350. All16 diagnostic
inputs and490 active Qbert pins remain unchanged,as does originalPID792387/
invocation460e0220583948f4b088b5dc7c92956a. Retain CPU co-host time without
correcting Qbert's training cost. No emulator replay,native/GPU work,updates,
failure or repeated analysis occurs. Complete;do not rerun qualification or
analysis. No automatic successor is selected;the five-game goal remains open.

Evidence `/x/Temp/kindle-breakout-pool-timing-20261101` includes all aligned
observations,prediction rows,exclusions,per-lane metrics and receipts.
[Fixed protocol](../experiments/2026-11-01-cdp-breakout-pool-timing.md).

- Review:`b24bcdb8411bdf3d2c5d8a228895919c53e198581f2e08fe174404414b1d80a2`.
- Aligned rows:`373f04e76f744ab25d9aedaa1f2a87d78d9c9fa4d0469f7a796de8671a35c807`.
