# Fixed causal estimate of the latest paddle position

**Complete:** the [result](../results/2026-11-03-cdp-breakout-paddle-timing.md)
passes the fixed screen: held-out overall MAE1.435→.975 and near-descending
1.470→.963, with lower RMSE and all eight lane MAEs improved. Eight CPU fixtures,
14 pins and the400-metric independent recount pass. Analysis completes once
08:55:43.996 UTC November3 in.785s; no failure,replay,GPU work or update occurs.
No teacher change,response-model fit,gameplay test or automatic successor is
selected. The root is closed; do not repeat stages. The goal remains unfinished.

Select one CPU-only analysis after the imagined-value comparison closes. The
retained pool-timing diagnostic finds that accepted pooled paddle centers equal
the midpoint of the last two raw-frame centers, while pooled-versus-latest MAE
is1.456 pixels. Test one fixed constant-velocity estimate:

`latest_center = pooled_center + observed_action_displacement / 8`.

The factor follows from half a raw frame of delay and four raw frames per action.
It is fixed before analysis, with no fitted coefficient, search or clipping.
Inputs are the teacher's current pooled center and its existing observed paddle
displacement; current commands, future pixels and raw target positions are not
inputs. Acceleration, reversals and sticky actions can invalidate the approximation.

Reuse the completed teacher trajectory at
`/x/Temp/kindle-breakout-edge-reset-20261028/standard24/edge-reset-standard24`
and aligned rows from `/x/Temp/kindle-breakout-pool-timing-20261101`. Do not repeat
their collection, replay or analysis. Join all52,072 source actions by lane,
episode and decision index; reconcile208,254 frames and42,303 selected decisions.
Preserve all existing observation exclusions. Require accepted current inputs
to match saved pooled centers and have a four-frame preceding action interval.
If that condition fails, stop and retain evidence; do not adjust the formula.

Compare the candidate and unchanged pooled center against the same saved latest
raw pixel detections. Report signed bias, MAE, RMSE, median and95th-percentile
absolute error, exact counts, all/near-descending/within-one-action subsets and
per-lane results. Near-descending uses the existing projected flag and ball y≥150;
the contact subset uses existing projected collision_steps in(0,1]. Preserve
the original24-episode cohort and report episodes0–1 and episode2 separately.
No fitting occurs; episode2 is a reused development holdout, not an unseen cohort.

A screen for a separate response-model investigation requires lower MAE and
RMSE in episode2 both overall and near descending, plus lower overall MAE in at
least5/8 held-out lanes. Failure selects no correction or retry. Passing does not
select a teacher change, gameplay test, native input change or learner: the
existing response model predicts pooled positions, and changing inputs/targets
would require a separately specified comparison. This is a state-estimation
diagnostic, not evidence that pooling causes the gameplay limitation.

Before analysis, qualify causal input isolation, duplicate/misaligned-row refusal,
episode resets and subset boundaries with CPU fixtures. Pin sources, old reviews,
aligned rows, original action ledger and this protocol. Require a clean committed
selection, no active Kindle service, bounded60s/1GiB CPU execution, all pins
unchanged and an independent JSON recount. No native module or emulator import,
environment replay, GPU work or learner updates. Root
`/x/Temp/kindle-breakout-paddle-timing-20261103`. Original teacher and native remain
unchanged; Breakout and the five-game goal remain unfinished.
