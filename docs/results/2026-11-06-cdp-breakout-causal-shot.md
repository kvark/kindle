# Pre-contact offset helps prediction but misses the fixed steering-model screen

The [fixed CPU analysis](../experiments/2026-11-06-cdp-breakout-causal-shot.md)
finds a useful pre-contact association, but fails its declared screen. Adding the
planner's predicted contact offset reduces holdout slope MAE by **12.972%**
relative to incoming direction alone, below the required 20%. Direction accuracy
reaches **76.761%**, below 80%. All eight lanes improve MAE, and RMSE improves
against both baselines. Keep the original teacher; no controller, fit retry,
learner or automatic successor is selected.

| Model | Holdout slope MAE | RMSE | Direction accuracy |
|---|---:|---:|---:|
| Training-mean constant | .725288 | .808316 | 58.803% |
| Incoming direction only | .691547 | .800543 | 64.437% |
| Predicted offset + incoming direction | .601837 | .733586 | 76.761% |

The offset model improves MAE by 17.021% over the constant baseline, also below
20%. Its median / 95th-percentile / maximum absolute errors are
.495503 / 1.320562 / 2.272746. Incoming-only values are
.517082 / 1.417082 / 2.480645. All targets are nonzero. These errors are signed
horizontal/upward slope units, not pixels, contact success rates or game scores.

## What this adds to the earlier evidence

Older contact-offset summaries reconstruct the impact point by intersecting
incoming and outgoing fitted lines. The outgoing trajectory therefore enters
that apparent predictor. This analysis instead uses a recorded decision at or
before the first raw frame in the incoming fit, strictly before the outgoing
fit starts. The chosen plan, landing target, current dx/dy and predicted impact
are all saved controller inputs/outputs available at that decision.

It uses no actual reconstructed contact offset, future paddle position or width,
later reward, outgoing fit parameter or future observation as a feature. Future
geometry still selects which successful contacts enter this diagnostic. Thus
the result supports a pre-contact predictive association, not a causal effect
of intervening on the action or evidence of calibrated steering precision.

All 776 previously matched contacts qualify, with no exclusions or replacement
rows. They come from the retained teacher's first three natural episodes per
lane, scoring 225.083 with 0/24 two-wall successes. The old 674 stable-width
subset is not required: width is not an input here. Original annotations and
failures remain unchanged. There is no raw-frame replay or contact refit.

Episode 0–1 supplies 492 fitting rows; episode 2 supplies 284 holdout rows from
the same heavily reused cohort. Every lane has at least 21 holdout rows. One
fixed ridge penalty .001 applies to non-intercept coefficients, with no feature
standardization, search or later refit. With `predicted_offset` in native pixels
defined as landing target minus predicted paddle center, the fitted model is:

`slope = .039076354 + .861330106 * predicted_offset/8 + .206871660 * dx/dy`

The positive offset coefficient and all coverage/rank requirements pass. Both
holdout MAE-improvement requirements and the direction threshold fail. Training
MAE is .616323, versus .679260 for incoming-only and .724227 for constant.
Training and test episodes, actions and labels remain separate during fitting.

| Lane | Holdout contacts | Incoming-only MAE | Offset model MAE | Offset direction accuracy |
|---|---:|---:|---:|---:|
| 0 | 33 | .846548 | .771856 | 66.667% |
| 1 | 21 | .716694 | .629916 | 80.952% |
| 2 | 31 | .588677 | .530758 | 87.097% |
| 3 | 46 | .575837 | .485775 | 89.130% |
| 4 | 51 | .670353 | .579804 | 70.588% |
| 5 | 39 | .762418 | .649551 | 74.359% |
| 6 | 24 | .780817 | .704639 | 75.000% |
| 7 | 39 | .667009 | .554080 | 71.795% |

The chosen action is observational; sticky actions, tracking error and nonlinear
physics remain. This small linear model's failure neither rejects every aiming
policy nor identifies the cause of the native learner's control gap. It does
not change any frozen learning score or the original 864/90% two-wall gate.

## Execution and independent verification

Six CPU fixture groups pass: decision-time/reset boundaries, interpolation and
future-feature exclusion, invalid plans, fit/holdout isolation and independent
ridge arithmetic, zero-target direction handling, and screen boundaries.
Preflight takes .233850s including its wrapper, checking 63 pins and 47 historical
sealed files. No new native, emulator or GPU construction occurs.

The analysis takes .325913s. The independent reader takes .314241s, directly
joining every saved feature back to the original 52,072-action / 208,254-frame
ledger, independently solving all three models through augmented least squares,
and checking all 776 rows, 54 metric groups and the failed screen without importing
the analysis implementation. The controller completes in **1.058103s** at
**21:17:03.571 UTC November 6**, with 64 pins unchanged and 18 sealed artifacts.
All subprocesses exit zero; there is no failed or repeated stage. The separate
terminal closure verifies hashes, original process termination and no active
Kindle service or render-device holder.

Execution source `569d34bf9535459ed371153163fd3b279845cad0`. Root
`/mnt/data/kindle-breakout-causal-shot-20261106` retains the protocol, scripts,
fixtures, predictions, coefficients, review, independent recount, declaration,
execution and closure receipts. Large artifacts stay outside Git. No new game
action, teacher integration, learner update or GPU work occurs. Keep the
retained recipe with the [fresh-seed/cohort limitations](2026-11-06-cdp-breakout-seed22003.md);
the full five-game goal remains unfinished.
