# Frozen Breakout diagnosis: visible ball, weak learned state and forecasts

The six declared initial/final probes complete in9m40s. All seven guards
including smoke, identical action/frame/pixel traces, exact346 tensors/model
and zero actor updates pass. **The pixel readout fails held-out generalization;
do not interpret that failure as lost visual information or encoder collapse.**

[Compact metrics](2026-10-07-cdp-breakout-world.json),
[full audit](../../runs/cdp-breakout-world-20261007.A9GKTNfK/audit.json),
[declaration](../experiments/2026-10-06-cdp-centered-five-game-budget.md#prepared-follow-up-frozen-breakout-diagnosis),
[gameplay results](2026-10-07-cdp-centered-five-game-budget.md).

## What the current readouts show

Held-out coordinate R² below is pooled over the same sampled random-action
trajectories, not independent gameplay episodes. Labels are diagnostic RAM
ball x/y and paddle x, never actor inputs. Negative R² loses to the held-out
constant mean; the separate training-mean control is approximately zero.

| Final learner | CNN ball x / y / paddle | Posterior ball x / y / paddle | Prior h15 ball x / y / paddle |
| --- | ---: | ---: | ---: |
| 1009 | .45 / .25 / .85 | .04 / .28 / .39 | −.46 / −.26 / .33 |
| 2017 | −.27 / .11 / .71 | −.34 / .19 / .76 | −.57 / −.26 / .61 |
| 3019 | .43 / .40 / .91 | .46 / .21 / .80 | −.32 / −.06 / .65 |

Initial recurrent ball readouts are also poor. Training improves readable
state, especially paddle state, but does not establish accurate ball state or
longer ball forecasts. Initial CNNs already retain some readable position
information. The full initial/final metrics and matched-visible persistence
controls remain in JSON; posterior estimates are not prior forecasts.

Prior h1 positive-reward AUC is.993/.935/.864 on only7 positive events. At h15
it is.612/.511/.624 on8 positives; zero prediction has lower reward MAE in every
trained seed. The h15 prior's raw cosine distance beats unrelated actions in
all three seeds, but beats persistence in only two;3019 barely beats a constant
training-mean embedding. These are limited diagnostics, not reliable reward
forecasting or proof that policy optimization alone is the bottleneck.

## Why the pixel control needs repair

All six models have byte-identical sampled pixels, labels and pixel-head fits.
The pixel head reaches training R² about.94/.92/.93, yet test R² is
−9.51/−9.36/−60.15. Validation error is already poor; no test-selected checkpoint
or hidden successful head is substituted.

A [read-only CPU analysis](../../runs/cdp-breakout-world-20261007.A9GKTNfK/pixel-readout-review.json)
finds per-pixel whitening magnifies rare intensity changes. Training normalized
inputs peak at27.69; validation/test peak at4,020.94, despite original GPU RGB
values staying within[−.5,.2843]. Some training standard deviations are only
.0001414. There are253 held-out columns with normalized magnitude above100.
An independent F64 forward pass reproduces the saved GPU predictions to
9.0e−7 relative L2 error (maximum absolute coordinate difference.00365).
This implicates diagnostic conditioning, not an observed GPU prediction fault.
No weights, pixels, targets, normalization or original evidence were changed.

**Declared bounded check (completed below):** fit one shared pixel head on the already saved corpus,
using its fixed native RGB range instead of per-pixel whitening. Keep the same
head seed20261003, hidden128/batch64,2,048 updates, Adam rate.001, regularization,
training-label normalization and validation-only checkpoint selection every128
updates. No new game actions, actor construction or actor updates. All six have
identical inputs, so do not repeat the same fit six times. Retain the failed
original control and report the new head even if it remains weak; no parameter
sweep, test selection or actor-training extension. Native6d38eea2 remains fixed,
one host-guarded GPU job limited to120s with the usual device/headroom checks.

### Fixed-range follow-up: complete October7

The declared head finishes in**5.01s**, selecting update1,792 by validation from
all2,048 updates. One guard passes without new warnings; no actor is constructed,
no actor updates and no new game actions. F64 reproduces saved predictions on
all three splits within7.4e-7 relative L2. All six source corpora/labels and the
failed original heads remain unchanged; pre-fix Python sources are retained.
[Compact follow-up evidence](2026-10-07-breakout-pixel-control.json).

| Fixed-range readout | Ball x R² | Ball y R² | Paddle x R² |
| --- | ---: | ---: | ---: |
| Training | .0008 | .0070 | .9792 |
| Validation | .0006 | .0097 | .9781 |
| Test | −.0005 | .0096 | .9794 |

The outlier failure disappears and paddle decoding generalizes. Ball decoding
does not even fit the training data; this small fully connected pixel head is
not an information ceiling. Do not tune it further or use it to blame resizing.

A separate **post-hoc, read-only visibility check** uses a fixed red-chroma
centroid in the open playfield (source x8–152/y96–180), excluding bricks and
paddle. RAM defines only the reporting subset/errors, never the detection.
It finds all483 eligible held-out balls with x/y RMSE**.628/.542 source pixels**
and R²**.9997/.9993**; training484/484 and validation331/331 also pass this
visibility check. Thus the actual saved GPU RGB64 input retains the ball in
these unobstructed frames and the diagnostic coordinates align. This is not a
full-frame/occlusion guarantee, a learned policy, or an actor-side game hint.
No actor code receives this detector. All masks/predictions remain in
`runs/cdp-breakout-pixel-range-20261007.kQHipWpq/visibility.npz`.

## Next decision: test existing capacity, not another loss

Keep centered CDP and its rewards. The current Size1M preset has only4 initial
CNN channels,256 embedding dimensions,512 deterministic units and32x4
categorical state. Small-object information reaches the pixels, but trained
CNN/state readouts are weak. This justifies a **capacity hypothesis**, not a
claim that capacity is proven to be the cause. Intrinsic reward is only a small
part of late imagined reward in Breakout1009 and Qbert3019, but substantial in
Breakout2017; a universal exploration-dominance explanation is unsupported.

First run a bounded construction/train/frozen-restore smoke of the **existing
Size12M preset**, keeping the loss, rates, replay ratio, sequence length and
exploration unchanged. It raises encoder and recurrent/policy capacity together;
it cannot isolate which module matters. Use measured cost/headroom to declare
three fresh Breakout learning seeds at a fixed interaction budget before launch.
No new encoder knobs, pixel decoder, privileged loss or automatic long queue.
See the [preflight declaration](../experiments/2026-10-07-cdp-capacity.md).

## Accounting and limits

Six models use32,768 forced-random actions and10,240 readout updates each.
Including the384-action/160-update smoke: **196,992 diagnostic actions and
61,600 readout updates, zero actor updates**. Initial actors stay at step0;
final actors at124,939. Original trajectories, fitted heads, predictions,
per-trajectory metrics and frozen exports remain under
`runs/cdp-breakout-world-20261007.A9GKTNfK`. One known standalone allocation
warning at initial1009 startup is retained; all jobs complete normally.

These are development trajectories, not a competence evaluation or a new
three-seed learning comparison. Readout failure does not prove absent
information; readable coordinates do not prove sufficient control state.
Raw cosine diagnostics are not the centered minibatch training objective.
The corrected pixel control and independent visibility check resolve the input
question only on their stated subsets. All five original quality targets remain
open; no broader training allocation is completed by these diagnostics.
