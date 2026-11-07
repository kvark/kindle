# Command history predicts some Breakout paddle-response error

A fixed nine-group error model passes its declared held-out prediction screen.
Relative to one global training error estimate, squared-error forecast MSE falls
**5.04% overall** and **4.49% near descending balls**, and Gaussian negative log
likelihood also falls. Both scores improve in all eight held-out lanes. The
original paddle-position predictions and teacher remain unchanged; this is not
a new gameplay result or evidence of a beneficial controller.

| Held-out subset | Rows | Error-forecast MSE: global → command | Gaussian NLL: global → command |
| --- | ---: | ---: | ---: |
| All | 13,102 | 1217.075 → 1155.792 | 2.6124 → 2.3624 |
| Near descending | 2,955 | 1215.734 → 1161.089 | 2.6110 → 2.3212 |
| Stationary edge | 849 | 969.242 → 971.635 | 3.0045 → 2.5825 |

Error-forecast MSE is the mean squared difference between actual squared position
error and its predicted second moment, in pixel⁴. These values are not position
MAE or gameplay losses. NLL uses the original predicted position as its mean and
a fixed 1e-6 pixel² floor; Gaussian errors are not established.

Training uses the same 19,146 interior rows from episodes 0–1 as the earlier mean
fits. Each current/previous horizontal-command pair receives its observed mean
squared residual, with no smoothing or search. Groups have 1,708–3,737 training
rows. The global second moment is 9.658 pixel². Reversals have 26.330/26.113,
repeated movement 4.118/3.467, and repeated NOOP .524. These are conditional
observations, not proof that a reversal causes the error or that avoiding one
improves control. Second moments include bias as well as variance.

Held-out mean predicted error² is 9.490 versus observed 10.808 pixel². At
stationary edges, it is only 4.809 versus observed 18.383; error-forecast MSE
worsens there despite the lower likelihood score. Coverage within twice the
predicted RMS error rises 91.63%→93.31% overall but falls 77.39%→73.85% at edges.
The model therefore remains poorly calibrated in an important subset. Its
interior-only training, clipping, pooled timing and correlated transitions limit
interpretation. No temporal-independence or irreducible-noise claim follows.

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-response-uncertainty.md)
reproduces all 37,448 original rows, targets, clipped predictions, splits and
control metrics. All 5,200 training exclusions remain in the prediction artifact;
the held-out episode-2 set is previously examined development evidence. The
[rejected command-dependent mean controller](2026-11-01-cdp-breakout-command-response.md)
and its fresh-gameplay screen remain unchanged. No new controller, native loss,
teacher integration, parameter sweep or learner is selected by this result.

Four CPU fixture groups and the 2.407s preflight pass. Analysis takes **.772s**,
controller **3.262s**, completing **19:46:15 UTC November 1**, source `1bbd75e`.
All 16 diagnostic inputs and 521 active Qbert inputs remain unchanged, preserving
original PID836919/invocation425aae9cee2541fe9623b753225af57f. Retain CPU co-host
time without correcting Qbert training time. No failed or repeated stage,
emulator/native import, GPU work or learner update occurs. Qbert RGB64 continues
under its original observers. The five-game goal remains unfinished.

Evidence root `/x/Temp/kindle-breakout-response-uncertainty-20261101` is complete;
do not repeat the fit, preflight or collection.

The subsequent [fixed one-step gameplay test](2026-11-01-cdp-breakout-one-step-risk.md)
lowers mean return225.083→184.417 (−18.07%), with only three of eight lanes
improved and zero two-wall successes. Its exact replay passes. Retain the
original teacher; better prediction did not establish a useful risk penalty.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `ee77f59f9f6f85a571ffa7d877f6d4f5921ea2def3fd828c36e29089eea9322f` |
| Review | `9fe6a2a716ce191141e829b17fc5f626a1b0eac4d9dfc29d93395da1c318ea5b` |
| Predictions | `8ea33c253a817746116b0b0df8b49cd37392247da5dafe7800dbfbb98dd1b28c` |
| Completion | `2b9e0d5db439d11d49b2df5de14ff19ff2bd988d6ec576626c62fbd7e4d722da` |
