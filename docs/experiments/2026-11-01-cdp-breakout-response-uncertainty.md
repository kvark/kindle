# Breakout: can commands predict paddle-response error?

The retained teacher has larger paddle than landing errors near selected losses.
Identical logged inputs can produce different next positions, and the recent
command-dependent mean model missed its fixed fresh-gameplay gain screen.
Keep that rejection and the original teacher. This separate diagnostic tests
whether a simple uncertainty estimate predicts errors of the original model.

Use the existing 37,448 history rows and unchanged episode-separated split:
19,146 interior training rows from episodes 0–1, and all 13,102 episode-2 rows,
including 2,955 near descending balls and 849 stationary-edge rows. Reproduce
the recorded original clipped response predictions and previous control metrics.
No new replay, refit of the mean, command change or trajectory is selected.

For each training row, let residual be next observed pooled paddle center minus
the original predicted center. Estimate its second moment as the mean squared
residual in each of nine current/previous horizontal-command groups. Compare
with one global training mean squared residual. Use no smoothing, extra feature,
coefficient search, grouping change or held-out fitting. Preserve the existing
stationary-edge command reset. Record group sizes, residual means and variances;
second moments include both bias and variance and are not pure noise estimates.

On held-out-all and held-out-near-descending rows, compare squared-error forecast
MSE, `mean((residual² - predicted_second_moment)²)`, and Gaussian negative log
likelihood, `.5 * mean(log(2*pi*v) + residual²/v)`. Use a fixed 1e-6 pixel² floor
only for the likelihood calculation. Require both scores to be strictly lower
in both subsets before considering a separately declared uncertainty-aware
controller. This is a prediction screen, not evidence that Gaussian errors or
independent successive errors are correct. No controller is selected by a pass.

Report all training and held-out group counts, observed moments, per-lane and
stationary-edge scores, and empirical coverage within twice predicted RMS error.
Retain every row, including training exclusions, in the prediction artifact.
The already-examined episode-2 set is development evidence. Interior-only fitting,
edge clipping, pooled timing, correlated transitions and policy distribution
changes limit interpretation; no irreducible-noise or causal-control claim follows.

Use one low-priority CPU core, a 30-second analysis cap and 60-second controller
cap. Tests cover command routing, known group moments, untouched held-out data,
scalar scoring and invalid input. Pin retained rows/reviews/completion receipts,
new code and this protocol. Preserve any failure and exclusive output creation.
Verify all 521 active Qbert inputs and original PID836919/invocation
425aae9cee2541fe9623b753225af57f before and after. Retain CPU co-host time without
correcting training wall time. No native/emulator import, GPU work or updates.

Root `/x/Temp/kindle-breakout-response-uncertainty-20261101`. No repeated fit,
alternative grouping, automatic gameplay run or learner follows. The original
Qbert RGB64 learner continues. The five-game goal remains unfinished.

CPU preflight passes four fixture groups in 2.407s, verifies both retained
completion receipts and 14 diagnostic inputs, and preserves all 521 active
Qbert pins and the original process identity. No fit has run at selection.
