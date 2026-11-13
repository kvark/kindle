# Breakout: locate the event-forecast regression in retained worlds

The nonlinear fusion learner improves recorded ball tracking but predicts false
endings on 276/304 non-ending branch cases and fails retention. Before selecting
another learner, distinguish errors on observed states from errors reached during
rollout. This is a new CPU analysis of retained arrays, not a repeated assessment.

Use five fixed final worlds: coverage H0, categorical H16, spatial H16,
image-feedback H16 and nonlinear fusion H16. Read their checkpoint weights,
dense fitting/development features and the final fitting/development terminal-
aligned groups. No new native query, gradient, training, model/emulator
construction, collection or GPU allocation. Keep every earlier failure closed.

Reconstruct the continuation MLP in F64 from saved weights: linear+bias,
channel RMS with epsilon1e-4, learned scale, SiLU, output linear+bias, sigmoid.
Require absolute agreement within1e-4 with every saved native first-draw
continuation at the five exported feature endpoints. Stop on parity failure;
preserve it rather than changing tolerances after seeing results.

For observed context features, join the exact corpus arrival at start+15 and
verify RGB equality. Report terminal/non-terminal support separately, observed
head probabilities and Brier scores, and explicit unsupported strata. Never use
an observed non-terminal cohort to claim terminal detection. Compare recorded
one-step and later conditional hazards with the original validity masks; keep
terminal arrivals and exclude reset crossings. Report each fixed horizon and
all valid rows, by natural and terminal-aligned cohort. These reused/selected
cohorts do not establish population calibration or fresh-seed reliability.

For all42 fixed plans, all seven categories, histories16/32 and all eight draws,
reconstruct survival exactly from the saved continuation values with the original
discount removed. Require the original survival arrays and reported event means
to agree within1e-12. Report instantaneous hazard and cumulative ending at
1/4/8/16/32, every individual non-ending alarm, and how much cumulative risk is
already present by8/16. Report ending cases separately; do not pool an outcome-
conditioned challenge set into a population estimate. Compare ball energy with
the same saved paths given survival weight one only as a diagnostic sensitivity;
the original survival-weighted primary gate remains authoritative and failed.

Pin scripts, protocol, helper/source semantics, raw arrays, weights, corpus and
closed receipts before execution; verify afterward. Keep the numerical-parity
summary, per-origin/horizon table and compact report. No scalar calibration fit,
new threshold, metric substitution or automatic learner follows. This diagnosis
cannot establish loss interference or a causal failure mechanism without a
separate intervention. Full physics, event timing, brick memory, late coverage,
untouched families and fresh confirmation remain required.

Root `/mnt/data/kindle-breakout-world-event-diagnosis-20261108`.
