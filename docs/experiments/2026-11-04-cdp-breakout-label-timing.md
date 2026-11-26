# Saved-log Breakout corrective-label timing audit

Selected after the completed B32/T16 comparison. This is a new CPU analysis
within the existing five-game goal, not another learner or a label-shift trial.

## Question and source review

Does the retained learner's sampled, unaided collection agree more with nearby
past teacher commands than with its current command? A systematic temporal
association could motivate a more focused control diagnostic. It cannot prove
neural reaction delay, a replay indexing bug, or a useful training intervention.

Review the exact retained native447fe50 sources and runnercf4ea238. The runner
proposes labels from current frames before acting. Native pending labels travel
with the resulting arrival, and `expert_targets` pairs state t with the label
stored at t+1. The last sequence state and reset crossings are masked. World
supervision uses those same targets with T/(T-1) normalization. Save those exact
source bytes and the inspected locations; do not rerun native qualification.

Use only the retained replay-value1 200k training log, whose full replay and
exposure audit already passed. Keep all200k actions in the accounting. No game,
native extension, model, replay, GPU context or learner is executed. The log has
labels for every action, including actions without guidance. Native proposed
actions during overrides are unavailable; exclude overrides rather than treating
teacher-executed actions as evidence of agreement.

## Fixed analysis

Compare recorded action a[t] with teacher label e[t+k] for every integer k from
−4 through+4. Negative k compares earlier teacher commands; positive k compares
later ones. Each action normally spans four emulator frames. This range is a
descriptive timing axis, not a hyperparameter sweep or a selected label shift.

Use identical anchors for all nine offsets. Every nine-action window must be
within one lane, one natural episode and one fixed50k-action quarter; contain
only four-frame transitions and no override, termination or truncation. Require
consecutive tick IDs. Report first50k and last50k quarters separately. The
primary subset has only NOOP/RIGHT/LEFT labels throughout the window; the
secondary subset permits FIRE. Also report the primary subset whose current
teacher label differs from the preceding label, with no outcome-based filtering.

For each subset and offset record exact matches, total anchors, agreement,
same-state agreement difference, teacher self-agreement e[t]==e[t+k], action
and label marginals, their independent-marginal match reference, and per-lane
counts. Preserve every anchor's key, nine actions and nine labels for recount.
No fitted parameter, hypothesis test, best-offset promotion or gameplay claim.

## Checks and interpretation

Before analysis, hand-built fixtures check lag sign, the identical-anchor rule,
episode/quarter boundaries, overridden and shortened transitions, FIRE filtering,
switch selection, and invalid/duplicate/gapped rows. Reconcile all action/frame,
episode and override counts with the original run_end and saved exposure audit.
Verify original log/source receipts and all new input hashes before and after.
One independent recount derives every offset's integer counts from saved anchors.
Use stdlib Python, one CPU process and a60-second analysis cap. Preserve failures;
never replace an incomplete result with a passing label or repeat completed work.

These are evolving sampled training policies on one reused seed, not the final
frozen greedy policy. Labels depend on executed-action feedback, and later labels
also depend on the current action's effects. Persistence, teacher oscillation and
state distributions can create shifted associations without causal latency.
Life boundaries are not logged here; natural episode boundaries are available,
and directional filtering does not guarantee exclusion of every life boundary.
Agreement does not identify the optimal command or measure task performance.
Keep143.667/70.750, the original864/90%-two-wall gate and native runtime unchanged.
No learner, teacher modification, loss change or automatic successor is selected.

Evidence root: `/x/Temp/kindle-breakout-label-timing-20261104`.
