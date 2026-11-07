# Breakout completed-return credit across lost lives

The conservative completed-return actor comparison fails its declared
learning-per-minute screen. Before another learner, measure how much of its
proposed positive credit can depend on rewards earned after the current life
is lost. This is within the existing five-game goal; it changes no policy,
reward, episode boundary, native code or training recipe.

Reuse exactly the480 outgoing states from the completed
[bonus-compatible target audit](../results/2026-11-03-cdp-completed-return-bonus-audit.md).
They belong to two previously selected B16/T16 windows per eight-lane group,
captured from the retained200k replay-value1 checkpoint on its sampled frozen
trajectory. They are not original training batches or states from the new
conservative-actor learner. Keep saved values, scales, actions, labels and
selection fixed. Values were approximately recovered from rounded native target
reports; retain that limitation and the fixed±.001 advantage sensitivity check.

Perform one new CPU annotation replay of all35,552 actions/142,169 frames.
Use the exact retained ROM/ALE/wrapper,full18/sticky.25/repeat4/noops0/native
pooled RGB protocol and recorded actions. Check every reward, frame count,
termination, truncation, reset and completed episode against the original log.
Match all eight input-pixel sequence hashes and512 selected input hashes against
the completed native capture. Record before/after life counts at each raw frame,
including gains and losses within one action. Life observations are diagnostic
only and never affect actions. Reuse the old replay/task ledger and captured
arrays; do not rerun a prior collector, model or fit.

For each fixed outgoing state, compute the original discounted external return
through natural game termination and a return stopping inclusively at the first
action containing a raw-frame life decrement, or at natural termination if it
comes first. Use the original F32 gamma1−1/333 with F64 accumulation, then F32
rounding as in the preceding audit. An action's complete reward remains assigned
to that action even if the life change occurs inside its four-frame interval.
Timeouts and unfinished segments remain ineligible. Life gains alone do not cut
a return. Distinguish raw-frame loss/gain counts from per-action net changes.

Require exact reproduction of all480 old rounded full returns, outgoing action
identities, values/scales and old390-positive/94.779770 weight totals. Independently
check backward recurrences against forward sums for all480 states, including
the decomposition into pre-cut reward plus discounted later-life reward. Since
rewards are nonnegative, the cut return and its clipped positive weight must
not exceed their full-return counterparts. Treat the unchanged combined-value
baseline as a conservative arithmetic threshold, not a life-value estimate.

Report complete/positive rows, clipped weight mass, lost positive rows and
removed weight fraction overall and per batch/lane. Report later-life reward
fractions with explicit denominators, and the fixed subset within16 outgoing
actions of a loss, retaining all other rows. Report teacher agreement only as
agreement with the imperfect pixel teacher. No coefficient, window, origin or
threshold search; no statistical independence or causal learning claim.

Before execution, qualify synthetic cases for terminal/reset alignment, inclusive
cut rewards, bonus lives, raw loss-plus-gain, missing tails, timeouts, discount
endpoints, nonnegative-return monotonicity and malformed inputs. Pin all new
scripts, protocol, receipts, source data and imported local helpers. Resolve
changed historical repository files to exact Git blobs rather than weakening
old hashes or editing historical manifests. Preserve failures with exclusive
output files. Bound annotation replay at180 seconds and analysis at60 seconds;
use one low-priority CPU core, block native learning imports and verify no GPU
descriptor opens. No GPU allocation, learner, integration or automatic successor
is selected. Root `/x/Temp/kindle-breakout-return-life-audit-20261103`.
