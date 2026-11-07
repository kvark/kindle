# Breakout: compare action-conditioned errors on retained evidence

The [motion-rollout learner](../results/2026-11-07-breakout-world-motion-rollout.md)
improves recorded eight-step tracking but fails fixed-action distribution gates.
Before another learner, inspect evaluation contracts, comparable metrics and
training support. Root `/mnt/data/kindle-breakout-world-action-gap-20261107`.
CPU analysis only: no model query, training, emulator construction or collection.

## Fixed scope

Use the coverage and motion-rollout final checkpoints' retained outputs, all
42 oracle plans across seven categories, histories16/32, eight draws, and all
1/4/8/16/32 horizons. Dense fitting/development retain the original first draw.
These are reused development data; snapshot/draw repetitions are not independent.
Do not change, replace or reinterpret failed retention gates as passed.

Verify shapes, action/order/normalization construction and source identities.
Check saved first32-step model images against their five stored endpoints, and
oracle first native sequences against repeat0 endpoints. Recompute saved
ball/paddle features from retained RGB and require exact equality. This is source/
data provenance, not proof of actual native argument buffers that were not saved.

Compare the same conservative visible-ball position metric at every horizon:
fixed model0/oracle0 and all64 model/oracle sample pairs, with original arrival
validity, terminal arrivals included and unavailable images excluded. No best
sample/oracle matching. Report support, detection and within1px counts by
category/plan and pooled; raw RGB metrics omit survival weighting and do not
replace the separate event/energy distribution scores. Unsupported cells stay
untested. Also expose the exact horizon/metric differences in previous summaries.

Inspect a predeclared action-no-effect subset: freeflight, horizons1/4/8, four
plans and every source lane. Require all32 plan/repeat oracle arrivals available,
nonterminal, with finite identical native ball positions and matching visible
RGB centers across plans/repeats. Report per-origin modeled ball presence and
mean-feature variation across actions, with paired-draw dispersion and its
finite-draw correction. Center action contrasts within each shared draw before
the correction, respecting common random numbers. Keep negative estimates.
No causal attribution to training or acceptance from constant/absent predictions.

Measure literal constant0/3/4 and reverse-plan support in the retained raw fitting/
development streams and all65,536 declared H16 training suffixes. Separate waits
from moving-ball/freeflight/last-life states using retained analysis labels;
labels never enter training. Inspect32-command raw runs separately: the learner
directly supervises only16 future arrivals. Check episode/reset boundaries and
original seed families. Do not assert all fixed plans are out of distribution
merely from their names, or call action-feedback confounding established.

## Reviewable clips and receipts

Use already-retained `first_native_sequence` and `first_prediction` fields for
full causal clips, without rerunning a scored query. Fixed cases: freeflight/noop,
contact/left and life_loss/noop, lane0, oracle repeat0/model draw0, history16.
Show oracle, coverage and new model with action/step and survival-weight labels.
Native oracle and RGB64 models are resized for display to a common physical
aspect; these display operations never enter model input or scored metrics.
Stop at the first unsupported oracle arrival, retaining terminal arrival.
Distinguish independently sampled futures from matched/optimal draws. Record
actual supported decisions; quote original frame counts only where derivable,
otherwise a repeat4 upper bound. No new emulator action/frame.

Pin exact scripts/protocol, source helpers, corpus/plan/manifest files, original
closed receipts and all selected raw outputs before execution; rehash afterward.
Preserve original protocol/CPU failures without modifying their receipts. Newly
pinned raw outputs are current authoritative bytes; do not claim a previous
closure hashed every raw RGB array. Supplement them with endpoint/feature and
published-summary consistency checks. No automatic learner, collector, scalar
sweep or actor/critic follows. Full world-model competence remains unproven.
