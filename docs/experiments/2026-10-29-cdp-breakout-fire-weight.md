# Bounded Breakout FIRE-weight check

The [timeout replay](../results/2026-10-29-cdp-centroid-timeout.md) establishes
a long failure to serve. The subsequent saved-log audit finds all200k states
labeled and99.805% of eligible sampled replay rows contributing imitation.
FIRE appears23,317 times; late passive sampled agreement is2081/2656 (78.35%),
but the longest passive FIRE-label run without a FIRE-bearing action is45.
The frozen greedy timeout enters a much longer waiting interval. This supports
a targeted serving-margin experiment, not a claim that labels are absent or
that class weighting is sufficient for strong ball control.

Select one fresh bounded check:12M64/Auto/N8/B16/T16/H15/R16, expert scale100,
world gradients, corrective labels,20% episode guidance/cap1920, actor-unimix.01,
centroid scale100, original rates and full18 actions. Change only the exact FIRE
class's imitation weight1→4; all other weights stay1. Native2974160 and the
qualified centroid runtime remain unchanged. A new optional CLI flag passes
the existing native action-weight field; default and frozen restore behavior
stay fixed. Weighting affects both actor and world-state supervision and changes
total imitation magnitude; it is not normalized within a minibatch.

Before GPU execution, test the CLI configuration boundary, legacy defaults,
FIRE meaning lookup, finite/product validation and refusal of frozen overrides.
Validate the actual two commands with stub environments/constructors, compare
the full generated config, and derive the372-tensor schema from the corrected
prior bounded receipt. Pin scripts, sources, runtime, schema and evidence.

Train8192 fresh aggregate actions /489 updates with seed1009 and a240s cap.
Require correct cadence, finite world/behavior/centroid reports, nonzero FIRE
targets, target-weight sums compatible with `labeled_rows + 3*fire_rows`, and
changed CNN/reward/actor/value parameters. Compare all372 initial tensors to
the prior centroid check and357 shared tensors to the retained control, plus
the complete pre-update semantic trace. No previous weights or experience are
reused. Then run1024 frozen greedy actions with seed4000001009 and a60s cap,
requiring zero updates and all372 tensors invariant. Fully CPU-replay training
pixels, labels, actions, rewards, resets and frames. Each GPU stage has a host
guard; stop on failure without retry or cap changes.

A single480s service owns both stages and the180s CPU audit. Require a
conservative projection `training_seconds * 200000/8192 * 1.10` below30 minutes
before considering a separately declared200k learner. The bounded check alone
measures execution and cost, not serving reliability or gameplay. No full learner,
coefficient sweep, evaluation intervention or automatic successor is selected.
The original centroid full learner remains `closed_protocol_failed`.

Log audit: `/x/Temp/kindle-breakout-fire-supervision-20261029`.
Bounded evidence: `/x/Temp/kindle-breakout-fire-weight-20261029`.
