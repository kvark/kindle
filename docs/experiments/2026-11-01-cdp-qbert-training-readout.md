# Frozen Qbert actor preferences on recorded training trajectories

The first-disagreement interventions identify five locally costly choices under
teacher continuation. A subsequent log-only recount finds an exposure imbalance:
wholly unguided training episodes contain only 159 actions at episode age1920+,
versus12,782 native actions after guided episodes release control. All200k actions
already have teacher labels. Later movement agreement in the last50k actions
is63.84% within unguided episodes and36.98% after guidance releases; these are
the changing training policy's executed actions, not final frozen preferences.

Select one frozen diagnostic within the five-game goal. Restore the retained
Qbert encoder4e-5 final checkpoint in the qualified447fe50+readout runtimef864fe4.
Reproduce all200k recorded training actions /799,739 emulator frames, preserving
the original eight lanes, seeds, resets, full18 actions, sticky.25, repeat4 and
native pixels with one GPU resize to48. Override every submitted command with
the original recorded command. Read the already computed actor features/logits
after act and before observe. No learning call is permitted.

This evaluates fixed final weights on recorded training histories. It does not
reproduce the original changing weights, hidden states or sampled replay batches.
The actor still computes before the command override; copied logits therefore
describe its preference at the reconstructed belief. Keep ordinary sampled
categorical posteriors, actor-unimix0 and greedy action calculation. The actual
submitted commands remain the originals, regardless of logit argmax.

Report teacher agreement, mean teacher probability and cross-entropy for all
actions and the fixed last50k. Split input states before/after the independently
replayed first-pyramid event; distinguish forced actions, all passive decisions,
wholly unguided episodes and released guided episodes. Within each group report
all labels, movements and waits. Include the original executed-action agreement
on exactly those rows. Check row counts and original agreements against the
separate verified phase recount. These are exposure groups, not a sweep.

Require all original input-pixel hashes, submitted actions, rewards, boundaries,
cumulative frames and reset behavior to match. Training logs have no separate
live output-pixel hashes; subsequent input hashes validate nonterminal outputs.
Check357 invariant checkpoint tensors, zero updates, finite features/logits and
unchanged counters across each readback. Repeat readback every256 vector ticks.
At the first128 decisions, check F64 actor probabilities from copied features
and frozen weights against native logits within the existing1e-4 bound.

Run one256-action smoke and one full200k capture, with exact prefix logits and
qualification arrays required to match. Use the existing host guard for each.
Caps:120s smoke,300s full capture; guarded process timeouts150s/360s and a600s
service bound. Four pure CPU analysis tests and the1.765s preflight pass, with
no emulator or native-model construction. No native code/build is changed.

Roots:
`/x/Temp/kindle-qbert-training-coverage-20261101` for the.909s log recount, and
`/x/Temp/kindle-qbert-training-readout-20261101` for the frozen readout. The recount
verifies all200k actions against the existing phase counter and immutable replay
outcomes. None of the25 disagreement-origin full-frame hashes occurs identically
in the training log; exact image equality is not semantic or recurrent-state
coverage, and absence of a match proves no information deficit.

The teacher is imperfect, trajectories and contexts differ, and selected frozen
failures are not a representative held-out accuracy sample. Differences from
logged replay loss cannot isolate generalization or recurrent-state mismatch
without the missing replay indices/contexts. This test narrows whether the final
actor fits the collected live histories; it does not select a loss, teacher
weight, replay change, new learner, retry or automatic successor. The five-game
goal remains unfinished.

## Completion — November 1

Source5d53dac,562 pins. Smoke4.708s/full128.729s;200k original commands and
799,739 frames match. All357 invariant tensors,128 F64 actor comparisons,
98 repeated full readbacks,exact smoke/full prefix,phase recounts and guards
pass. Service succeeds05:56:14 UTC after139.692s;original terminal collection
passes. Later movement agreement on the last50k training actions is96.92%
forced,85.56% unguided and75.10% after guidance release. See the
[result](../results/2026-11-01-cdp-qbert-training-readout.md) for distributions
and limits. Zero updates,failures,retries or automatic successor.
