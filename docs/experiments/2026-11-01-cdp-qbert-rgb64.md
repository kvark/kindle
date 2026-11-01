# Bounded Qbert12M/RGB64 execution and cost check

Select one bounded RGB64 check of the retained supervised 12M/H5/R24 recipe.
Qbert's later-game quality still varies across learners. Higher spatial detail
is a candidate, not an established limitation or expected improvement. The
earlier unaided 4M/H15/R32 RGB64 comparison scored lower than RGB48 and took
13.14% longer; retain that negative result. This check uses the later 12M model,
encoder4e-5, expert100 and greedy collection, so it tests a different recipe.

Relative to the retained seed10007 control, change RGB48 → RGB64 only as a
learner setting. Keep centered CDP, 12M/Auto/N8/B16/T16/full BPTT/microbatch16,
H5/R24, base4e-5/encoder4e-5/dynamics4e-4/behavior unset, warmup1000, greedy
collection, sampled posteriors, actor-unimix0 and action-effects1. Keep
expert100/unit weights/world gradients/all-state corrective labels, 20%
episode continuation guidance/cap1920, full18/sticky.25/repeat4 and replay100k.
Preserve one GPU resize directly from native pooled RGB; no CPU resizing.

Use native447fe50 and runneraba4300d with the qualified equivalent faster
teacherfa261c53 at `/x/Temp/kindle-qbert-color-masks-20261101/runtime`.
No native or production change/build is required. The original RGB48 control
used the slower teacher implementation; timing is not an isolated geometry
effect. Learner10007/evaluation4000010007 are reused development inputs.
No earlier weights, replay, beliefs, games or RNG state are reused.

RGB64 raises encoded width from 576 to 1,024 and trainable parameters from
14,123,473 to 16,334,353 (world14,688,256 plus behavior1,646,097). CNN output,
posterior and prediction-head shapes and initialization change together.
No isolated spatial-information, parameter-count or reliability claim follows.

Fresh training is fixed at 8,192 actions / 733 updates, first update376 and
residual credit .75, with a 180-second training cap. Follow with 1,024 greedy
frozen actions / zero updates, capped at 60 seconds. Require all 357 initial
tensors to match the independently derived native schema, finite reports and
changed CNN/reward/actor/value parameters, invariant final/frozen tensors,
exact configuration and supervision accounting, full training pixel/action/
corrective-label replay and both host guards. Do not require cross-geometry
initial-value or initial-action equality. Audit cap180s; service cap480s.

Before launch, CPU qualification exercises both actual CLI paths, configuration
rejection cases, schedules and schema checks without a native agent or emulator.
The shape derivation is checked against all 357 retained Qbert12M48 tensors
and all 357 retained Breakout12M64 tensors. Pin runtime, runner, teacher,
controllers, source definitions, protocol and completed control evidence.
Require the preceding encoder8e5 service terminal and the GPU host idle.
Preserve failures and never repeat completed GPU stages.

The cost screen is
`check_training_seconds * 200000 / 8192 * 1.10 <= 1800`.
A pass establishes execution and early affordability only. A full learner
would need separate selection, a fresh 200k start, 100k/9,340 four-file recovery
and final/actual-initial greedy-primary plus sampled 24-natural-episode arms
against retained14,339.583/13,761.458. Keep the original mean15k/90%-first-pyramid
gate. No full learner, experience reuse, extension, cap increase, resolution
sweep, retry, confirmation or automatic successor is selected by this check.

CPU preflight passes at 19:31:37 UTC on November 1 in 1.161s (1.196s wrapper):
two CLI paths, 19 configuration cases, the 8192/100k/200k schedules, both
357-tensor reference geometries, 442 retained control inputs and 16 runtime
package files. No native agent or emulator is constructed. Actual candidate
execution and cost remain unmeasured.

Root `/x/Temp/kindle-qbert-rgb64-20261101/native-check`, trial
`qbert-12m-64-b16-h5-r24-encoder4e5-greedy-expert100-check-seed10007`, service
`kindle-qbert-rgb64-check-20261101`. The five-game goal remains unfinished.
