# Breakout replay-value1 passes execution and early cost checks

The [bounded check](../experiments/2026-11-02-cdp-breakout-replay-value1.md)
completes **8,192 actions /32,727 frames /489 updates in47.689684s**. Its
conservative200k projection, including10% margin, is **1,280.728814s
(21m20.729s)**, below30min. The1,024-action /4,096-frame frozen restore takes
.871872s, makeszero updates and completesno natural episodes. This establishes
execution and early cost, not gameplay benefit or task competence.

Only replay-value loss scale0.3→1.0 changes from the retained behavior8e-5
control:59.000 sampled/117.292 greedy, bothzero of24 two-wall successes.
Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16, base/encoder4e-5, behavior8e-5,
dynamics4e-4, sampled collection/posteriors, actorunimix.01, expert100/unit
weights/world gradients/all-state corrective labels and20% episode guidance/
cap1920. Native447fe50/SHA35ad0db, teacher, full18/sticky.25/repeat4/noops0,
native pooled RGB and one GPU resize remain unchanged.

The existing coefficient weights critic fitting and representation gradients
through frozen value logits. It retains real rewards/boundaries and imagined
bootstraps. Neither an isolated critic cause nor a missing-loss defect is
established. The preceding critic diagnostics concern neighboring behavior4e-5
controls, and Qbert's development gain did not confirm on its fresh seed.
Breakout benefit therefore remains a separate empirical question.

## Completed evidence

All **320 pins**,357 finite control-matching initial and invariant frozen
tensors,47 matching pre-update semantic events, correct cadence,489 finite
supervised reports, aligned world/behavior imitation values and changed
CNN/reward/actor/critic weights pass. Full training pixel/action/teacher-label
replay and both host guards pass. Training executes3,902 actual overrides;
guided returns are not unaided competence. There are16,334,353 parameters.

Candidate runnerSHAcf4ea238 adds only ten lines to the exact control6c1b69fb
runner, exposing the already-qualified fresh-only replay-value option. Defaults
stay unchanged. All23 relevant private-runner loss/rate CPU tests pass, followed
by2.186s command/config/schema/cadence preparation (2.218s wrapper): two actual
commands,17 config cases and all282 control pins. No native module or emulator
is constructed by preparation. No production or native source changes occur.

The original mainPID887239/invocation`491916f869dd4cbfa75b236f3a82f0dd` launches
**03:01:28.714 UTC November2/source70b7567** and succeeds **03:02:34.926**.
Original collectorPID887242/invocation`e5e1a7157da34264a98ccafd99a37ae7`
writes completion **03:02:40.372** and is terminal success. Its journal has no
separate resource-consumption timestamp; original identity, completion payload
and successful terminal state are verified.

The separate terminal reader runs once and passes **03:03:34.459** in1.289s
(1.308s wrapper), verifying all inputs, completion files, original journals,
the replay and projection arithmetic. No preparation, GPU, replay, audit or
collection stage fails or repeats. No cap or setting changes after selection.

Root `/x/Temp/kindle-breakout-replay-value1-20261102/native-check` is closed;
trial `breakout-12m-64-b16-r16-replayvalue1-expert100-check`. Do not repeat this
check, collector or reader. No active Kindle allocation remains at closure.
A full learner requires a separate selection; no automatic promotion, extension,
retry, sweep or check-experience reuse. The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`5106a120d82d3711c3d3a53b0bb401b5a4493bf757448ce82b77ee8304ec1228`|
| Review |`0eab7daf89deb10670dc2d01d621e35ab7f12565e69a1bf5e9ff5a07751f624d`|
| Original completion |`56cd5cbacddff169edc5a210fc4e55161b4b5012cd9ee409f4075efed81e027a`|
| Terminal summary |`cfc4b63918f5b0e2a0cbe81062c17eda54cc28084809887d160b5a642c8e6ba3`|
