# Fresh 200k Breakout encoder8e-5/current-critic comparison

After the [bounded check](../results/2026-11-04-cdp-breakout-value1-encoder8e5.md)
and its original collector complete, select one fresh200k learner. The check
takes48.178s for8192 actions/489 updates; padded200k projection21m33.842s passes
30min. Initial tensors/pre-update events, frozen invariance, full replay and
host guards pass. Execution and early cost do not establish gameplay benefit.

The earlier encoder8e-5 result improved greedy104.667→125.375 at similar cost,
but sampled play stayed flat. Its behavior4e-5/replay-value.3 differs from the
current behavior8e-5/value1 recipe. Test this combination against the retained
single-image control143.667 greedy/70.750 sampled in21m27.910s; all0/24 two-wall.
The preceding frame-pair negative stays closed. This is not a broad rate sweep
or a repetition of either completed learner.

## Locked learner and assessment

Only encoder4e-5→8e-5 changes from retained control. Keep centeredCDP12M64/
Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16/context1, base4e-5/dynamics4e-4/
behavior8e-5, policy1/imagined-value1/replay-value1, discount333/lambda.95,
expert100/unit weights/world gradient, all-state corrective labels,20% episode
guidance/cap1920, actor-unimix.01, sampled collection/posteriors, replay100k/
warmup1000 and full18/sticky.25/repeat4/noops0. Preserve one GPU resize of native
pooled RGB.16,334,353 trainable parameters; no native/runner/teacher change.

Fresh seed1009, without previous weights, replay, games, beliefs or RNG.
200k actions/12,477 updates; first update376/residual credit.5. Training cap1800s.
Save actual initial,100k/6227 recovery and200k final checkpoints. Verify recovery
once independently while the original learner continues; retain read-only files.
Checkpoints preserve weights and optimizer moments, not complete live resumption.

Assess final and actual initial in both greedy and sampled action modes at
seed4000001009. Guidance/labels off; zero updates; first3 natural episodes per8
lanes=24 in every arm. Each arm has600k-action/300s cap; no truncations accepted.
Do not increase caps or replace failed/unfinished episodes with favorable ones.
Preserve excess natural episodes and partial tails. Require357 invariant tensors
per arm, bounded/control initial equality,47 pre-update semantic events and
actual-initial evaluation semantics matching retained control, except finite
episode wall times. Full recorded-action outcome replays and videos are required;
ordinary frozen logs have no live image hashes, so no unavailable pixel equality
is claimed. Full training replay verifies all pixels, actions and teacher labels.

Primary: greedy final minus actual initial, divided by training minutes.
Favor candidate only if mean greedy return and mean gain/min both improve and
the paired lane-bootstrap95% gain/min interval lies above zero. Use10,000 lane
resamples/seed0. Report sampled results, absolute scores and original mean864/
90%-two-wall gate regardless of outcome. This reused seed/cohort and historical
control timing do not establish independent reliability or an isolated mechanism.
Guided returns and replay CE are descriptive, not frozen competence.

## Execution and closure

Root `/x/Temp/kindle-breakout-value1-encoder8e5-20261104/full`, trial `candidate`;
service `kindle-breakout-value1-encoder8e5-200k-20261104`, plus original recovery
and terminal observers. Runtime `/x/Temp/kindle-breakout-corrective-20261028/runtime`:
native447fe50/SHA35ad0db, runnercf4ea238, exact retained teacher/schedule.
No build is needed. Require idle GPU/service state, clean source and fixed pins.
CPU qualification covers actual five-command configuration, checkpoint schema,
strict changed-header contracts, comparison entry with synthetic outcomes,
invalid settings and replay inputs. Keep null journal-message and cleared
terminal-invocation handling qualified.

Host guard and2GiB headroom floor apply to every GPU stage. Training replay
cap900s/process930s, frozen replay500s, training audit1000s/frozen audit600s,
final audit300s; metrics/plots/comparison120s each. Main7200s/collector7500s;
recovery observer2000s. Stop on real host or numerical faults. Preserve any
original failure; no repeated GPU or already successful CPU stage. No cap
increase, extension, second learner, experience transfer or automatic successor.
Keep source, declarations, logs, checkpoints, videos, curves, exposure, coverage,
comparison and original terminal receipts outside Git. Report observer/co-host
CPU time without correcting training wall time. Update PR31 and current docs;
the five-game goal remains unfinished until its original gates are handled.
