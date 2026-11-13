# Larger effective batches at the retained Breakout sequence length

Select one fresh bounded B32/T16 check after the completed
[discount100 regression](../results/2026-11-04-cdp-breakout-discount100.md).
Retain horizon333, greedy143.667/sample70.750 and the original mean864/90%-two-wall
gate. No learner is active at selection.

Learning accounts for roughly85% of retained training time. The
[B32/T8 check](../results/2026-11-04-cdp-breakout-b32-t8.md) kept256 replay
positions and the update cadence almost unchanged; learning time stayed near
40.5s and it missed its cost screen. This check instead retains T16 and doubles
the effective batch256→512 positions, halving steady-state optimizer updates
at ratio16. It tests amortizing update overhead without shortening temporal
gradients. Neither faster execution nor better learning is established.

Change only batch_size16→32 and world_microbatch_size16→32 relative to retained.
Keep centeredCDP12M64/Auto/N8/T16/fullBPTT/H15/R16/context1, base/encoder4e-5,
dynamics4e-4, behavior8e-5, horizon333/lambda.95, policy/value/replay-value1,
representation.1, dynamicsKL/free-nats/continuation1, CDPfuture500 and disagreement1.
Keep expert100/unit weights/world gradients, all-state corrective labels,
20% episode guidance/cap1920, sampled collection/posteriors, actor-unimix.01,
replay100k and full18/sticky.25/repeat4/noops0. Native pooled RGB receives one
GPU64 resize. No optional crop, reward or additional loss.

Batch diversity, centered-target statistics, optimizer cadence, update-based
warmup and target rows240→480 change together. This is a learning configuration
experiment, not an exact backend optimization. Keep learning rates and warmup
updates unchanged; any quality comparison must include these consequences.
Native447fe50/SHA35ad0db, exact retained runnercf4ea238 and teacher stay fixed.
No build or production source change is needed.

Fresh seed1009:8,192 actions, first update632,237 updates, residual credit.25;
training cap180s. Follow with1,024 greedy frozen actions, seed4000001009, cap60s,
zero updates and357 invariant tensors. No previous weights or experience transfer.
Require357 finite/matching initial tensors,47 common pre-update semantic events
through376 actions, finite correctly weighted reports, exact32-action cadence,
label bounds480, changed CNN/reward/continuation/actor/value, complete training
pixel/action/label/override replay and both host guards. Short frozen actions
test restore/execution only, not natural-episode competence.

CPU preflight must exercise both actual queued commands with native/emulator
construction intercepted, derive cadence from the pinned scheduler/replay source,
reject wrong config/cadence/labels/replay paths and validate the original terminal
collector on saved fixtures. Preserve prior sources and failures; no completed
stage repeats. Replay cap120s, audit180s, main600s, terminal observer650s. Check
live service/render-holder state before launch; no GPU overlap or host recovery.

The locked early cost screen requires training time at most95% of retained
47.689683854s and time×200000/8192×1.1 at most30min. Report construction and
stage timing separately. The later warmup means the candidate does slightly less
sample work within8,192 actions; report actual update/row counts. This historical
single timing is a development screen, not a reliable sustained-speed estimate.

Only after execution and cost pass may a fresh full comparison be selected
separately:100k/3,106 recovery,200k/6,231 final, four final/actual-initial sampled/
greedy24-natural-episode assessments, no bounded experience reuse. Favor only
higher raw greedy score and own-initial gain/min with a positive paired lane
interval. No threshold change, batch sweep, retry, extension or automatic successor.

Root `/x/Temp/kindle-breakout-batch32-20261104`; service
`kindle-breakout-batch32-check-20261104`. Five-game goal remains unfinished.
