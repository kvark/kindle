# Frozen posterior-mode comparison on retained Breakout replay-value1

Select a two-arm frozen comparison of the retained200k replay-value1 checkpoint,
which scores143.667 with greedy actions and sampled posterior states. Compare
sampled versus modal categorical posteriors, keeping actions greedy in both.
This asks whether removing live posterior sampling helps this trained policy.
It selects no learner, loss change, initial-model assessment or training-cost gain.

The [older Breakout comparison](../results/2026-10-29-cdp-greedy-posterior.md)
remains protocol-failed: its trained17.875→18.833 change is not an accepted
four-arm improvement because the initial modal arm has five timeouts. Qbert's
modal comparison also lowers scores. Neither predicts the stronger current
checkpoint's behavior. Do not restart either old assessment or reinterpret its
failure. Compare saved initial tensor values on CPU for provenance, without
repeating the failed initial arm or assigning it an accepted score. This new
assessment concerns trained inference only, not paired learning gain.

The current encoder8e-5 learner is closed and rejected. Retain encoder4e-5/
behavior8e-5/dynamics4e-4/replay-value1 and original200k/12,477-update training
cost21m27.910s. Source:
`/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009/final`.
No parameters, configuration, optimizer state or training counters may change.

## Fixed evaluation and decision

Two arms, sampled posterior first and modal posterior second. Each restores
the same final checkpoint, with greedy actions and sampled-policy training
history. Use eight independent lanes, seed4000001009, full18/sticky.25/repeat4/
noops0/native pooled RGB/one GPU64 resize. No guidance or expert labels.
Select first3 natural episodes per lane=24; retain all extra episodes/tails.
Each arm has600k-action/300s cap, zero updates and zero truncations required.
An incomplete cohort, timeout or invariant failure rejects the assessment;
no retry, cap increase or post-outcome cohort replacement.

The sampled-posterior arm must reproduce the retained greedy score, full
semantic action/episode/reset traces and final counters/outcomes/tails, excluding
finite wall times and source/path identity. Audit this compatibility arm before
modal execution. Require357 finite/invariant tensors per arm, exact source/
config/checkpoint/counters, host guards and2GiB headroom. Replay every recorded
action, reward, frame count, reset and task outcome, retaining lane0 videos.
Ordinary frozen logs lack live pixel hashes; no unavailable live-image match
is claimed. Replay cap300s per arm. Replay an already produced protocol-failed
arm once before rejecting it, if its runtime outputs are otherwise auditable.

Report absolute means, all24 returns/lane means, two-wall successes and the
original mean864/90%-two-wall gate. Favor modal only if its mean exceeds the
sampled-posterior control by at least10%, the paired lane-bootstrap95% return
interval lies above zero, and it has no fewer two-wall successes. Use10,000
resamples, seed0. Keep evaluation action/frame/time cost separate from unchanged
training cost; no learning-gain/min claim without an accepted modal initial.
One reused checkpoint/cohort cannot establish seed reliability or a global
inference default. Changed posterior states do not isolate perceptual precision
or the mechanism of any score difference.

## Qualified runtime, CPU checks and closure

Reuse nativeb76f91c/SHA7555bf9 and exact runnerfac1ff7 from
`/x/Temp/kindle-greedy-posterior-20261029/resumed/synchronized/runtime` and the
preserved Breakout assessment. Original modal argmax/RNG/reset/learning-block
checks, default parity and135 Rust/625 Python CPU checks/Clippy remain the
qualification; no rebuild or repeated native test. Resolve the one historical
runner pin using exact saved bytes. Verify69 qualification and338 retained
inputs, retaining the separate historical replayer resolution. Original
qualification test/readback and assessment failures remain unchanged.

Before launch parse both actual commands with native/emulator construction
blocked, verify source checkpoint hashes/config/357 finite tensors, reject
training/modal-mode misuse and altered header/counters, and exercise score/
cohort/selection arithmetic. Keep actual runtime sources immutable. Require
clean committed source, idle Kindle services and no GPU descriptors at launch.
Main cap1800s, original terminal observer1860s. Stop on real host or numerical
faults; no driver recovery or repeated successful stage.

Root `/x/Temp/kindle-breakout-value1-modal-20261104`, service
`kindle-breakout-value1-modal-eval-20261104`. Pin commands, helpers, protocol,
runtime, final checkpoint, original learner/qualification closure and old failed
assessment receipt. Preserve original service identity, failures and terminal
artifacts. No initial arm, new learner, inference sweep, extension or automatic
successor is selected. Breakout and the five-game goal remain unfinished.
