# Sharper sampled Qbert actions give a small gain but still miss the gate

Frozen temperature0.5 scores **12389.583 /24of24 first pyramids**, a **3.38%**
higher point estimate than the retained greedy11984.375/24. It also improves
ordinary sampled6759.375/24 by83.29%, with zero new training. All means remain
below15000. Retain temperature0.5 as a useful12M development option, alongside
the established greedy result; this small selected-cohort difference does not
establish reliable superiority. No temperature sweep, learner extension, seed
confirmation or automatic successor is selected. The five-game goal remains
unfinished.

| Action policy on the retained12M learner | Frozen mean | First pyramids |
| --- | ---: | ---: |
| Retained greedy, previous assessment | 11984.375 | 24/24 |
| Default sampling, exactly reproduced | 6759.375 | 24/24 |
| Sampled temperature0.5 | 12389.583 | 24/24 |

The four new arms use final and actual-initial checkpoints, each with default
sampling or temperature0.5. Both initial arms score177.083 with zero pyramids.
Posterior sampling, weights, preprocessing and checkpoint settings stay fixed;
temperature acts only on the action distribution. Every arm selects the first3
natural episodes per8 lanes=24,guidance/labels off,zero updates/truncations,
with partial/excess episodes retained. The original15000/90%-pyramid gate and
the primary comparison with retained greedy remain fixed.

All57 input pins,357 finite/invariant tensors per arm,source/config/counter
checks,full action/outcome replays/videos and host guards pass. New default
final/initial traces match all4204/1138 retained semantic events and reproduce
their scores exactly. The greedy primary is the earlier retained evaluation;
it is not an additional arm of this service. Ordinary source logs lack live
pixel hashes. Four arms total **96,872 actions /387,366 emulator frames /
58.516s** of evaluation, with zero learner updates. Original training remains
200k actions /12,477 updates /30m27s. These are reused development weights and
evaluation lanes, not another learner or independent confirmation.

Service `kindle-action-temperature-full-20261029` runs14:08:10–14:10:15 UTC
October29,source `4b24f75`,whole2m04.158s including CPU audits. The service and
terminal collector succeed,without GPU repeats,cap changes or host recovery.
No GPU allocation remains active. Root
`/x/Temp/kindle-action-temperature-20261029/full` retains the four logs,
checkpoints,videos,per-arm reviews and completed comparison outside Git.

- Full declaration: `d3c8bca87b9949d350efce751b209f721a1fc03490cc3654069d15ce0aa566c3`
- Comparison: `a746c77d45ef7e019600da8dd9ba2debc077dbbe9854dc30d8388e5e2dc2e441`
- Completion: `c2964928da86d26bbc0523dedbc5078a29180fc8f2795f47e49517a3734d3bcc`

## Bounded native checks

The optional native action temperature passes its guarded probability/RNG/reset
checks and a default vector/serial parity check. Both1024-action Qbert restores
complete with4096 frames,zero updates and357 finite/invariant tensors. Default
sampling takes.811s and temperature0.5 takes.813s. All128 default transition rows
exactly match the retained12M sampled evaluation. No natural episodes complete;
these checks establish execution and compatibility, not task benefit.

Temperature changes the actor logits before softmax and the checkpoint's uniform
mixture. Posterior sampling stays unchanged. The option requires sampled frozen
restore in the CLI/auditor, positive normal finite F32 input and selection before
the first observation. Both native learning entry points reject the selected
evaluation option, including explicit temperature1. It is not checkpointed.
Unspecified temperature preserves the existing shader arithmetic and RNG draws.

The GPU test compares selected actions with an independent F64 probability
calculation using known logits, temperatures including.5/1/2 and the valid F32
extremes, masks, greedy selection and multiple fixed draws. It also checks native
action RNG advancement, unchanged posterior RNG during action selection,
episode resets, invalid/late selection and learning rejection. The existing
vector/serial parity test exercises the default belief and action path.

All136 Rust CPU tests,658 packaged Python tests and strict native/binding Clippy
pass. The original packaging controller failed only when finding the test
executable: its adapted build target and copied lookup path disagreed. A separate
CPU repair uses the already compiled executable; no build or GPU stage repeats.
Original script,error,successful stage receipts and repair remain preserved.

Service `kindle-action-temperature-check-20261029` starts14:02:42 UTC October29
and succeeds14:02:57 after15.363s,source `e416696`. All52 pins,source/config/counter
checks,complete bounded action/outcome replays and four host guards pass. The
terminal collector succeeds. No faults, cap changes or driver recovery occur.
The857,476,155-byte build cache is released;wheel,validation executable,runtime
and evidence remain. The original native447fe50 and completed runtimes remain.

Runtime `/x/Temp/kindle-action-temperature-20261029/runtime`, native SHA256
`40872a32541b9099c6a09d9e61348cc35a6a6c4fbf489fdf8c35140eecf1cdc7`.
The packaging receipt pins native,wheel,test executable and Python sources.

- Declaration: `16380c8cdbf8b5f1a45389f5d25adafbb1f9386f7df9ddfef9f1614b1fd3a21d`
- Bounded review: `1e93104d43d6f7c32e4896ac680181e5c8e24fdaafa9cd80d209cdd7293de011`
- Completion: `e4ce26c8b192e9cb156179a6b3daac316ac9b76286bcc7b1780dc8bea2008be0`
- Packaging: `f44880e3058599bf91bb891480f615e71ee089cb486042b324f001dcb4ecf1a4`

The retained12M learner's original200k-action/12,477-update training cost remains
30m27s. Its greedy11984.375/24 and sampled6759.375/24 both miss mean15000.
The four-arm assessment above was separately selected after these checks;
see the [protocol](../experiments/2026-10-29-cdp-qbert-action-temperature.md).
No bounded experience was reused for training.
