# Native action temperature passes bounded frozen checks

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
One fixed four-arm frozen assessment is selected separately after these checks;
see the [protocol](../experiments/2026-10-29-cdp-qbert-action-temperature.md).
There is no temperature sweep,new learner,extension or automatic successor.
The five-game goal remains unfinished.
