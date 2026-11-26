# Temperature0.5 does not improve the stronger Qbert checkpoint

Frozen action temperature0.5 scores **14,025 /22of24 first pyramids**, versus
the retained H5/R32 greedy result **14,050 /23of24**. The25-point difference
supplies no useful gain, and first-pyramid successes are lower. Both means
still miss the original15,000 gate. Retain greedy for this checkpoint; no
temperature sweep, extension, confirmation or automatic successor is selected.

| Action policy on the H5/R32 learner | Frozen mean | First pyramids |
|---|---:|---:|
| Retained greedy, prior assessment |14,050.000|23/24|
| Default sampling, exactly reproduced |13,331.250|23/24|
| Sampled temperature0.5 |14,025.000|22/24|

Both actual-initial default/temperature arms score177.083 with zero pyramids.
The four new arms select the first3 natural episodes per8 lanes=24 each, with
guidance/labels off, zero learner updates and no truncations. Excess completed
episodes and partial tails remain. The prior greedy assessment is the declared
primary control; it is not a new arm of this service. The original H5/R32
training comparison remains14,050 greedy/13,331.250 sampled after200k actions,
24,954 updates and **40m24s** training.

Temperature0.5 was fixed before launch, motivated by the earlier H15/R16
assessment. It changes action sampling only, preserving posterior sampling,
weights, preprocessing and checkpoint configuration. This is post-training
inference tuning on reused learner4013 and evaluation4000004013, not a new
learner, held-out confirmation or a revision of the original training primary.
The successful H15/R16 temperature option does not transfer as a score gain here.

All63 input pins,357 finite/invariant tensors per arm,config/source/counter
checks, complete frozen outcome replays/videos and host guards pass. Default
final/initial reproduce all **6,722 /1,138 semantic events** and their original
scores exactly. Each default arm and replay completes before its corresponding
temperature arm. Ordinary source logs lack live pixel hashes; outcome replay
does not compare against unavailable live images.

Four arms total **135,136 actions /540,407 emulator frames /81.325s** of
evaluation, with zero updates. The final-default and final-temperature arms
finish25 and26 natural episodes respectively, selecting24 in each. Both initial
arms finish24. No checkpoint tensors change.

Use the already qualified native`e416696`, SHA256
`40872a32541b9099c6a09d9e61348cc35a6a6c4fbf489fdf8c35140eecf1cdc7`, from
`/x/Temp/kindle-action-temperature-20261029/runtime`. No rebuild, production
change or repeated native qualification occurs. CPU preflight takes0.737s,
parses four commands,compiles three external scripts,checks all229 source-learner
pins and357 finite tensors in each source checkpoint,without constructing an
environment or native agent. Packaging and preceding experiment receipts pass
before declaration.

Service`kindle-qbert-h5-temperature-20261029` starts **16:42:40 UTC October29**,
source`81a0280`, and succeeds **16:45:24**, whole **2m43.706s** including CPU
audits. Terminal collection succeeds16:47:10.486. All stages succeed without
retries,cap changes or host recovery. No GPU allocation remains active at
completion. The five-game goal remains unfinished.

[Fixed protocol](../experiments/2026-10-29-cdp-qbert-h5-temperature.md),
[source learner](2026-10-29-cdp-qbert-h5-r32.md).
Evidence root:`/x/Temp/kindle-qbert-h5-temperature-20261029`; logs,weights,videos,
four arm reviews and the comparison remain outside Git.

| Receipt | SHA256 |
|---|---|
| Declaration | `db9a58d2833a21ffb59b873cc2742cb7c4294978ed7f21884e44020c92bba2c0` |
| CPU preflight | `dbded3b86138b0b9c88420ceb89de1064816d80d807ea1dd011b8b9e5e83d783` |
| Comparison | `ecce753e928c10ad4f605c070021330b4eee24eeb5fbdb26a85fc05372e490d6` |
| Completion | `511d595591a4c985a9ab1b93a4453c037053481c3278e322316caed02c4b0347` |
