# Breakout temperature0.5 improves sampling but trails greedy

Frozen temperature0.5 scores **83.292**, below retained greedy **104.667** by
20.42%. It improves on default sampled56.292 by47.96%, but every arm still has
zero two-wall successes in24 natural episodes. Retain greedy as the stronger
inference option for this checkpoint. The learner's original sampled-primary
result remains56.292; this separate inference assessment does not change it.

| Action choice | Final mean | Own actual initial | Two-wall successes |
|---|---:|---:|---:|
| Greedy, retained primary comparator | 104.667 | 2.208 | 0/24 |
| Temperature0.5 | 83.292 | 1.333 | 0/24 |
| Default sampled, reproduced | 56.292 | 1.417 | 0/24 |

Both new default-sampled arms exactly reproduce the retained semantic sequences
and scores:4,394 final and799 actual-initial transition/episode/reset events.
The temperature change affects action logits before softmax and the unchanged
0.01 uniform action mixture. Posterior sampling stays unchanged. No training
updates occur, and all357 tensors remain finite and invariant in every arm.

The original training cost remains21m28s for200k actions/12,477 updates.
Temperature0.5 gives3.819 own-initial-subtracted return per original training
minute; it does not improve the retained greedy choice. Four new frozen arms
total **81,016 actions /323,877 frames /72.156s**, withzero updates and truncations.
The original greedy arms are retained without repeating them. No temperature
sweep, new learner, extension, cap increase or automatic successor is selected.

## Scope and verification

Use the retained encoder4e-5 final and actual-initial checkpoints on reused
learner1009/evaluation4000001009. Keep centeredCDP/12M64/Auto/N8/B16/T16/H15/R16,
expert100, actor-unimix0.01, full18/sticky0.25/repeat4 and one GPU resize to64.
Guidance and labels are disabled. Each arm selects the first three natural
episodes per eight lanes, with extra episodes and partial tails retained.
The unchanged task gate requires90% both-wall/864-point successes.

The qualified temperature runtimee416696/SHA40872a3 and exact runnerabc2c2b
differ from the training runtime447fe50. Both exact default-sampled compatibility
checks pass, but this remains one reused development checkpoint/cohort, not
independent reliability or an isolated architectural explanation. Ordinary frozen
logs lack live pixel hashes.

All348 pinned inputs,357 finite/invariant tensors per arm,full recorded-action/
outcome/frame/reset replays and videos,host guards and original terminal identity
pass. Main service succeeds **09:27:46 UTC November1**, sourcea285376, after
2m21s; terminal collection succeeds **09:28:08**. MainPID776380/invocation
`33ea1a8cc3494d618ebffbe63ef1fa92` and collectorPID776383/invocation
`b96831bc0e6f469281e7ca94e6c19e23` are the original processes. Both services are
terminal success, withzero failures or repeated GPU/collection stages.

The preceding7.996s CPU qualification checks four actual CLI paths,32 header
cases,four retained frozen audits and337 inputs,while all401 active Qbert pins
and its original identity remain unchanged. Its8.123s controller cost stays in
the Qbert co-host timing caveat. Two historical runner references resolve to
exact Git bytes. The qualification is not repeated, and no build occurs.

[Fixed protocol](../experiments/2026-11-01-cdp-breakout-temperature.md).
Root `/x/Temp/kindle-breakout-temperature-20261101`. The assessment is complete;
do not rerun its arms,preflight or collector. No GPU allocation remains active
at closure;check live state before new work. The five-game goal remains unfinished.

| Receipt | SHA256 |
|---|---|
| Declaration | `c61000f0934e09853feeaada426d920d033e8549270c3ff67f5fd0770e78374b` |
| Comparison | `5bea06e0f94c27793fb4408293ce926931d2e07daa9a4caab7fc008a1d1f90a6` |
| Completion | `101f8b1ca0fca32aaad8f95a35107298510c916f23edaf7c68d6c2367322e7cd` |
