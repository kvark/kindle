# Longer Breakout sequences pass execution and early cost checks

A separately selected [fresh200k comparison](../experiments/2026-11-03-cdp-breakout-value1-context32-200k.md)
launches21:30:20 UTC November3/source06c873c2 with122 pins. Startup passes357
initial tensors against both controls,63 bounded/47 shared semantic events and
first update504. MainPID1071673 remains active. Read-only 100k/6,219 recovery
passes 21:42:21.105 UTC with all 357 finite tensors and 122 pins unchanged; the
independent CPU check takes .478s and preserves the original process identity.
Recovery contains weights and optimizer moments, without replay, games, live
beliefs or complete RNG state. Four final/actual-initial 24-episode arms remain
pending. Greedy remains primary against143.667, with sampled70.750 secondary. No full result isavailable; do not restart or
overlap the active learner. The bounded evidence andoriginal failures below
remain closed.

The current-critic B8/T32 check completes **8,192 actions /32,722 frames /
481 updates in51.909s**. Its conservative200k projection is **23m14.028s**,
below the fixed30-minute limit. Frozen1,024 actions /4,096 frames take.875s,
with zero updates, no natural completed episodes and all357 tensors unchanged.
This establishes execution and early cost; it does not measure gameplay
improvement. Retain sampled70.750/greedy143.667 and the original
mean864 /90%-two-wall gate; the five-game goal remains unfinished.

| Measurement | Current B8/T32 check |
| --- | ---: |
| Training time |51.908524s|
| Native construction, recorded separately |4.980663s|
| Learning stage |44.939904s|
| Training updates / first update action |481 /504|
| Actual overridden actions |3,922 /8,192 (47.876%)|
| Repeated labeled replay rows |118,950|
| Finite initial/final tensors |357 each|
| Initial tensors matching retained full18 |357|
| Shared pre-update semantic events |47|
| Frozen tensors unchanged |357|

Training takes8.85% longer than the retained B16/T16 bounded measurement of
47.689684s. The projection uses training seconds ×200000/8192 ×1.1; construction
is separate. These are single early measurements with different trajectories,
not a paired kernel benchmark or an estimate of final learning efficiency.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-value1-context32.md)
tests the earlier promising longer-sequence setting with the retained actor/
critic recipe. Earlier B8/T32 raised greedy104.667→130.417 but lowered the then-
primary sampled score; it used behavior4e-5/replay-value0.3. This check keeps
the current behavior8e-5/replay-value1. Their gameplay interaction remains open.
The negative context9 recomputation experiment changes a different mechanism.

Only batch/world-microbatch16→8 and sequence/world-backprop16→32 change from the
current retained learner. Keep256 sampled positions/update,12M64/Auto/N8/H15/
R16/context1, encoder/base4e-5, dynamics4e-4, policy1/critic1/1, disagreement1,
expert100 and sampled collection/posteriors. Keep the unchanged full18 teacher,
20% episode guidance/cap1920, all-state corrective labels and native pooledRGB/
one GPU64 resize. Native447fe50/SHA35ad0db and runnercf4ea238 remain unchanged.
No native build, production edit, transferred weights or reused experience occurs.

Longer temporal gradients, sequence diversity, sampling, warmup and target
rows240→248 change together. Cached recurrent state already carries history;
this does not establish added live memory or an isolated memory benefit.
The learner/evaluation seeds1009/4000001009 are reused development data.

All481 reports are finite and supervised. CNN,reward,actor andvalue parameters
change. World/actor expert losses agree within6e-7. Full training pixel/action/
label replay verifies all21 natural episodes and3,922 overrides. Both guarded
GPU stages pass once. The short frozen arm has no completed natural episodes,
so its lack of success is not a24-episode task assessment.

## Preserved CPU failures and repaired closure

The initial CPU preflight fails on a collector fixture expecting a file-hash
map from the older checkpoint audit. The repaired collector checks the metadata
hash and its three recorded tensor hashes. Both actual queued commands and24
negative config/cadence/label/prefix/checkpoint fixtures then pass in3.193s at
21:13:39.520 UTC November3,with zero agent/emulator constructions and zeroGPU
descriptors. All338 retained pins verify with one exact historical replay-source
resolution. Original scripts, stdout, stderr and failure receipt remain retained.

The original main launches21:15:01 UTC/source`9de712d1`,PID1071117/invocation
`7e4c8026d1df4074b518453522fad9a0`,with67 pinned inputs. Training and frozen restore
both succeed. The CPU audit then fails21:16:10 because its replay argv expects
`check/replay-manifest.json`,while the pinned manifest is at the package root.
The replay fails before constructing an emulator. The original main and terminal
collector remain **failed**; no original passing completion is fabricated.

A separate CPU repair preserves both original journals, verifies the exact
replay argv and supplies the identical pinned manifest at the expected path.
The unchanged failed audit then completes, including the first successful full
emulator replay. All67 original/7 repair pins,338 retained pins and39 original
output hashes stay unchanged. Both original service identities and failed states
remain unchanged. No training, frozen or successful replay stage repeats.

Repair closes **21:19:14.395 UTC** in9.605s as
`complete_with_cpu_audit_repair`. A separate1.335s readback completes
**21:20:20.243**,checking the saved closure without another audit, replay, model
forward or learner update. Evidence is under
`/x/Temp/kindle-breakout-value1-context32-20261103`; authoritative closure is
`cpu-repair/completion.json` and `cpu-repair/terminal-review.json`.
The original declaration, protocol and controller remain unchanged.

The cost prerequisite passes. No full learner, context sweep, confirmation,
extension or automatic successor is selected by this bounded check. No Kindle
allocation remains active at closure; check live state before further work.

| Artifact | SHA256 |
| --- | --- |
| Original declaration |`677c5d377c375b2834d646ab0d3023fa56e8069a429f02dac7ecc5e8f5511be3`|
| Repaired CPU preflight |`ff95a3c405f5c616b7a0fe44b061aa025d9fb7d7a26cdf75d9a1b0a58e0f2b3d`|
| Bounded review |`43eba44e14c616d541230ddcb10161b534b70c907899dcf487611c01be4e572c`|
| Training replay |`2285ae6903c6fe65a586bc8861ce902e3d7610b5a63a69984a04c90ea86b26c7`|
| CPU repair completion |`d8037ad9dc16ca2f1b8e3f9bd649e78a28c356beb86d7c3e5e299dc0cb55d2c4`|
| Repair readback |`93800b1281ce10bbf2daa2900ffdebf8dad57fc1f5a1862c438037a971b92f86`|
