# Longer Breakout sequences miss the primary improvement test

The fresh B8/T32 comparison finishes **200,000 actions /799,554 frames /12,469
updates in 23m06.548s**, 7.66% more training time than retained B16/T16. Primary
greedy score falls **143.667→116.667** and own-initial-subtracted gain per minute
falls **24.84%**. Sampled score rises **70.750→94.583**, with gain per minute up
24.82%. Both final modes have **0/24 two-wall successes**. **Retain B16/T16 as
the primary recipe.** The original mean 864/90%-two-wall gate remains unfinished.

| Measurement | Retained B16/T16 | Candidate B8/T32 |
| --- | ---: | ---: |
| Training time |21m27.910s|23m06.548s|
| Updates |12,477|12,469|
| Primary greedy mean |143.667|116.667|
| Secondary sampled mean |70.750|94.583|
| Actual-initial greedy / sampled |2.208 /1.417|2.208 /1.417|
| Greedy gain over own initial per training minute |6.590|4.953|
| Sampled gain over own initial per training minute |3.230|4.032|
| Two-wall successes, either final mode |0/24|0/24|

The fixed paired lane-bootstrap difference in greedy gain/min is −1.637,
95% interval [−3.928, .686]. The sampled difference is +.802 [−.517, 2.496].
Both intervals span zero. The declared primary improvement rule fails; these
point estimates do not establish reliable harm or a reliable sampled advantage.
Do not switch the primary mode after seeing the outcome. This is one reused
development learner/evaluation cohort, not training-seed confirmation.

## Full 200k comparison

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-value1-context32-200k.md)
changes batch/microbatch 16→8 and sequence/backprop 16→32, preserving 256 sampled
positions per update. Keep 12M64/Auto/N8/H15/R16/context1, encoder/base 4e-5,
dynamics 4e-4, behavior 8e-5, policy 1/critic 1/replay-value 1, disagreement 1,
expert 100, unit weights, world imitation gradients and all-state corrective
labels. Sampled collection/posteriors, actor-unimix .01, replay 100k, the full 18
pixel teacher, 20% guided episodes/cap1920, sticky.25/repeat4/noops0 and native
pooledRGB/one GPU64 resize remain fixed. Native 447fe50/SHA35ad0db and runner
cf4ea238 stay byte-identical. No new build or transferred experience occurs.

Temporal gradients, diversity, sampling, warmup and target rows 240→248 change
together. The actor already carries recurrent state between actions. This is
not an isolated memory effect or proof that longer sequences cannot help.
The earlier B8/T32 experiment used a different actor/critic recipe; its higher
greedy score did not predict this combination's result.

All 357 actual-initial tensors match both the bounded candidate and retained
control. The 63 bounded and 47 shared pre-update semantic events match, with
first update 504 and final residual credit .5. Read-only 100k/6,219 recovery passes
independent verification at 21:42:21.105 UTC in .478s: 357 finite tensors, all 122
pins and the original process identity are unchanged. Recovery contains weights
and optimizer moments, without replay, live games, beliefs or complete RNG state.

Each of four final/actual-initial sampled/greedy arms selects the first three
natural episodes per eight lanes, 24 total, with zero updates/truncations and 357
invariant tensors. Actual-initial traces match 799 sampled/794 greedy semantic
events against the retained control. The four arms total 88,168 actions /352,493
frames /77.573s. Excess episodes and partial tails remain recorded. Full frozen
recorded-action/outcome/frame/boundary replays and videos pass; ordinary frozen
logs do not supply live pixel hashes for a direct live-image comparison.

## Fit, exposure and cost

All 12,469 supervised reports are finite. CNN, reward, actor and value parameters
change. There are 3,087,050 repeated labeled replay rows. Last 50k-action expert
cross-entropy is .610334 versus retained .560514; the changed visited-state
population prevents a matched-generalization or causal interpretation.
Actual overrides are 93,896/200,000 (46.948%). Guided online returns are not
unaided competence. The curve's online last 50 mean is 82.02, distinct from the
frozen scores; the rendered curve has been visually checked.

Maximum training score including tails is 406. There are 1,598 native actions at
pre-action scores 216–431 and none at 432+. These are score-band exposure counts,
not verified wall boundaries or evidence that coverage alone causes the gap.

Native construction takes 4.654s, recorded separately. Wrapper timers attribute
1,228.750s to learning, 63.193s to acting and 62.888s to environment stepping;
these are not kernel-exclusive GPU timings. CPU startup/recovery observers and
light document/source reads co-host with training. Retain the measured wall
time without subtraction. Full training replay takes 108.445s and verifies all
265 natural episodes, pixels, actions, labels and overridden-action accounting.

## Verification and closure

Before launch, the changed CPU helpers and all five actual queued commands pass
qualification, including the actual replay argv and manifest paths, recovery
schedule and terminal identity handling. The command fixtures construct no
native agents or emulators. The private exact historical replay source resolves
one retained pin; original declarations and pinned sources stay unchanged.

All 122 current/338 retained pins, initial/recovery/final checkpoint checks,
finite reports, replays/videos, five host guards, curve, exposure, score coverage
and fixed comparison pass. Main PID1071673/invocation
`2cecc3ab00f9498795197cf1de25639a` launches 21:30:20.400 UTC November 3/source
`06c873c24fcca0db6ce551f0bb48d2a99b9dd8d3`. Main completes **21:58:08.786 UTC**;
the independent terminal collector completes **21:58:53.098** in 2.466s. Main,
recovery and collector are all terminal success. No full-run stage fails or
repeats, and no cap changes. Original bounded CPU failures below remain preserved.

Root `/x/Temp/kindle-breakout-value1-context32-20261103/full` is closed. Do not
restart its learner or repeat its observers. No extension, context sweep,
confirmation or automatic successor is selected. No Kindle service or render-node
holder remains active at closure; check live state before new work. The original
five-game goal remains unfinished.

| Full-comparison artifact | SHA256 |
| --- | --- |
| Declaration |`6028fe0cb571a8e8ad12d47fb35f7063a2e4102be928ab1b1e63031892d7967f`|
| Learning review |`669ecd49b79164c157c0c729f075ba756ddd44664694ae8989db54c887f9fcde`|
| Training replay |`c320a7ced3dbeb87985f230ce9a79933da0e22ae5db098881d66caf2bb2ede46`|
| Comparison |`221b4b44f8662ed405937c8187d5bbed0f9a87d10e7db37ba4da4ce1bc1b7d57`|
| Completion |`41aeea44bef4e538d19586433212ecc9d1bbd5af370b7920495d1fb34a72c6d1`|
| Terminal summary |`7de1d6febbfe54947012811a1b9a4e41ba695847b51647c7add595fd547efe44`|

## Earlier bounded execution and cost check

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
