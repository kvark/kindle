# Lower Breakout dynamics rate regresses both frozen scores with replay-value1

The [fresh 200k comparison](../experiments/2026-11-02-cdp-breakout-value1-dynamics1e4-200k.md)
scores **42.917 sampled /112.125 greedy**, below the control's 70.750/143.667
by **39.34%/21.95%**. Training takes **21m36.378s**, 0.66% more time. Initial-
subtracted return per training minute falls **40.54% sampled /22.81% greedy**.
Retain dynamics4e-4 with replay-value1/behavior8e-5. All final arms still have
**zero of24 two-wall successes**; the five-game goal remains unfinished.
No extension, confirmation, rate sweep or automatic successor is selected.

| Dynamics rate | Sampled final / actual initial | Greedy final / actual initial | Two-wall successes, both modes | Training |
| --- | ---: | ---: | ---: | ---: |
|4e-4 control|70.750 /1.417|143.667 /2.208|0/24|21m27.910s|
|1e-4 candidate|42.917 /1.417|112.125 /2.208|0/24|21m36.378s|

Only dynamics4e-4→1e-4 changes. Keep12M64/Auto/N8/B16/T16/full BPTT/
microbatch16/H15/R16, base/encoder4e-5, behavior8e-5, replay-value1 with world
gradients, sampled collection/posteriors, actor unimix.01 and expert100.
Corrective labels, unit action weights, expert world gradients,20% episode
guidance/cap1920, full18/sticky.25/repeat4/noops0 and native pooled RGB remain
fixed. Native447fe50/SHA35ad0db and teacher are unchanged. Private runner
ea2bf8c6 adds the qualified twelve fresh-CDP dynamics-rate CLI lines to control
cf4ea238. No build or prior experience reuse occurs.

The earlier lower-dynamics result at behavior4e-5/replay-value0.3 was mixed;
combining it with subsequent improvements does not preserve its greedy gain.
This reused seed1009/cohort4000001009 result does not establish an isolated
mechanism or broad reliability. The rate affects recurrent and predictive
world components together. Final50k replay expert CE rises .560514→.662474;
actual overrides rise43.0585%→48.841%. Guided online last50 return rises
67.78→71.18 despite lower frozen scores. Neither online returns nor imitation
fit measures unaided competence or identifies the cause of regression.

## Completed evidence

Training completes **200,000 actions /799,575 frames /12,477 updates**. All422
input pins,100k/6,227 recovery,357 finite matching initial/invariant frozen
tensors,47 matched pre-update events,800 sampled/795 greedy actual-initial
semantic events, finite aligned reports, full pixel/action/teacher-label replay,
four frozen replays/videos, host guards, curves, exposure and comparison pass.
Four frozen arms total **83,208 actions /332,670 frames /79.039s**, with zero
updates and truncations. Ordinary frozen logs lack live pixel hashes; no direct
live-image equality is claimed. The comparison plot is visually reviewed:
guided online curves and frozen scores are labeled separately.

Original main907551/invocation47b00c7eccad47d6a116a2792b0a6f8b launches
06:44:30.656 UTC November2/sourcec691416. Main/postprocess/collection succeed
**07:10:06.890 /07:10:34.218 /07:11:08.882**. All five original services succeed;
the progress observer lacks a separate resource-use timestamp but its original
identity/journal and terminal success pass. The recovery observer succeeds
06:55:37.842; its separate reader passes06:55:54.958 with four retained files.

The qualified terminal reader completes once **07:11:52.749**, in2.877s
(2.897s wrapper), verifying original identities, completion hashes, score/gate
arithmetic and comparison. The original bounded source-generator failure and
separate correction remain preserved; full preparation and all original full
stages pass without repeats. CPU co-host costs, including the2.462s coverage
qualification, remain recorded without subtracting them from training time.

## Higher-score experience remains scarce

The separately qualified saved-log reader completes once **07:12:03.388**, in
2.969s (2.992s wrapper). It reuses the completed scale1 control analysis and reads
only the new learner's training/frozen logs. All283,208 new actions/1,132,245
frames, completed episodes, partial tails and96 selected frozen scores reconcile.
All48 diagnostic/422 full pins pass. No GPU work, native/emulator import,
learning, environment replay, control recount or successful-stage repeat occurs.

| Training coverage | 4e-4 control | 1e-4 candidate |
| --- | ---: | ---: |
| Completed episodes / partial tails |263 /8|279 /8|
| Maximum accumulated score, including tails |408|419|
| Episodes reaching216 |15|20|
| Actions with pre-action score216–431 |5,947|5,552|
| Teacher overrides in that range |4,685|4,625|
| Native actions in that range |1,262 (0.631%)|927 (0.4635%)|
| Actions at432+ |0|0|
| Completed unguided maximum |102|79|
| Completed guided episodes reaching cap1920 |13|13|
| Native actions after that cap |1,286|969|

All20 candidate training episodes reaching216 are guided. Sampled frozen play
reaches a maximum118 and none of24 episodes reaches216; greedy reaches405,
with two of24 at216 and none at432. The fixed score ranges are fractions of
the task target, not verified wall boundaries or visual-state coverage. Counts
describe the evolving learner's experience and cannot establish that coverage
caused the frozen regression. No guidance change is selected by this analysis.

Root `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102/full` is closed.
Both final readers and the recovery reader are complete; do not repeat any
completed stage, collector or reader. No Kindle service or GPU allocation is
active at closure; check live state before separately selected work.

| Full artifact | SHA256 |
| --- | --- |
| Declaration |`c14ab341372b82c6c5b971cdc790826bdb2c6764f7346de9a63b59cb0321b859`|
| Learning review |`0080098beae3a4e329baf3bf82bd7396f3383aae1b7adb1e3642bc6eaa3dc6f7`|
| Original completion |`5f2cd92c8528b454e06f1369c2582eba4bcb33e3ea856718b62da1156278e0a9`|
| Terminal summary |`8ca4e94714e3b541ebff41d8ffa69278433ba2b61a4f172feb5985d4f00f4d84`|
| Comparison |`be51447716c4d07e3166a9ae853482898ad3f7918b85753d7d64edecbc771fa1`|
| Coverage review |`65b130da2b94cec29327119cc45cf0cd1fa25bd08ec729ffdb6c9ce371a49b7e`|

## Earlier bounded execution and early-cost check

The fresh bounded check completes **8,192 actions /32,727 frames /489 updates
in48.489s**. Frozen restore takes **1,024 actions /.861s**, with zero updates
and no natural episodes. The conservative200k projection is **21m42.198s**,
below30 minutes. This establishes execution and early cost, not gameplay benefit.

Only dynamics4e-4→1e-4 changes from the replay-value1/behavior8e-5 control.
Keep12M64/Auto/N8/B16/T16/H15/R16/expert100, sampled collection, native447fe50,
teacher and all other retained settings. Private runnerSHAea2bf8c6 adds twelve
qualified fresh-CDP dynamics-rate lines. No native build or prior experience
reuse occurs. The lower-rate result on the earlier recipe remains mixed;
combining these settings does not imply an additive gain or isolated mechanism.

All382 inputs,357 matching initial/invariant frozen tensors,47 matching
pre-update semantic events,489 finite aligned supervised reports, changed
CNN/reward/actor/value weights, full training pixel/action/teacher-label replay
and both host guards pass. Actual overrides3,676/8,192 are guidance exposure,
not unaided competence. All32 targeted CPU tests and the two actual-command,
17-configuration/schema/cadence preflight pass. The original source-generator
snippet mismatch and separate correction remain retained; no passed stage repeats.

Original main906991/invocationbd41699bfdfd4f9a857366754a6b59c0 launches
06:38:49.986 UTC November2/sourcef15e58e. Main succeeds **06:39:55.422** and
original collector906994/028bbb8dd58a4cbb8c909d841faf48b7 succeeds **06:40:01.115**.
The separate terminal reader passes once **06:40:38.369** in2.045s (2.062s wrapper),
verifying both identities/journals, all hashes and comparison arithmetic.
No GPU, replay, audit or collector stage fails or repeats. The host is idle
at closure; check live state before separately selected work.

[Fixed protocol](../experiments/2026-11-02-cdp-breakout-value1-dynamics1e4.md).
Root `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102/native-check`, trial
`breakout-12m-64-b16-r16-value1-dynamics1e4-expert100-check`, is complete.
Do not rerun the check, qualification, collector or reader. No full learner is
selected by this bounded result; its experience cannot carry into a later run.
The original90%-two-wall gate and unfinished five-game goal remain unchanged.

| Artifact | SHA256 |
|---|---|
| Declaration |`b2115334122065dd0e74bdde98eb80a2b6aa8070256009abac47d7da9d9cc1e2`|
| Review |`7727c8a8cc11df07402b3184ac0023b623dc5af5ab6cfb48d9739d9a1a5a867e`|
| Completion |`16baa2c6a6435cb2ee78744fa341a7562f42cc8f9039cfeae68a491ef86ea923`|
