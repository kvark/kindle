# Breakout replay-value1 improves frozen returns at the fixed budget

The [fresh200k comparison](../experiments/2026-11-02-cdp-breakout-replay-value1-200k.md)
scores **70.750 sampled /143.667 greedy**, versus59.000/117.292 at scale0.3.
Returns improve19.92%/22.49%, while training takes **21m27.910s**,1.27% less
than control. Initial-subtracted return per training minute improves
**21.96% sampled /24.50% greedy**, satisfying the declared development decision.
Retain scale1 as the stronger development point. Both modes still have
**zero of24 two-wall successes**; the original90% task gate remains failed.
This reused learner/evaluation seed does not establish independent confirmation
or broad reliability. No extension, coefficient sweep or confirmation is selected.

| Replay-value scale | Sampled final / actual initial | Greedy final / actual initial | Two-wall successes, both final modes | Training |
| --- | ---: | ---: | ---: | ---: |
|0.3 control|59.000 /1.417|117.292 /2.208|0/24|21m44.517s|
|1 candidate|70.750 /1.417|143.667 /2.208|0/24|21m27.910s|

Only `loss_scales.replay_value` changes,0.3→1. Keep12M64/Auto/N8/B16/T16/
fullBPTT/microbatch16/H15/R16/base-encoder4e-5/behavior8e-5/dynamics4e-4,
sampled collection/posteriors, actorunimix.01, expert100/unit action weights/
world gradients/all-state corrective labels,20% guided episodes/cap1920.
Native447fe50/SHA35ad0db and teacher remain unchanged. Runnercf4ea238 matches
the bounded check; its only change from control6c1b69fb is ten fresh-CLI lines
exposing the existing coefficient. Defaults and native implementation stay fixed.
The coefficient changes critic and representation gradients together; neither
an isolated critic mechanism nor a missing-loss defect is established.

## Full result and evidence

Training completes **200000 actions /799595 frames /12477 updates**. All338 pins,
the100k/6227 recovery,357 finite matching initial and invariant frozen tensors,
47 paired pre-update events,800 sampled/795 greedy actual-initial semantic
events, full pixel/action/label training replay, four frozen outcome replays and
videos, host guards, curves, exposure and comparison pass. All12477 reports are
finite. Actual initial values match both control and bounded check. Four frozen
arms total **90856 actions /363254 frames /80.001308s**, withzero updates and
zero truncations. Ordinary frozen logs lack live pixel hashes; no unavailable
live-image equality is claimed.

Actual training overrides are86117 (43.0585%), versus81918 (40.959%). Final50k
replay expert CE is.560514 versus.551145, and guided online last50 return is
67.78 versus66.00. These do not explain the frozen gain or measure unaided
competence. The comparison plot is visually checked: curves are guided online
returns, with frozen means and training time reported separately in its caption.
The small wall-time difference is measured, not an isolated runtime-speed claim;
trajectories and CPU co-host work differ.

MainPID887827/invocation`4b67c053d030409f8734e6bfbdab6707` launches03:14:00.615
UTC November2/source45cd1ab. Original main/postprocessing/collection succeed
**03:39:28.406 /03:39:34.988 /03:39:37.809**. Allfive original services succeed.
The progress observer has no separate resource-consumption timestamp; its
original identity/journal and successful terminal state are verified. No original
training, frozen, replay, audit or collector stage fails or repeats.

The separate terminal reader runs once and passes **03:40:38.280** in1.749s
(1.768s wrapper), verifying original identities, completion hashes, retained
recovery, per-episode score/gate arithmetic and comparison. Original in-memory
generator and stale declaration-filename failures remain preserved; separate CPU
repairs precede the sole GPU launch. All qualified sources and the existing
queue remain unchanged through the latter repair. Startup, recovery confirmation,
terminal-reader preparation, the4.038s Qbert phase diagnostic and10.229s/8.404s
crossed-assessment preparation co-host costs remain uncorrected.

Root `/x/Temp/kindle-breakout-replay-value1-20261102/full` is closed. Do not repeat
its learner, observers, collector or terminal reader. No GPU allocation remains
at closure; check live state before separately selected work. No automatic
successor or reuse of this life's experience is selected. The five-game goal
remains unfinished.

| Full artifact | SHA256 |
| --- | --- |
| Declaration |`e3b25b6595b331227ed5c4eb88191c4e068205c1d09f40a3f27a318f238d8a20`|
| Learning review |`2dec02fba407ca843fed9d365f1c42f214a4b5508ce67b9e8e648424de0b6b39`|
| Original completion |`6254f7296aa574e93d9442a9fcbfb2672391a39d0c1aa2ece59edb4d89396464`|
| Terminal summary |`b55b201673150fb39d2b92903e53a85c062d12dccea8af5b67cdfb07b9d8dd72`|
| Comparison |`148b6d928733183d4de6d90602ef3e1298b97649d42f88a33a58eccba74e599f`|

## Bounded execution and early cost check

The [bounded check](../experiments/2026-11-02-cdp-breakout-replay-value1.md)
completes **8,192 actions /32,727 frames /489 updates in47.689684s**. Its
conservative200k projection, including10% margin, is **1,280.728814s
(21m20.729s)**, below30min. The1,024-action /4,096-frame frozen restore takes
.871872s, makeszero updates and completesno natural episodes. This establishes
execution and early cost, not gameplay benefit or task competence.

Only replay-value loss scale0.3→1.0 changes from the retained behavior8e-5
control:59.000 sampled/117.292 greedy, bothzero of24 two-wall successes.
Keep centered CDP12M64/Auto/N8/B16/T16/H15/R16, base/encoder4e-5, behavior8e-5,
dynamics4e-4, sampled collection/posteriors, actorunimix.01, expert100/unit
weights/world gradients/all-state corrective labels and20% episode guidance/
cap1920. Native447fe50/SHA35ad0db, teacher, full18/sticky.25/repeat4/noops0,
native pooled RGB and one GPU resize remain unchanged.

The existing coefficient weights critic fitting and representation gradients
through frozen value logits. It retains real rewards/boundaries and imagined
bootstraps. Neither an isolated critic cause nor a missing-loss defect is
established. The preceding critic diagnostics concern neighboring behavior4e-5
controls, and Qbert's development gain did not confirm on its fresh seed.
Breakout benefit therefore remains a separate empirical question.

## Completed evidence

All **320 pins**,357 finite control-matching initial and invariant frozen
tensors,47 matching pre-update semantic events, correct cadence,489 finite
supervised reports, aligned world/behavior imitation values and changed
CNN/reward/actor/critic weights pass. Full training pixel/action/teacher-label
replay and both host guards pass. Training executes3,902 actual overrides;
guided returns are not unaided competence. There are16,334,353 parameters.

Candidate runnerSHAcf4ea238 adds only ten lines to the exact control6c1b69fb
runner, exposing the already-qualified fresh-only replay-value option. Defaults
stay unchanged. All23 relevant private-runner loss/rate CPU tests pass, followed
by2.186s command/config/schema/cadence preparation (2.218s wrapper): two actual
commands,17 config cases and all282 control pins. No native module or emulator
is constructed by preparation. No production or native source changes occur.

The original mainPID887239/invocation`491916f869dd4cbfa75b236f3a82f0dd` launches
**03:01:28.714 UTC November2/source70b7567** and succeeds **03:02:34.926**.
Original collectorPID887242/invocation`e5e1a7157da34264a98ccafd99a37ae7`
writes completion **03:02:40.372** and is terminal success. Its journal has no
separate resource-consumption timestamp; original identity, completion payload
and successful terminal state are verified.

The separate terminal reader runs once and passes **03:03:34.459** in1.289s
(1.308s wrapper), verifying all inputs, completion files, original journals,
the replay and projection arithmetic. No preparation, GPU, replay, audit or
collection stage fails or repeats. No cap or setting changes after selection.

Root `/x/Temp/kindle-breakout-replay-value1-20261102/native-check` is closed;
trial `breakout-12m-64-b16-r16-replayvalue1-expert100-check`. Do not repeat this
check, collector or reader. No active Kindle allocation remains at closure.
A full learner requires a separate selection; no automatic promotion, extension,
retry, sweep or check-experience reuse. The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`5106a120d82d3711c3d3a53b0bb401b5a4493bf757448ce82b77ee8304ec1228`|
| Review |`0eab7daf89deb10670dc2d01d621e35ab7f12565e69a1bf5e9ff5a07751f624d`|
| Original completion |`56cd5cbacddff169edc5a210fc4e55161b4b5012cd9ee409f4075efed81e027a`|
| Terminal summary |`cfc4b63918f5b0e2a0cbe81062c17eda54cc28084809887d160b5a642c8e6ba3`|
