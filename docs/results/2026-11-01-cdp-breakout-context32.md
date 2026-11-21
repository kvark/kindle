# Breakout B8/T32 improves greedy play but lowers the primary sampled result

The fresh full learner finishes **200,000 actions/799,549 frames/12,469 updates
in23m12.128s**, 8.11% longer than the retained B16/T16 control. Sampled mean
falls13.69%, while greedy mean rises24.60%. Every trained arm still has zero
of24 two-wall successes. Retain B16/T16 for the primary sampled objective and
B8/T32 as a stronger greedy development point; no overall win, gate pass or
seed reliability is established.

| Measurement | B16/T16 control | B8/T32 candidate |
| --- | ---: | ---: |
| Training seconds | 1287.701 | 1392.128 |
| Updates | 12,477 | 12,469 |
| Final sampled mean | 56.292 | 48.583 |
| Final greedy mean | 104.667 | 130.417 |
| Actual-initial sampled / greedy | 1.417 / 2.208 | 1.417 / 2.208 |
| Two-wall successes, either final mode | 0/24 | 0/24 |
| Initial-subtracted sampled return/minute | 2.557 | 2.033 |
| Initial-subtracted greedy return/minute | 4.774 | 5.526 |
| Final50k-action replay expert CE | .558097 | .592406 |

Initial-subtracted return/minute falls20.49% sampled and rises15.75% greedy.
Late replay CE rises6.15%; this is training fit, not held-out imitation accuracy
or a cause of the mode-dependent gameplay change. Learning-call wall time rises
1142.476→1229.179s; acting/feedback remains63.208→63.202s and environment stepping
63.030→62.977s. Checkpoint time also differs (.632→8.489s). These are wrapper
wall timers, not isolated kernels or utilization measurements.

## Full comparison and closure

Only batch/world-microbatch16→8 and sequence/world-backprop16→32 change in the
learner configuration. Both sample256 replay positions/update atR16; temporal
gradients, diversity, sampling, target rows240→248 and warmup change together.
Cached recurrent history already exists and refreshes after replay learning.
Keep12M64/Auto/N8/H15/R16/encoder4e-5/expert100 and the same native447fe50,
runner6c1b69fb and teacher. The reused1009/evaluation4000001009 cohort and prior
negative context trials preclude an isolated memory or reliability claim.

All292 declared inputs,357 matching initial tensors and357 invariant tensors
per frozen arm pass. The47 common pre-update transitions and800 sampled/795
greedy actual-initial semantic events match. All12,469 supervised reports are
finite; CNN/reward/actor/value change. The100k/6,219 recovery is audited and all
four files retained. Full training pixel/action/corrective-label replay, frozen
outcome/frame/reset replays/videos, host guards, curves, exposure and comparison
pass. The comparison plot is visually reviewed. Ordinary frozen logs lack live
pixel hashes, so frozen replay does not claim unavailable live-image equality.

Four frozen arms total83,696 actions/334,616 frames/76.505s, withzero updates or
truncations. Each score uses the firstthree natural episodes per eight lanes.
Guidance is off. Training uses101,804 overridden actions (50.902%); online guided
returns are not unaided competence.

Launch15:58:36 UTC November1/source4f7b4c4, originalPID812861/invocation
`258e9d9deb0b463a91d4206a198085e4`. Training ends16:21:55; all GPU stages finish
16:23:39. Main succeeds16:26:03, postprocessing16:26:09, progress16:26:22 and the
original collector16:26:47. Recovery succeeds16:10:44. All five original services
are terminal success, with no failed or repeated GPU stage, cap change or repair.
No Kindle service remains active at closure. No extension, context retry,
confirmation or automatic successor is selected; the five-game goal remains
unfinished. Do not repeat completed training, assessments, analyses or collection.

Retain co-host timings without correction: startup1.047s; recovery copy.310s and
independent saved-hash read.095s; Qbert phase selection1.539s/analysis1.719s
(1.763s wrapper); centroid-head selection3.463s/analysis2.640s (2.696s wrapper).
Light source/document reads and the existing unrelated CPU workload remain
timing caveats. Host guards are host-only; GPU telemetry is unmeasured.

Root`/x/Temp/kindle-breakout-context32-20261101/full`, trial
`breakout-12m-64-b8-t32-r16-encoder4e5-expert100-200k-seed1009`, service
`kindle-breakout-context32-200k-20261101`. `final-summary.json` derives the table
once from original completed audits; it does not repeat an experimental stage.
See the [fixed full protocol](../experiments/2026-11-01-cdp-breakout-context32-200k.md).

| Full receipt | SHA256 |
| --- | --- |
| Declaration | `6ae1164b4365f74e563b2a89f589783acd4c33505b87b58ef7753925412f176e` |
| Learning review | `12a55842830fbf905580626338ac030cf194411b39dce66fbd49804e25b47e5f` |
| Postprocessing | `5a74493b9ca97d18908190fc5def11b16ec36f088c030f74f7f402c6e2ffa5de` |
| Comparison | `6f3bd6588c395ab0a0cf5b67444db861d6345925c04ee7c2d6f531aa1e435cfa` |
| Original completion | `295cde7c69b8b729bbb36a757de3efabdc63038636fd5203bdf738efaa8c8b29` |

## Earlier bounded execution and cost check

The bounded longer-sequence check completes **8,192 actions/32,730 frames/
481 updates in53.799s**. Frozen1,024 actions/4,096 frames take.870s withzero
updates and no natural completed episodes. The conservative200k projection,
including10% margin,is **24m04.805s**,below the fixed30min gate. Execution and
cost pass;gameplay benefit remains unmeasured.

All261 declared inputs remain unchanged. All357 initial tensors match the
retained B16/T16 control,47 common pre-update events through action376 match,
and all357 frozen tensors stay invariant. Candidate learning starts at504,
with481 updates and residual.5 at8,192. All481 supervised reports are finite;
CNN,reward,actor and value tensors change. Full training pixel/action/corrective-
label replay and both host guards pass. Actual overrides3,932/8,192=47.998%;
guided online returns are not frozen competence.

The candidate uses B8/T32 instead of B16/T16,including world microbatch8 and
backprop32. Both sample256 replay positions/update atR16. Temporal gradients,
sequence diversity,sampling,target rows240→248 and warmup change together.
Cached recurrent state already carries history and is refreshed after replay
learning;this does not test adding live memory. Keep centeredCDP/12M64/Auto/N8/
H15/R16,encoder/base/actor4e-5,dynamics4e-4,expert100/world gradients/unit weights,
all-state corrective labels and20% episode guidance/cap1920. Native447fe50/
SHA35ad0db,runner6c1b69fb and teacher remain unchanged. Reused1009/cohort and
previous negative context comparisons limit any general conclusion.

The qualified CPU preparation is reused without repetition. Selection follows
completed Qbert closure and an idle-host check. Launch15:51:35 UTC November1/
sourcec2f7abf,originalPID812306/invocation`c51dbe709ad14d8289d6a921769c621c`.
Main succeeds15:52:52;the original collector succeeds15:53:01 in.669s wrapper.
No failed or repeated GPU stage,cap change,build or production edit occurs.
No GPU allocation remains active at closure. No full learner,context sweep or
automatic successor is selected. A fresh full comparison would require separate
selection,100k/6,219 retained recovery and all four final/actual-initial24-episode
assessments against sampled56.292/greedy104.667. No check experience may be reused.

The retained audit's phrase “on12M64/B16” names the baseline;actual configuration,
queue,schema and CPU fixtures require candidateB8/T32. Qualified evidence bytes
remain unchanged. Host guards are host-only;kernel utilization is unmeasured.
The five-game goal remains unfinished.

Root`/x/Temp/kindle-breakout-context32-20261101/native-check`,trial
`breakout-12m-64-b8-t32-r16-encoder4e5-expert100-check`,service
`kindle-breakout-context32-check-20261101`.
See the [fixed protocol and separate selection](../experiments/2026-11-01-cdp-breakout-context32.md).
Do not repeat completed qualification,training,frozen pass,replay or collection.

| Receipt | SHA256 |
|---|---|
| Declaration |`1ef747e1af459c7ba6b8f444dc5c64ca2e7e6704b180cfea5d610ea113b0d0e4`|
| Learning/frozen review |`9bd0ed162475b0ee4d46ec551cfc7ffd0c3942e7d9c812856124fafdb4853928`|
| Full guided replay |`0e29cda9b7e2df2cec1258274d356eb22152217ce6c22bd866a0a69753d10df4`|
| Original completion |`64e7ec1487c2d56a0591ae387ea02b977abbf46790f45c12d0dd2c43f31bcc85`|

## Full selection and startup history

After reviewing the completed check and idle state,a [fresh200k comparison](../experiments/2026-11-01-cdp-breakout-context32-200k.md)
is separately selected. Full CPU qualification passes15:57:41 UTC November1 on
the first attempt,without native/emulator construction. It does not reuse this
check's experience or repeat its stages. The completed outcome is above.

The full learner subsequently launches15:58:36 UTC/source4f7b4c4,originalPID812861/
invocation258e9d9deb0b463a91d4206a198085e4. Startup15:59:27 passes292 pins,357
matching initial tensors,47 common-prefix transitions and383 finite reports
through6,632 actions. All five original services are active at startup. See the
protocol for exact caps,assessments and recovery;startup1.047s CPU co-host caveat
remains.

The original full recovery observer subsequently succeeds16:10:44 UTC November1:
100k/6,219,357 finite tensors,all four files/206,067,157 bytes retained with matching
hashes and unchanged source (.310s). The original learner continues at that time;
its completed result is above. Weights/metadata omit replay,belief,games and RNG.
No copy,audit or GPU stage repeats;keep CPU co-host timing without correction.
