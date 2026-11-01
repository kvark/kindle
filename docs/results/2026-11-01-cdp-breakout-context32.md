# Breakout B8/T32 passes execution and early cost

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

## Separately selected full comparison

After reviewing the completed check and idle state,a [fresh200k comparison](../experiments/2026-11-01-cdp-breakout-context32-200k.md)
is separately selected. Full CPU qualification passes15:57:41 UTC November1 on
the first attempt,without native/emulator construction. It does not reuse this
check's experience or repeat its stages. No full frozen result is available.
