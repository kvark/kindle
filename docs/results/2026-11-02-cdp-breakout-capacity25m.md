# Larger Breakout capacity lowers frozen scores and costs more

Retain the 12M recipe. The [fixed fresh comparison](../experiments/2026-11-02-cdp-breakout-capacity25m-200k.md)
with the 25M preset completes **200,000 actions /799,572 frames /12,477 updates
in36m08.795s**, within its45-minute training cap. Sampled and greedy frozen
returns fall **29.39% and27.41%**, while training takes **68.40% longer**.
Both models remain **0/24 two-wall successes** in each assessment. The larger
preset does not earn its cost at this recipe and budget.

## Full comparison

| Fixed endpoint | Retained12M | Candidate25M |
| --- | ---: | ---: |
| Trainable parameters |16,334,353|36,608,401|
| Actions / updates |200,000 /12,477|200,000 /12,477|
| Training time |21m27.910s|36m08.795s|
| Sampled final / own initial |70.750 /1.417|49.958 /1.458|
| Greedy final / own initial |143.667 /2.208|104.292 /1.583|
| Sampled initial-subtracted return/min |3.230|1.342|
| Greedy initial-subtracted return/min |6.590|2.841|
| Sampled / greedy two-wall successes |0/24 /0/24|0/24 /0/24|
| Final50k-action replay expert CE |.560514|.499035|
| Actual overridden training actions |86,117 (43.0585%)|83,624 (41.812%)|

Initial-subtracted return per training minute falls58.46% sampled and56.88%
greedy. Each score uses the first three natural episodes per eight lanes,
without guidance,updates or truncations. All357 tensors remain invariant in
each frozen arm. The candidate's four captures total90,216 actions/360,667
frames/131.176s, withzero updates. Its own initial values differ from12M;
723/836 sampled/greedy initial semantic events are retained, versus800/795
for12M, without an equality claim across architectures.

Only the preset changes atRGB64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16,
encoder4e-5,behavior8e-5,dynamics4e-4,replay-value1 with imagined bootstrap,
expert100 and sampled collection. Native447fe50,runnercf4ea238,teacher,
guidance schedule and Atari settings remain identical. The preset expands CNN,
recurrent state,categorical classes and behavior heads together. This is one
reused development seed1009/cohort, with changed initial values and subsequent
experience; it does not isolate a component or establish general capacity limits.

Better replay imitation fit does not transfer to better frozen gameplay here.
The guided online last50 mean is79.7 versus67.78 for12M; neither is an unaided
competence measurement. No extension,confirmation,capacity sweep,learning-rate
retuning or automatic successor is selected by this result. The five-game goal
remains unfinished.

## Saved score exposure

The separately qualified CPU reader reconciles all290,216 candidate actions and
1,160,239 frames across training and four frozen logs. It reuses the completed
control analysis; no emulator trajectory or prior analysis repeats.

| Training pre-action score |12M actions / overrides / native|25M actions / overrides / native|
| --- | ---: | ---: |
|0–215|194,053 /81,432 /112,621|195,731 /79,662 /116,069|
|216–431|5,947 /4,685 /1,262|4,269 /3,962 /307|
|432 and above|0 /0 /0|0 /0 /0|

Native actions in216–431 fall from.631% to.1535% of training. None of the24
sampled episodes reaches216; three greedy episodes do, with maxima106/365.
No selected frozen episode reaches432. These are score fractions, not verified
wall boundaries or comparable visual states. Different collection trajectories
limit causal interpretation; the counts do not establish why performance fell.

## Completion and preserved CPU failure

All420 original input pins,exact configuration/cadence,12,477 finite reports,
357 bounded-matching initial tensors,47 pre-update transitions,100k/6,227
retained recovery and frozen invariance pass. Full guided replay verifies input
pixels,actions,labels and outcomes. Allfour recorded-action frozen replays,
videos,host guards,curves,exposure,comparison and original collection pass.

Original mainPID933689/invocation`d3457431b99a41aebd2ec054df3f4d7d`, launched
10:47:21.939 UTC November2/source`0e69b83`, succeeds **11:28:52.929**.
Postprocessing/collection succeed **11:29:26.592/11:29:35.005**. Allfive original
services are terminal success. The prepared terminal reader runs once at
**11:30:31.187**, taking3.019s (3.039s wrapper), source`28186c9`.

The first supplementary coverage preflight fails10:55:41.798 because its new
fixture selector matches both training and frozen counter assertions. Preserve
that reader and failed1.397s receipt. A separate selector correction passes six
config/three counter cases,55 diagnostic pins and all420 active pins before
outcomes; parser,metrics and thresholds remain unchanged. Its final analysis
runs once at **11:30:39.622**, taking3.214s (3.239s wrapper), with all55/420 pins
passing. The original failure remains preserved; no original full-pipeline
failure,learner restart,GPU repeat,successful replay repeat or cap change occurs.
All CPU co-host costs remain recorded without correcting training time.

Root `/x/Temp/kindle-breakout-capacity25m-20261102/full` is closed. Recovery,
terminal and repaired coverage readers are complete; do not repeat them. No
Kindle allocation is active at closure; check live state before further work.
The original90%-two-wall gate remains unmet.

| Full artifact | SHA256 |
| --- | --- |
| Declaration |`1985ac2c278ab9262d04715ef6cdffa6a3bc9c0b54617d890e15f5b46ecb8697`|
| Learning review |`86de613bbde9fb3990310c23d11d94c0f23a72a5364ba2ffc9572dc55fed069c`|
| Completion |`1988e62174b33ceeb22b9c16e239ab7873e40a3bb12c1fee4f701c90e8b255bb`|
| Terminal summary |`b80dc80927a90617441a6044cf683126429e03d71f7ee2adbd66f55c4154ec0a`|
| Comparison |`210af63d459d13f5deb203f4cd0ece8edfa37b2bba521e1a35466cef69621647`|
| Repaired coverage |`7f2efd01f494aa775b2bc3a5ae6d915694c03676c9358fd7b7c94a4336a8e323`|

## Earlier bounded learning and restore check

The existing 25M preset completes the declared **8,192 actions / 32,722 frames /
489 updates in84.864s**. Its **1,024-action /4,096-frame frozen restore takes
1.343s**, with zero updates and no completed natural episodes. Execution and
the early cost screen pass. Gameplay benefit remains unmeasured; no full
learner is selected at this check's closure.

The [fixed protocol](../experiments/2026-11-02-cdp-breakout-capacity25m.md)
changes only the model preset from the retained replay-value1 recipe. It expands
the CNN, recurrent state, categorical classes and behavior heads together.
Trainable parameters grow **16,334,353→36,608,401**. RGB64, Auto precision,
N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100,
sampled collection, guidance and the full18 Atari protocol remain fixed.
Native447fe50, runnercf4ea238 and the teacher are unchanged; no build occurs.

| Bounded learning cost | Retained 12M | Candidate 25M |
| --- | ---: | ---: |
| Actions / updates | 8,192 /489 | 8,192 /489 |
| Training seconds | 47.690 | 84.864 |
| Conservative 200k projection, including 10% margin | 21m20.729s | 37m59.071s |

The candidate costs **77.95% more** on this short check. Its projection clears
the separately declared45-minute ceiling for a possible larger-capacity life.
The projection is not a full-run measurement or a speed improvement. A future
comparison must report absolute frozen scores and initial-subtracted return per
training minute against the retained12M scores70.750 sampled /143.667 greedy.
Useful quality per GPU-hour, and the original90%-two-wall gate, remain the aim.

The CPU schema derivation exactly reproduces the actual12M checkpoint before
deriving25M. All **357 actual candidate tensors** match, including342 changed
shapes; initial/final values are finite and all357 frozen tensors are invariant.
The different architectures have different initial values and trajectories:
no equality with12M is claimed. A separately selected full25M life must match
this bounded initial checkpoint and keep its own initial evaluations.

First update376, residual credit.5, exact configuration and all489 finite reports
pass. All489 reports include supervision, covering117,037 labeled replay rows
with equal weight sums; maximum raw world/behavior imitation-loss difference is
9e-7. CNN/reward/actor/value parameters change. Full guided replay verifies
every recorded input pixel, teacher label, action and outcome;3684 actions are
overridden,4508 have passive labels and3979 differ from their teacher label.
These guided training observations do not measure unaided competence.

Both GPU host guards, all **380 pins** and both original services pass. Main
PID933088/invocation`0bf2c9a4f0564d3695115d95b027b2aa` starts10:37:27 UTC and
finishes **10:39:11.653 UTC November2**, source`a8144ec`; the collector's review
completes10:39:18.624. Final read-only confirmation passes10:40:52.368, with both
services terminal and no active Kindle allocation. There are no preparation,
GPU or audit failures, repeated stages, cap changes or reused learner experience.
No automatic promotion, capacity sweep, retry, extension or full learner follows
from this check alone. The five-game goal remains unfinished.

Root `/x/Temp/kindle-breakout-capacity25m-20261102`; trial
`native-check/breakout-25m-64-b16-r16-replayvalue1-expert100-check`. The check and
collector are complete; do not repeat them.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `16622f5edd1c5de2180a707ca03b7791b56f16d4bbb8981306ffa5dc924136de` |
| Review | `0bcca24c778376b7a1904d546beecaa154acfe997f0681d97191ea3c2941ee69` |
| Guided replay | `946b62433928bd19b221e4d6adc017cf877d4b0b98800c245c0a6d9c81f3cf09` |
| Completion | `8ee60866c7400d9805428c0caf959d1c82e28957454a62fea369eba4d02ee24d` |
| Terminal summary | `df437e69f40f17d75881788f1320878688d170826f969c6fc86230dcbd4f06bd` |
