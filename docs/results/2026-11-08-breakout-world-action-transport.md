# Breakout: action-conditioned pixel transport

The bounded learner and final frozen assessment are complete.
The composition fails its fixed retention rule. Keep H0 as the broad reference.
Actor/critic and CDP comparison remain parked. No extension, sweep, collector or
successor is selected. [Protocol](../experiments/2026-11-08-breakout-world-action-transport.md).
Root `/mnt/data/kindle-breakout-world-action-transport-20261108`.

## Learning change

Combine existing direct action planes/motion width 16 with learned separable pixel
transport, shared RGB fallback and two-image history. Adds 595 weights to local
v18: 82 logical/8,893,798 scalars/246 tensors. Explicit v19 binds the two flags;
legacy defaults/headers remain. No new backend primitive, pretrained encoder,
object/RAM training labels, generated-image belief correction or actor/critic.
Reward and continuation still read RSSM features only.

Fresh8,192 B8/T32/H16/full BPTT32,RGB64,Auto,seed41001,all rates4e-4,mixing.01,
RGB100,KL1/.1/free-nat1,event scales1. Reuse the original16 fitting/eight
development streams and exact4/2/1/1 plan:975,941 valid suffix targets,9,570
terminal/16,008 positive suffix targets. No new environment actions or frames,
collection or recount. This tests a composition;decoder gradients and later
optimization change. It does not isolate action planes,capacity or transport.

## Frozen visual forecasts

Same 256 fitting/128 development origins, actual initial and fixed final;first
fixed model draw below. Counts are visible balls within one resized RGB64 pixel.
Missing/ambiguous balls fail. Reused related windows are development evidence.

|Model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|H0 reference|108/112|94/109|33/109|14/107|3/114|0/95|
|Local action|98/112|101/109|63/109|43/107|29/114|11/95|
|Transport|94/112|102/109|50/109|31/107|16/114|1/95|
|Action + transport|103/112|100/109|65/109|43/107|22/114|8/95|

Fitting, other draws, paddle, reward, ending and brick metrics remain in
`learning-review.json` and raw exports. A lower image loss cannot substitute for
physical accuracy. Periodic observations did not alter the budget or final choice.

## Fixed-action assessment

New final only: 84 groups, histories 16 primary / 32 secondary, eight model draws,
1/4/8/16/32 endpoints and the unchanged cached sticky-action oracle. Every old
control is reused without a new query. Primary balanced survival-weighted16/32
ball energy 0.387593263465; H0 .169502219090,
local action .240565683475,transport .200277257110;required <=.127126664317.
This is 128.666% worse than H0.

Ending16/32 Brier 0.093754356874/
0.249306093393;RIGHT16 false ending risk
0.488926931827 against oracle zero. Individual
nonending alarms above 5%: 197/304,max
0.991243012315. Category-mean gates and individual
alarms remain distinct. Freeflight8 detection NOOP/LEFT/RIGHT/REVERSE:
64/64/64/64
of64 each; presence alone does not establish position.

|Original intermediate requirement|Result|
|---|---|
|observed ball|Pass|
|one step ball|Pass|
|eight step ball|Pass|
|serve noop|Pass|
|ending brier|Fail|
|ending16 brier|Fail|
|ending16 right|Fail|
|nonending false alarms|Fail|
|mean ball energy reduction|Fail|
|freeflight each plan presence at8|Pass|

`supplementary-metrics.json` retains every per-action/horizon regression versus
seven cached controls and every individual alarm. Against H0, regressed cells:


|Metric|Regressed / scored cells versus H0|
|---|---:|
|Ball energy|114/210|
|Paddle energy|172/210|
|Reward Brier|204/210|
|Ending Brier|180/210|
|Alive-only brick Brier|24/209|

Supported point pairs omit survival and are not independent episodes. All oracle
outcomes through 16 are alive;better 16-step Brier there is false-alarm suppression,
not evidence of terminal recognition or natural-frequency calibration.

## Correctness, cost and limits

Native/source/runner `5fe4545caeac1b278653e109329b234667c35108`,native SHA256
`15bd4a8db077b1d8aaddfcb97ea0898ca51829b26f5c758a5f7d69ea116688f9`.
146 affected Rust CPU/392 packaged Python checks and strict core/binding Clippy
pass. Preserve the first CPU missing-v19-scope failure and first package helper-
path collection failure;their separate repairs do not relabel original failures.

First native qualification passes 1,274 pins in118.533593s: combined
F64/native values/derivatives/action/lane controls,240 common initial tensors,
16 cached nonimage arrays,10 non-RGB reports,246 own finite/frozen tensors,
13 finite reports/mask/total,projection changes and18 initial/trained restores.
All 8,192 learner reports are finite, the first matches qualification exactly;
all 82 weights change. Exact plan/counts, 246 own initial/frozen tensors, 384 dense
actual-initial/final origins,58 finite exports, 18 final restore arrays and exact
v19 configuration at all four milestones pass. No completion or superiority
claim follows from these implementation checks.

Native learning calls 1794.491362s;training/evaluation loop
1973.503401s;final frozen assessment
236.292693s;CPU branch comparison
182.078238s. Construction,initial dense exports,build/package,
qualification,guard/poll spans,failures and publication are additional.
Other host GPU work was present;these are not exclusive GPU benchmarks.

Closure seals 3,700 pins, four successful guards/36 files and three
terminal allocation services. Native Meganeura+Blade/Auto on RX7900XT/RADV,
no driver recovery/reset/NVML. Disposable 837,729,332-byte cache was released
following qualification; weights, runtime, wheel, source, logs and failures remain.
Comparison/learning figures and three fixed first-draw lossless causal APNGs
use cached arrays only;all decoded clip frames match. No best-draw selection.

Precise physics,event timing/calibration,persistent removed-brick memory,
late/tunnel/second-wall states,untouched families and fresh confirmation remain
unproven. Current oracle has no above-top-brick endpoint;its detector is restricted.
Passing intermediate gates could not by itself complete the world-model goal.
