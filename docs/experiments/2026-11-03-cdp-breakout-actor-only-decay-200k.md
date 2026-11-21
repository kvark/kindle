# Fresh200k Breakout comparison with late actor-only imitation decay

Select one fresh comparison after the [numerical and bounded qualification](../results/2026-11-03-cdp-actor-only-imitation-decay.md)
passes8,192 actions/489 updates in47.329s,599 pins,357 matching initial/invariant
frozen tensors,legacy action reproduction and full guided replay. The conservative
200k estimate21m11s passes30 minutes. Bounded experience is not reused.

The intended setting change from the retained replay-value1 control is actor
imitation100 held through update6227 (100k actions), linearly reaching zero at9352
(150k), then staying zero through12477. Include the endpoint:3,126 zero-weight
reports. **World imitation stays100 throughout.** Retain teacher guidance,
corrective labels and optimizer history; zero current actor imitation does not
remove earlier momentum or constitute unaided collection. Current actor gradient
attribution motivates this test but does not establish a causal gameplay defect.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/lambda.95/discount333,
base-encoder4e-5/behavior8e-5/dynamics4e-4, policy1/imagined-value1/replay-value1,
expert unit weights/world gradients/all-state corrective labels, actor/critic
representation gradients off, sampled collection/posteriors, actor-unimix.01/
entropy.0003, action-effects1, replay100k/warmup1000,20% episode guidance/cap1920,
seed1009/eval4000001009, full18/sticky.25/repeat4/noops0/native pooled RGB/one
GPU64 resize/raw rewards. No crop,centroid or open-loop loss is enabled.

Use qualified nativec2d627f3/SHA0015dff7 and the matching runtime/runner under
`/x/Temp/kindle-actor-only-decay-20261103`. Native and runner differ from retained
447fe50; record the explicit `replay_value_bootstrap=imagined` default addition
when comparing configs. Teacher and other learner settings stay fixed. Matched
initial values and legacy traces do not establish isolated causality or broad
runtime equivalence. Seed/cohort reuse is development, not independent confirmation.

Fresh training reuses no weights,replay,belief,RNG,games or experience. Stop at
200,000 actions/12,477 updates or1800s. Preserve actual initial and100k/6227
recovery with all four retained files and357 finite tensors. First update376,
residual.5,finite reports,exact actor schedule and constant world weight must pass.
Reconstruct behavior totals with the scheduled actor term and require raw
world/actor expert CE alignment. Record actual overrides separately from competence.

Fixed frozen arms are final/actual-initial, each sampled and greedy, first three
natural episodes per8 lanes=24 per arm. Sampled remains primary. Guidance/labels
off; zero updates/truncations and357 invariant tensors. Keep300s/600k actions
per arm. No outcome-based checkpoint/mode selection,retry or episode-cap increase.
Compare against retained sampled70.750/greedy143.667 and own initials1.417/2.208,
all0/24 two-wall. The90% two-wall gate remains unchanged. Report both modes and
own-initial-subtracted frozen learning per training minute.

Require357 initial values matching control and bounded check,47 pre-update events,
both initial frozen semantic traces,changed CNN/reward/actor/value,full training
pixel/action/label replay,frozen action/outcome/frame/boundary replays and videos,
guards,curves,exposure and comparison. Ordinary frozen logs have no live pixel
hashes; do not claim unavailable image equality. Keep CPU co-host timing without
wall-time corrections.

Before launch,qualify all five actual CLI paths,audit/comparison assertions and
negative cases,schedule accounting,header readiness,observer wiring,retention and
checkpoint schema on CPU. Pin source,runtime,scripts,inputs and protocol from a
clean commit. Retain one original main/progress/recovery/postprocessing/collection
service. Main cap4500s,audit1200s and fixed observer deadlines stay unchanged.
Preserve failures and repair CPU-only processing separately; never repeat a
successful GPU stage. Independently collect startup,recovery and terminal evidence
once. Any additional saved-data reader must be declared before reading outcomes.

No extension,coefficient sweep,confirmation learner or automatic successor is
selected. Root `/x/Temp/kindle-actor-only-decay-20261103/full`; service
`kindle-breakout-actor-only-decay-200k-20261103`. This is a deliberate comparison
within the unfinished five-game goal; execution success alone cannot finish it.

## CPU preparation

All five actual CLI paths,48 audit/comparison config cases,14 prelaunch command
cases,19 header checks,four observer-parser cases,14 recovery-body cases,seven
schedule cases,six retention cases and two collector corruption cases pass.
All19 controllers compile and44 embedded paths resolve;599 bounded and338
control pins remain unchanged. CPU preflight completes11:00:35.722 UTC
November3 in2.635s wrapper/2.522s body,with no native/environment construction.

The original.096s preflight import failure is retained: the F32 converter is
local to the general audit function,so the repaired controllers import the
existing independent decay-schedule converter. Before that first test,a source
review corrects one endpoint-neighbor fixture to the retained native report's
F32 value. Production code,runtime and schedule arithmetic remain unchanged;
no GPU stage occurs during preparation. Exact prior sources and receipts remain
under the private package.

## Launch and startup

The learner launches **11:02:01.359 UTC November3**, source
`18d493e6`, with649 pins. Original mainPID1015785/invocation
`b5f028d5fc304d1ea304ddd47e544b6d`; all four original observers start once.
Startup passes **11:02:37.174**:357 initial tensors match bounded check/control,
47 pre-update transitions match,first updates376/392/408/424/440/456,and320
finite reports through5,480 actions. Actor/world weights remain100. The1.575s
CPU wrapper co-hosts with training; keep that timing without correction.

DeclarationSHA`8118fc778be64e576de7ae7d6ad91c7aa4df3195adad855940d88070d3642396`.
At startup, no frozen gameplay result was available. The completed outcome is
recorded below. Do not restart the original learner or repeat completed stages.

The independent terminal reader qualifies five score checks at11:03:43.994
in2.503s CPU/2.524s wrapper,verifying649 pins and the original live identity.
Recovery and terminal readers are fixed before frozen outcomes. A separate
first100k-action saved-prefix comparison is selected11:05:36.936,seven fixtures/
six diagnostic pins/all649 learner pins,in1.315s CPU/1.334s wrapper. After the
original recovery succeeds,the fixed reader compares transitions,learner reports,
episodes and resets through100k;exclude only timestamps,episode wall time,
learner timing,and the candidate's two scale fields,each required100. Retain
differences as evidence; do not use them to trigger a rerun or alter this learner.
This does not compare uncaptured hidden state,gradients or intermediate weights.
All CPU co-host timings remain uncorrected. Private selection and source remain
under`/x/Temp/kindle-actor-only-decay-20261103/prefade-100k` and`review_prefade.py`.

## 100k recovery and pre-fade equivalence

The original recovery observer succeeds **11:13:07.765 UTC November3**, retaining
100,000 actions/6,227 updates,357 finite tensors and four files/206,067,315 bytes.
Its audit checks6,399 finite reports through102,760 actions. Independent confirmation
passes once **11:13:49.130** in1.357s CPU/1.375s wrapper; all649 pins and original
mainPID1015785/invocationb5f028d5fc304d1ea304ddd47e544b6d remain unchanged.
Recovery reviewSHAabc36a7b/retentionSHA0f76b715/confirmationSHA8315793a.

The declared first100k-action trace comparison completes **11:14:31.209**. All
12,500 transitions,6,227 learner reports,170 episode records and167 resets match
exactly after the declared timing exclusions and removal of candidate actor/world
scale fields,each required100. Actions,rewards,pixel hashes,labels and overrides
are included. There are zero differing or unpaired records. CPU wrapper4.442s;
reviewSHA`550f0fb18982c0074638c46901c941caff21d4ae39a06bc682e985ff5aeaab3e`.
The snapshot comparison is complete; do not rerun it.

A complementary retained-tensor comparison is selected **11:10:48.661**, before
the candidate milestone exists:five fixtures,ten diagnostic pins and649 learner
pins pass in3.519s wrapper. It compares every tensor in both100k retained
checkpoints,with exact schema/config/counter and file-hash checks. Completion
**11:14:30.090**,3.255s wrapper,finds **all357 tensors /51,502,659 elements exactly
equal**, including parameters,optimizer moments and slow values. World280,
behavior66 and slow-value11 havezero differing tensors/elements andzero maximum
absolute/L2 differences. Serialized file equality is not the criterion. ReviewSHA
`c11f24696b3fc9317f0bf41d439d5c1d7e9f6e8993577d8710933e6324ca904e`.
Evidence lives in`prefade-checkpoint-100k` with its fixed reader/selection;do not
repeat this completed analysis.

Both comparisons use saved evidence only,with zero updates or GPU/emulator work;
their analysis wrappers overlap,so keep each measured co-host duration without
summing them as an exclusive interval or correcting training time. All inputs
and the original learner identity stay unchanged. Checkpoints omit replay,belief,
games and RNG; these results do not prove full-state equivalence or broad runtime
equivalence. They strengthen the comparison before the actor schedule diverges.

## Completed outcome

The original comparison completes200,000 actions/799,600 frames/12,477 updates
in21m42.074s (+1.10% versus control). Sampled51.833/greedy104.750 fall26.74%/
27.09% below70.750/143.667; own initials1.417/2.208 match. All0/24 two-wall.
All3,126 zero-actor reports retain world100. Keep constant actor/world imitation.
See the [completed result](../results/2026-11-03-cdp-breakout-actor-only-decay.md)
for learning per minute, exact pre-fade evidence, validation and limits.

All649 pins,recovery,initial/frozen tensors,paired initial traces,replays/videos,
guards,curves,exposure and comparison pass. All five original services succeed;
main/post/collection finish11:28:15.248/11:28:35.619/11:28:40.720 UTC November3.
Independent terminal reading completes once11:29:04.321 in1.488s CPU/1.512s wrapper.
No full-run failure,GPU repeat or original collector/reader repeat occurs.
Earlier CPU preparation failures and co-host costs remain recorded without
training-time correction. No extension,sweep,confirmation or successor is selected.
The package and all readers are closed; no allocation is active at closure.
Check live state before new GPU work. The five-game goal remains unfinished.
