# More Breakout learning raises scores but lowers return per training minute

The fixed fresh400k/R16 comparison scores **92.792 sampled /173.708 greedy**,
up **31.15% /20.91%** from the retained200k control's70.750/143.667. Training
takes **43m36.578s**, up103.16% from21m27.910s. Both final modes still have
**0/24 two-wall successes**. Retain200k as the cheaper baseline and400k as a
higher-score budget point; neither passes the original90%-two-wall864 gate.
No extension,intermediate-checkpoint promotion,confirmation learner or automatic
successor is selected. Breakout and the five-game goal remain unfinished.

| Budget | Sampled final / actual initial | Greedy final / actual initial | Training | Initial-subtracted score/min, sampled / greedy |
|---|---:|---:|---:|---:|
|200k /12,477 updates|70.750 /1.417|143.667 /2.208|21m27.910s|3.230 /6.590|
|400k /24,977 updates|92.792 /1.417|173.708 /2.208|43m36.578s|2.095 /3.933|

Initial-subtracted score per training minute falls **35.13% sampled /40.33%
greedy**. This is a useful absolute gain with a worse compute tradeoff, not a
speedup or evidence that unchanged continuation will reach the task gate. The
[halfway assessment](2026-11-03-cdp-breakout-halfway.md) previously found larger
gains from100k to200k. These fixed points do not identify an optimal budget,
an asymptotic plateau or the architecture's causal limitation.

## Fixed comparison and exact common prefix

The [declared protocol](../experiments/2026-11-03-cdp-breakout-budget400k.md)
completes400,000 actions/1,599,381 frames/24,977 updates within its50-minute cap.
It preserves centeredCDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,behavior8e-5,dynamics4e-4,policy1,imagined/replay critic1/1,
constant actor/world expert100,unit action weights and all-state corrective
labels. Actor/critic representation gradients remain off. Sampled collection/
posteriors,20% episode guidance/cap1920,full18/sticky.25/repeat4/noops0,native
pooledRGB/one GPU64 resize and raw rewards remain unchanged. H3,centroid,crop
and imitation decay stay off. Native447fe50/SHA35ad0db,runnercf4ea238 and the
original teacher are unchanged; no build or new bounded learner is needed.

This is a fresh life with reused learner1009/evaluation4000001009, without prior
weights,replay,beliefs,RNG,games or experience transfer. Recovery checkpoints
retain100k/6,227,200k/12,477 and300k/18,727 updates. All357 tensors match the
retained control at initial,100k and200k, including optimizer state. The complete
common prefix matches38,050 events:25,000 transitions,12,477 learner reports,
263 episodes,259 resets,48 progress records,two checkpoints and one header.
Only declared timing,budget/path and serialization ordering are normalized.
This verifies recorded behavior and checkpoint contents, not complete live
replay/RNG/belief equality. The comparison passes14:53:46.448 UTC November3.

All four frozen arms use the first three natural episodes per eight lanes,
guidance/labels off,zero updates and zero truncations. Each preserves357 tensors.
The actual-initial sampled/greedy traces reproduce800/795 semantic events.
The four arms total91,952 actions/367,641 frames/84.321s. The original evaluation
cohort is reused; this is development evidence, not independent confirmation.

## Learning fit and exposure

Final50k replay teacher CE falls.560514→.411859. Guided online last50 return
rises67.78→116.0; actual overrides are148,372/400,000 (37.093%), versus43.0585%
in the200k control. These fit and exposure measures are not unaided competence,
and the changing visited-state distribution prevents a matched-generalization
interpretation of replay CE.

Training still never reaches432 cumulative points. Maximum recorded episode
score,including partial tails,is408. Thirty-three episodes reach216; only2,397
native actions (0.59925% of all training actions) occur at pre-action scores
216–431. Another7,991 actions in that range are guided;389,612 actions occur
below216. None occurs at432 or above. These are score-range exposure counts,
not verified wall boundaries or causal proof that coverage alone explains the
frozen gate failure.

Wrapper timers attribute2,286.222s (87.37%) to learning,127.825s (4.89%) to acting
and125.726s (4.80%) to environment stepping. They are not kernel-exclusive GPU
measurements. CPU controller repair,checkpoint comparisons and milestone reads
co-host with training; retain wall time without correction. The comparison plot
is visually checked: its curves are guided online scores, with frozen scores
and training costs stated separately in the caption.

## Verification and closure

All446 original pins,24,977 finite learner reports,three four-file recoveries,
357 finite initial/invariant frozen tensors,changed CNN/reward/actor/value,
full training pixel/action/teacher-label replay,frozen replays/videos,host guards,
curves,exposure,coverage and comparison pass. The full training replay takes
199.224s. Ordinary frozen logs do not contain live pixel hashes; their complete
recorded action/outcome/frame/boundary replay passes without asserting direct
live-image equality.

The original launch wrapper fails14:31:31 UTC on serialized initial-file hashes,
after starting the learner/progress/postprocessing services. All initial tensor
values,payload bytes and parsed headers match; JSON header ordering changes file
hashes. The failure/log/receipt remain preserved. A separate six-case CPU repair
verifies each raw file against its recorded identity, then compares canonical
headers and exact payloads. It starts the previously unstarted recovery and
collection observers once. Original pinned scripts/protocol,training commands,
native runtime and learner remain unchanged. No successful GPU stage repeats.

MainPID1034231/invocation13c5fb9a8be543deb2c279522c8498a2 launches14:31:26.282
UTC November3/source4bb0d101. All five original services succeed. Main,
postprocessing and collection finish **15:21:22.175 /15:21:33.374 /15:22:02.467**;
recovery finishes15:04:28.160. Progress has no separate resource-consumption
timestamp; its original journal identity and terminal success pass.

The fixed independent terminal reader and repair closure pass once at
**15:22:18.072**, taking2.689s (2.708s wrapper). They verify original identities,
output hashes,recoveries,scoring and comparison arithmetic. Closure is
`complete_with_cpu_launch_repair`; the original failed launch wrapper is not
relabeled. All11 repair pins pass. No training,frozen,replay,postprocessing,
collector or terminal-reader stage repeats. No Kindle GPU allocation remains
active at closure; check live state before separately selected work.

Evidence root: `/x/Temp/kindle-breakout-budget400k-20261103`, with trial
`full/breakout-12m-64-b16-r16-replayvalue1-expert100-400k-seed1009` and separate
`cpu-repair`. Do not restart closed services or repeat completed readers.

- Learning reviewSHA`ba1bfc0483b256fad5cd387ca9e9dff1172cac28b2fc403f05c53b5984b7daaf`.
- ComparisonSHA`833e44df990a3e15921bdffefcd144ca6df513aec7d6406540f5c418fab04a6d`.
- Original completionSHA`7d3d79c3bb1df48cce4b0409f1e21b6857e17e9ef163160a669281c7250ea571`.
- Terminal summarySHA`57b0086561103d21bbd8399270d367682960ce916b772260d15f9542adf2a8e5`.
- Repair completionSHA`f0cdca667135bb1671a61dff43cb61de9c6242340fe65c1a5548bdd5215c9aaf`.
