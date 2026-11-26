# Fresh Qbert dynamics1e-4 seed14009 confirmation

**Full confirmation is complete and misses the score gate.** Fresh14009 scores
14,770.833 greedy and14,254.167 sampled, each24/24 first pyramids, after25m09.906s.
Only one of two learners passes the fixed15k gate. All original full services and
audits pass; no GPU stage repeats. Earlier preparation/startup milestones below
are historical; the final result is at the end of this report.

The [fixed bounded check](../experiments/2026-11-01-cdp-qbert-dynamics1e4-seed14009.md)
completes **8,192 actions / 32,755 frames / 733 updates in57.303s**. Frozen
1,024 actions /4,096 frames take.865s withzero updates andno natural episodes.
The conservative200k projection is **1,538.902s (25m38.902s)**,below30min.
This establishes execution and early cost; it does not establish game competence.

Only learner/evaluation seeds12011/4000012011→14009/4000014009 change from the
[retained dynamics1e-4 recipe](2026-11-01-cdp-qbert-dynamics1e4.md). Keep
centeredCDP/12M64/Auto/N8/B16/T16/H5/R24/dynamics1e-4/encoder-base4e-5/greedy
collection/expert100. Native447fe50/SHA35ad0db,runner e161bf38 andteacherfa261c53
stay unchanged. All-state corrective labels,world gradients,unit weights and
20% continuation-guided episodes/cap1920 stay fixed. No previous experience is
reused. Seed changes affect both initialization and trajectories; there is no
isolated initialization or reliability claim.

All **774 pinned inputs**,357 own source-derived initial tensor shapes,357
invariant frozen tensors,733 finite supervised reports,full training pixel/
action/label replay andhost guards pass. First update376,residual credit.75.
CNN,reward,actor andvalue weights change. All175,782 labeled rows haveunit
weight; world/actor expert losses agree within8e-7. Actual overrides are2,653
(32.3853%); online guided returns are not frozen competence. No cross-seed
initial-value or trajectory equality is required or claimed.

CPU preparation completes23:19:30 UTC November1 in2.163s (2.197s wrapper):two
actual commands,23 config cases,16 runtime files,357-tensor schema and733/9340/
18715 cadence pass without native/emulator construction. It verifies the
reference's606 original/111 repair pins and explicit repaired closure. The
reference's original main andcollector remain failed; no original passing
receipt is fabricated. Direct-global checks accept the new bounded audit and
identifyCONTROL as missing in the retained failed full audit.

MainPID863135/invocation`38798e91970b471490a59719ee61d10a` launches23:22:41.810
UTC/source60f2142 andsucceeds **23:24:04.897**. Original collectorPID863139/
invocation`37a2a579dce0420eb06b2a8558915d37` succeeds **23:24:09.036**. Both
services areterminal success; no original stage fails orrepeats. Root
`/x/Temp/kindle-qbert-dynamics1e4-seed14009-20261101/native-check` iscomplete.
Do not rerun its learner,collector orreplays,or reuse its experience.

A supplemental inline terminal reader omitted the required completion argument
to the journal identity helper andfailed after original completion. Its failure
expression andseparate corrected reader are retained. The corrected2.091s read
completes23:25:42 UTC,verifies original invocations/complete journal payloads
andall774 pins,anddoes not invoke a collector,GPU orreplay. Original completion
bytes remain unchanged. No original experiment failure is inferred from this
supplemental reader error.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `ca6719ae77fb41820cf17d8f8712305862b454c584433145c3153b7a2aa19cb2` |
| Bounded review | `0137f5eafc735a905a05ef4575cebb9dc8e256830827e97be9e91d4883f9491e` |
| Original completion | `6c298a19dec490b3f4649562a6aeffa2c7a8eb9699f847ec745c39c09d866ee9` |
| Guided replay | `2982377c5df532cdcd2232935d23f92d7c9c7cd82b9c8acbef39c9e2fe210ddd` |

## Separate full confirmation is CPU-qualified

The [fresh200k protocol](../experiments/2026-11-01-cdp-qbert-dynamics1e4-seed14009-200k.md)
keeps the recipe and30-minute cap fixed,with100k/9,340 recovery andfour actual-
initial/final sampled/greedy24-episode assessments. Greedy remains primary at
the unchanged15k/90% gate; no full task result is available.

Preparation completes23:25:46.971 UTC in6.895s (7.009s wrapper). Five commands,
32 config/five teacher/eight mode/17 header/four observer/12 recovery cases,
six retention/two actual-collector fixtures,51 paths/19 scripts andfive direct-
global checks pass. All774 bounded/606 reference/111 repair pins remain intact.
No native module,agent or emulator is constructed. Audit/preflight/comparison
bindings pass,andthe exact old missingCONTROL error is detected. This is a
focused regression check,not general proof of controller correctness.

Full preflight SHA256
`d36b832b95eeb078a050c822ae65ab02b86b9b321fff1fd5cac02de3b9341805`.
No full learner is launched at this qualification boundary. No retry,cap increase,
seed substitution,extension orautomatic successor is selected. The five-game
goal remains unfinished.


## Full launch and startup (historical)

The separate200k learner launches **23:27:17.442 UTC November1**,source9cce545,
with768 pins,mainPID863643/invocation`80c3cd392a2c45e68c200e362ab6c00c`.
Allfour original observers start:postprocessPID863646/invocation
`ea3bb7b2c5b5425b919202facbce10c8`,progressPID863649/invocation
`447c3905d0cf4ffa9f4eca1926849d84`,recoveryPID863783/invocation
`ff8f7644f9d041bfb9b494d6760f5088`,collectionPID863786/invocation
`b6b024156b5f4684b61eeae7372ee8f2`.

Startup23:28:11.937 verifies all768 pins,357 bounded/full matching initial
tensors,47 identical pre-update semantic events,correct cadence and609 finite
supervised reports through6,872 actions. All47 reference events differ across
seeds; this is descriptive,not a failure or causal initialization result. No
cross-seed initial-value equality isrequired. The2.387s CPU startup wrapper
co-hosts withtraining; retain its cost without correcting training wall time.

Declaration SHA256
`6301683e1e1f5f0907d4b99489b36abfcf7b5ee5f2091a66d7381d6a1d98c57d`.
Keep the fixed200k/18,715/30min budget,100k/9,340 recovery andfour frozen arms.
No final frozen result isavailable. Preserve the original main andobservers;
do not restart oroverlap another GPU allocation. Bounded experience isnot reused.
Fresh confirmation andthe five-game goal remain unfinished.


## Original100k recovery passes

The original recovery observer completes **23:40:24.950 UTC November1** under
PID863783/invocation`ff8f7644f9d041bfb9b494d6760f5088`. The100,000-action/
9,340-update checkpoint has357 finite tensors. Itsfour retained files total
206,067,158 bytes;retention takes.319s andpreserves the source. The observer
also checks9,735 sequential finite reports through104,216 actions. Recovery
metadataSHA256`af0a7d123c834a36aa36ae661812fccdc6c9722aa9042e22761986191c94a3e4`;
retentionSHA256`eddcdd6e8111622dc4e97e52220e48e289618e8c7e38e175f75b7f30479ad0ba`.

A separate2.216s read-only confirmation (2.236s wrapper) completes23:41:11 UTC,
checks all768 active inputs,allfour retained hashes,exact original invocation
andthe single recorded recovery audit/completion. It does not repeat the audit,
copy,collector orGPU work. MainPID863643/invocation80c3cd392a2c45e68c200e362ab6c00c
remainsactive;progress,postprocess andcollection continue. Keep co-host timing
without correction. Final200k/18,715/30min andallfour frozen assessments remain
unchanged andpending;no midpoint evaluation orcheckpoint selection occurs.

The [prepared Breakout critic diagnostic](../experiments/2026-11-01-cdp-breakout-current-critic.md)
adds9.541s repaired CPU preparation plus its retained test/argument failures
while this learner continues. All768 inputs andoriginal identity stayfixed.
No second GPU allocation orqueue iscreated;selection waits fororiginal Qbert
closure andidlehost. Saved recovery omits replay,live beliefs,games andRNG.


## Full confirmation completes below the score gate

The original full run completes **200,000 actions / 799,795 frames / 18,715
updates in 1,509.906s (25m09.906s)**, within its fixed 30-minute cap.

| Learner seed | Greedy mean / first pyramids | Sampled mean / first pyramids | Training |
| --- | ---: | ---: | ---: |
| Development12011 |15,471.875 /24 of24|14,862.500 /24 of24|25m40.027s|
| Fresh14009 |14,770.833 /24 of24|14,254.167 /24 of24|25m09.906s|

Fresh14009's actual-initial means are147.917 greedy and205.208 sampled, withzero
first pyramids. Both final means miss15,000 despite24/24 first pyramids. Thus
**one of two learners passes the primary greedy gate**; this confirmation does
not establish reliable success. The training recipe, native, runner and teacher
are unchanged; only learner/evaluation seeds differ. Initialization and evaluation
trajectories change together. No matched-state causal claim follows.

All768 pins,100k/9,340 recovery,357 own bounded/full initial tensors and357
invariant frozen tensors perarm pass. Full pixel/action/label training replay,
all four frozen replays/videos,host guards,curves,exposure andcomparison pass.
Cross-seed initial traces differ as expected; their equality is not required.
Actual guidance is59,478 actions (29.739%), and the aided online last50 mean is
15,280.5. Late50k replay expert CE is.064864 versus development.069685; this
lower loss does not imply better frozen play. The comparison plot was inspected.

Original main succeeds **00:00:06.318 UTC November2**, postprocessing
**00:00:27.036**, collection **00:00:58.833**. Progress and recovery also succeed;
all five original services are terminal. No original stage fails or repeats.
The prepared read-only terminal summary takes6.056s (6.075s wrapper), verifies
all original identities, pins, completion files and the prior12011 repaired
closure, and constructs no native agent or emulator. Earlier bounded reader
and CPU preparation errors remain recorded above; none changes this original
full completion. All CPU co-host costs remain uncorrected.

| Full artifact | SHA256 |
| --- | --- |
| Learning review | `144d0e638b32f067166b7542ee06e956a6681ce9c1d0983f9ea656700dfdfe39` |
| Original completion | `9a56dbdaa703690d573d25ae80d69eeca4155d0bd5eaf3425093e97a666826e8` |
| Comparison | `fc9414a89658e9d06588caf448407e5d20000ba89ff39b049bfbd00b268358db` |
| Read-only summary | `956a7394f7598d3b48eb32c9a36e72e254c4f89d986afa015e8264d4e7c94702` |

Retain this completed result without extending its budget, substituting seeds,
repeating stages or selecting a third learner. No Qbert successor is selected.
The five-game goal remains unfinished. Breakout's separately prepared frozen
critic diagnostic can now be considered on the idle host.
