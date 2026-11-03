# Stronger imagined-policy weighting worsens frozen Breakout

Increasing `loss_scales.policy` from 1 to 100 lowers sampled frozen score by
**45.94%** and greedy score by **35.88%**, with **1.67% more training time**.
Retain policy1, expert100, imagined-value1/replay-value1 and actor/critic
representation gradients off. The result does not support this stronger policy
weight in the retained recipe. Current actor-gradient dominance remains
unmeasured; this comparison does not isolate the failure mechanism.

| Measure | Retained policy1 | Candidate policy100 |
| --- | ---: | ---: |
| Sampled final / actual initial | 70.750 / 1.417 | 38.250 / 1.417 |
| Greedy final / actual initial | 143.667 / 2.208 | 92.125 / 2.208 |
| Two-wall successes, each final arm | 0/24 | 0/24 |
| Training time | 21m27.910s | 21m49.372s |
| Initial-subtracted sampled score/min | 3.230038 | 1.687832 |
| Initial-subtracted greedy score/min | 6.590132 | 4.120295 |

The candidate completes **200,000 actions / 799,581 frames / 12,477 updates**.
Initial-subtracted score per training minute falls **47.75% sampled / 37.48%
greedy**. The final training duration includes final checkpoint completion;
the earlier 200k progress report was 1,297.274s and is not the final duration.
Four frozen arms total 73,544 actions / 294,011 frames / 68.243s, with zero
updates and truncations. Each selects the first three natural episodes in
eight lanes, with guidance and labels off. The 90% two-wall gate is unchanged.

Only the policy coefficient changes in the retained native configuration.
Score-function and entropy terms scale together; expert100/world imitation,
critic1/1 and actor/critic representation gradients off remain fixed. Native
447fe50/SHA35ad0db and teacher are byte-identical. Runner5a86e224 adds 12 CLI lines.
The [full protocol](../experiments/2026-11-03-cdp-breakout-policy100-200k.md)
records the unchanged recipe and reused seed1009/evaluation4000001009. No
experience transfers from the bounded check or retained control. This is one
development comparison, not independent confirmation or an entropy ablation.

Late replay expert CE improves .560514→.548005, while guided online last50
is 67.78→67.68. Actual overrides are 89,090/200,000 (44.545%), versus 43.0585%
in the control. Better replay fitting does not imply better unaided play.
Native training actions at raw scores216–431 fall 1,262→1,196 (.598% of all
actions), with none432+. Candidate greedy reaches216 in3/24 selected episodes;
sampled reaches it in none. These fixed bins describe exposure, not verified
wall transitions or a causal coverage limitation.

All436 input pins,100k/6,227 recovery,357 matching initial/invariant frozen
tensors,47 pre-update events,800/795 matching initial semantic events,exact
cadence,finite reports and changed CNN/reward/actor/value pass. Behavior total
reconstructs as `100*policy + value + replay_value + 100*expert` on all12,477
reports; maximum absolute error is3.06e-5. This checks reported weighting,
not derivatives or optimizer correctness. Full training pixel/action/label
replay includes260 completed episodes. Frozen replays/videos,host guards,
curves,exposure and comparison pass. The comparison figure is visually checked.

The fixed coverage reader reconciles273,544 actions/1,093,592 frames with
52 diagnostic/436 full pins and reuses the completed control analysis. It
performs no emulator replay,GPU work or learning. Original main/post/collection
succeed **10:00:29.787 /10:00:57.364 /10:01:01.652 UTC November3**, source
`0c40e657b3f9d9c7e79038ca0c81f8f1a011c852`. Allfive original services succeed.
Independent terminal/coverage readers complete once **10:02:01.134 /
10:05:11.888**, taking1.338s/3.038s CPU (1.359s/3.060s wrappers).

No full-run stage failure,GPU repeat or original collector repeat occurs.
Two preceding bounded CPU arithmetic-fixture failures and the missing-generator
path in coverage-reader qualification remain preserved, with their CPU repairs.
The full learner's preparation has no failure. CPU co-host costs remain recorded
without wall-time correction. These prior qualification failures do not change
the successful original full-service outcomes.

Root `/x/Temp/kindle-breakout-policy100-20261103/full` is closed. Recovery,
terminal and coverage readers are complete; do not repeat stages or observers.
No Kindle allocation remains active at closure; check live state before work.
No extension,coefficient sweep,confirmation learner or automatic successor is
selected. Breakout and the five-game goal remain unfinished.

| Artifact | SHA256 |
| --- | --- |
| Learning review | `9bec0e67a95b07b3ead3fca08a243d3af2f07e5ec8745a56109ea1438a018727` |
| Completion | `387760564007944bfdac12c2fbaea9483bc45ab13b197f424735e4acfba24ea3` |
| Independent terminal summary | `26e0da3d40c394f63bddf4e8e754dd8a4bca7c2231646d280b5ed325176b56a0` |
| Fixed coverage review | `ab72765dc9a31eb7489b117056f80057b9eb0050ed243fc9359fdaba233b5462` |

## Bounded execution and early cost

The [bounded policy100 check](../experiments/2026-11-03-cdp-breakout-policy100.md)
completes **8,192 actions/32,728 frames/489 updates in46.910s**. The1,024-action
frozen greedy arm takes.889s, with zero updates and no natural episodes. The
conservative200k projection is **1,259.778s (20m59.778s)**, below30min.
This bounded check establishes execution and early cost; full gameplay is reported above.

Only imagined-policy loss weight1→100 changes. Score-function and entropy terms
scale together, while expert100/world imitation,imagined-value1/replay-value1,
actor/critic representation gradients off,12M64/Auto/N8/B16/T16/H15/R16 and all
other retained settings stay fixed. Native447fe50/SHA35ad0db and teacher remain
byte-identical. Private runner5a86e224 adds12 CLI lines. Current actor-gradient
dominance is not measured, and this does not predict an isolated mechanism.

All396 pins,357 control-matching initial/invariant frozen tensors,47 pre-update
semantic events,exact cadence,finite reports,changed CNN/reward/actor/value,
full guided pixel/action/label replay and host guards pass. Behavior total loss
reconstructs as `100*policy + value + replay_value + 100*expert` on all489 reports;
maximum absolute difference is3.06e-5, within the fixed tolerance. This verifies
reported weighting, not derivatives or optimizer correctness.

All489 reports have supervision, with117,049 repeated labeled replay rows and
the same total weight. Maximum raw world/actor expert-loss difference is8e-7.
Actual teacher overrides are3,733/8,192; online guided returns are not competence.
No bounded experience will transfer to a later learner.

Source `e5ebbaeebfb1d9021cf7841ad3696d7446785cc2`; original mainPID999583/
invocation17ae4bd7576f4022bcd23d308898587c succeeds **09:30:47.023 UTC November3**.
Original collectorPID999586/invocation85c31fea96c4446ea76d4923fe2fea17 succeeds
**09:30:52.508**. Independent terminal reader completes once **09:31:21.424**
in1.927s (1.944s wrapper), verifying original identities,artifacts and pins.
No bounded stage fails or repeats. Two preceding CPU arithmetic-fixture failures
and corrections remain preserved as detailed in the protocol; no native agent
was constructed in preparation.

Root `/x/Temp/kindle-breakout-policy100-20261103/native-check` is closed; do not
repeat qualification,learning,restore,replay,collection or terminal reading.
No allocation is active at bounded closure. A separately fixed
[fresh200k comparison](../experiments/2026-11-03-cdp-breakout-policy100-200k.md)
was selected to measure final/actual-initial sampled and greedy gameplay
against70.750/143.667,all0/24 two-wall, with100k recovery and30min training cap.
That full comparison is now complete, as reported above.
The five-game goal remains unfinished.

Full CPU preparation passes **09:33:15.246 UTC November3** in2.406s (2.521s
wrapper):18 compiled scripts,five actual commands,38 config cases,19 startup
header cases,four observer-parser cases,14 recovery-body cases,six retention
cases,two collector-retention cases and ten command-validation cases. All396
bounded/338 control pins and357 shapes match. Actual bounded/full loss arithmetic
is identical; its qualified cases are retained without repetition. No emulator
or native agent is constructed, and no full preparation failure occurs.

| Artifact | SHA256 |
| --- | --- |
| Bounded declaration | `897f8faf5ca7655220dca54d0b29e2f3000b10e59f5922e8d568528caf284f2c` |
| Review | `0ebbce59bb9c476b8567a245dcdbfaa6580f9f0df24f5de8f035fadd2a30adc3` |
| Completion | `df88afd36854d25e222f3728c53bf52b1a34d2fef01cf8293f1f88000404189b` |
| Independent terminal summary | `a3a320fca7bb48a4dcd52f476e27e5566f784ea3a1f01b860aab6f3d992900ed` |

## Launch and startup

The full comparison launches **09:34:23.440 UTC November3**, source
`0c40e657b3f9d9c7e79038ca0c81f8f1a011c852`, with436 pins. MainPID1000011/
invocation`dbf3484b87e947cfa048d424a144d958`; service
`kindle-breakout-policy100-200k-20261103`. Original postprocessing/progress/
recovery/collection services start once. Startup passes **09:34:53.761**
(1.505s wrapper):357 bounded/control-matching initial tensors,47 pre-update
transitions,first updates376/392/408/424/440/456,and250 finite reports through
4,360 actions. No restart,overlap or extension occurs; the completed outcome is above.

DeclarationSHA`377c240b0aae298a1ff87300019d415a7deabce7aca3c65bf419c7127a685ebf`.
The100k recovery,all four assessments and original terminal collection subsequently
pass. No full preparation failure or repeated GPU stage occurs.

The independent terminal reader qualifies five score checks at09:35:51.569,
3.545s body/3.572s wrapper. Coverage qualification first fails in2.440s because
the copied reader pins a separate prepare_coverage.py, while this package uses
one combined prepare_readers.py. Preserve that source/failure and the.019s
CPU repair. The corrected reader pins the actual generator,original source and
failure;52 diagnostic/436 active pins,seven config cases and five retained
boundary fixtures pass09:36:34.672 in2.439s (2.460s wrapper). Original process
identity stays unchanged. No terminal analysis,emulator or GPU stage repeats.

Keep these CPU co-host timings without correcting learner wall time. Recovery
confirmation ran once after the original100k observer succeeded, while the
learner remained active. Terminal and coverage summaries each ran once after
their required original completion artifacts existed. Their private sources
and provenance were fixed before frozen outcomes; do not repeat the readers.

## Recovery retained during the original learner

The original100k observer succeeds **09:45:29.841 UTC November3**, retaining
100,000 actions/6,227 updates,357 finite tensors and four files/206,067,164 bytes.
Its audit also verifies6,373 finite reports through102,344 actions. Independent
confirmation passes once **09:45:50.268** in1.291s (1.309s wrapper), checking all
436 input pins,original service identity,metadata and retained file hashes.
No restore,observer restart or repeated checkpoint audit occurs.

MainPID1000011/invocationdbf3484b87e947cfa048d424a144d958 remained unchanged
through recovery confirmation; all original services are now terminal success.
Retained weights and metadata omit replay,belief,games and RNG; they are not
uninterrupted training recovery. Keep CPU co-host time without correction.

Recovery reviewSHA`ba51d1417b97567ddb218b5411ee3f04df149524e3c470bbe35282a0d305a18d`;
retentionSHA`400d91b82609cb9d87644d47941e34318d412dac67442cc4ed2c9bef6c62e7f7`;
metadataSHA`b73b9a62f7945a8d64d0fc86b4814f00fc0f67a03ac09925a1449997fdb8118f`;
confirmationSHA`abf7c27de7215c4a1bb481505775063c23333bea87f8116721e6ad1a51e33949`.
Do not repeat the completed recovery reader.
