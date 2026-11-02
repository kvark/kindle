# Breakout H30/R12 costs more and lowers both frozen scores

The [fresh 200k comparison](../experiments/2026-11-02-cdp-breakout-h30-r12-200k.md)
finishes **200,000 actions / 799,521 frames / 9,358 updates in 23m15.641s**.
Sampled **58.917** and greedy **124.708** fall below the retained H15/R16
control's **70.750 / 143.667**, while training takes **8.36% longer**.
Both final arms have **zero of 24 two-wall successes**. Retain H15/R16; no
extension, confirmation, horizon/ratio sweep or automatic successor is selected.
The five-game goal remains unfinished.

| Metric | H15/R16 control | H30/R12 candidate | Change |
| --- | ---: | ---: | ---: |
| Sampled frozen mean | 70.750 | 58.917 | −16.73% |
| Greedy frozen mean | 143.667 | 124.708 | −13.20% |
| Training time | 21m27.910s | 23m15.641s | +8.36% |
| Sampled gain over own initial per training minute | 3.230038 | 2.471982 | −23.47% |
| Greedy gain over own initial per training minute | 6.590132 | 5.266396 | −20.09% |
| Actual training overrides | 43.0585% | 50.4635% | +7.405 percentage points |
| Final50k expert cross-entropy | .560514 | .636098 | Higher |

Own initial sampled/greedy means **1.417 / 2.208** reproduce exactly. Each
assessment uses the original first three natural episodes per eight lanes,
with guidance and labels off. The original90%-two-wall gate stays fixed;
sampled remains primary. The guided last50-episode training mean rises
67.78→88.74, but neither guided returns nor replay fit is frozen competence.
The comparison plot is visually reviewed and its axes and score annotations
match the retained numbers.

## What changed and what this establishes

Keep12M64/Auto/N8/B16/T16/full BPTT/microbatch16, encoder4e-5, behavior8e-5,
dynamics4e-4, replay-value1/world gradients, expert100, sampled collection and
all retained guidance settings. Native447fe50/SHA35ad0db, exact runnercf4ea238
and teacher are unchanged. No build or previous experience reuse occurs.

H15/R16→H30/R12 changes rollout length, cadence, sampling, action-clock warmup
and relative learning work together: fewer world/expert updates and more
nominal imagined transitions. Reused learner seed1009/evaluation cohort gives
one development comparison, not independent reliability or an isolated horizon
mechanism. The earlier short-Breakout and long-Qbert regressions remain.

## Completed execution and original evidence

All **422 pins**, 100k/4,670 recovery, **357 matching initial and invariant
frozen tensors**, 47 pre-update transitions, 800 sampled/795 greedy actual-initial
semantic events, finite reports, full training pixel/action/teacher-label replay,
frozen action/outcome/frame/reset replays and videos, host guards, curves,
exposure and comparison pass. Ordinary frozen logs lack live pixel hashes;
no unavailable direct image equality is claimed. Four frozen arms total
**88,800 actions / 355,033 frames / 83.663s**, with zero updates or truncations.

Original main912196/invocation5bfcdae0db9f4682a7e9e6c72c98adcd launches
07:31:11.304 UTC November2/source538ef64. Main, postprocessing and collection
succeed **07:58:54.411 / 07:59:16.112 / 07:59:20.722**. All five original services
are terminal successes; no original GPU, replay, audit or collector stage fails
or repeats. The separate terminal reader passes once **08:00:10.140**, in
2.623s (2.641s wrapper). Recovery succeeds07:42:47.665 and its separate reader
passes07:43:12.544, retaining four files/206,067,161 bytes. CPU co-host durations
remain uncorrected. The documentation condensation changes no pinned input.

## Saved-log coverage and separate CPU repair

The candidate has **7,378** pre-action216–431 actions versus **5,947** in the
control, but actual overrides rise6,221 versus4,685. Native actions in that
range therefore fall **1,262→1,157**, or **0.631%→0.5785%** of training. Neither
run has any actions at432+. These are score fractions, not verified wall
boundaries or an isolated coverage mechanism.

There are280 completed candidate training episodes and eight partial tails,
versus263/eight in control. All29 candidate episodes reaching216 are guided
(control15, also all guided). Candidate training maximum392 is below control408;
unguided maxima are179/102. Nineteen guided candidate episodes reach the1920
cap versus13 control, with1,186 versus1,286 subsequent native actions. Frozen
sampled maximum215 never reaches216; greedy maximum399 reaches216 in six of24
selected episodes, with none at432.

The original coverage reader **fails at08:00:19.392**: its reused helper still
requires12,477 updates, while H30/R12 correctly has9,358. Its failed receipt,
reader, helper and52-pin declaration remain unchanged. It produces no completed
coverage review. This is a separate analysis failure; the original full learner
and completion audit remain successful.

A distinct CPU repair qualifies **08:03:28.686**, in1.754s (1.783s wrapper),
using the actual header/end and five rejected counter/configuration corruptions.
It checks H30/R12/200k/9,358 against both parsed end and accepted review. Parsing,
aggregation, thresholds and selection logic are unchanged; nine helper functions
match by syntax tree. All52 original,11 repair and422 full input pins pass.
The repaired reader completes once **08:03:41.018**, in2.918s (2.941s wrapper),
reconciling **288,800 actions / 1,154,554 frames** and all96 selected frozen scores.
It reuses the completed control analysis without recount, emulator imports,
new trajectories, GPU work or successful-stage repeats.

Root `/x/Temp/kindle-breakout-h30-r12-20261102` retains the original and repaired
readers and all receipts. The full learner is under `full/`, trial
`breakout-12m-64-b16-h30-r12-replayvalue1-expert100-200k-seed1009`.
Recovery, terminal and repaired coverage readers are complete; do not rerun
stages, collectors, qualification or readers. No learner or GPU allocation
remains active at closure. Check live state before separately selected work.

| Artifact | SHA256 |
| --- | --- |
| Full declaration | `aa57e4a522107cb28777cf4ef58f0f70eb56d19acbbfdb437bf6d6968f34b030` |
| Original full completion | `ac50672c952b5d49cc16ba4ff83449a92f891377abc0ade1aacca90cdeeae74f` |
| Full terminal summary | `5d7058c45b23f59ec29734039ab12adcc433c13e714b718f46d3ce77028f879e` |
| Original coverage failure receipt | `4858cf37b7fb38e6e92bc15a4bb8ba5d467963747b598193bb0808b611cf27a4` |
| Coverage repair declaration | `a7bd23fd12cd0e3d528f82cb12d5c81b2d1c818b69811bd16e13d49f9ec0f41d` |
| Repaired coverage review | `41792f03be92263e47637f27bc6ad6ea03eccb13f4138d51a4a6af2a3b00f4e3` |

## Earlier bounded execution and early-cost check

The [bounded check](../experiments/2026-11-02-cdp-breakout-h30-r12.md) completes
**8,192 actions /32,719 frames /367 updates in51.413s**. Frozen restore takes
**1,024 actions /4,096 frames /.892s**, withzero updates andno natural episodes.
The conservative200k projection is **23m00.725s**, below30 minutes. This
establishes execution andearly cost, not gameplay benefit or task competence.

Change H15/R16→H30/R12, keeping12M64/Auto/N8/B16/T16/encoder4e-5/behavior8e-5/
dynamics4e-4/replay-value1/expert100,sampled collection andall retained guidance
settings. Exact runnercf4ea238,native447fe50/SHA35ad0db andteacher are unchanged.
No build orprevious experience reuse occurs. Longer rollouts,fewer world/expert
updates,more imagined transitions,cadence,sampling andwarmup change together;
no isolated horizon mechanism orreliability claim. Earlier short-Breakout and
long-Qbert negative comparisons remain.

All382 pins,357 finite control-matching initial/invariant frozen tensors,
47 pre-update semantic events,367 finite aligned reports,changedCNN/reward/
actor/value,full pixel/action/teacher-label replay andhost guards pass. Actual
overrides3,994/8,192 aretraining exposure,not frozen competence. Two actual
commands,20 CPU configuration cases,357 schema tensors andderived cadence pass
without native/emulator construction. Source preparation andall stages pass;
no failures,retries orsuccessful-stage repeats.

Original main911725/invocationb1e9ffc11150406d90b365ea08afd560 launches07:25:19
UTC November2/source4a0032e andsucceeds **07:26:30.396**. Original collector
911728/66fa177bf888495b9f0f481222fa2de5 succeeds **07:26:40.871**. The separate
terminal reader completes once **07:26:54.448**, in1.078s (1.097s wrapper),
verifying original identities/journals,hashes andprojection arithmetic.

Root `/x/Temp/kindle-breakout-h30-r12-20261102/native-check`,trial
`breakout-12m-64-b16-h30-r12-replayvalue1-expert100-check`,isclosed. Do not repeat
the check,preparation,collector orreader. This bounded result established only
execution and early cost; the separately selected full comparison above reused
none of its experience. No allocation remains active atclosure; check
live state. The original90%-two-wall gate andfive-game goal remain unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`75a64618782dce7b96599242eed85e5d9f966e5933a41aa98572e78815dd8681`|
| Review |`8f2b90c264e5477f6456de069fe065b9a6330e8079f724e639825b530770a7e7`|
| Original completion |`697e58e0f54102e914af08337bf569daf80df17887485833d5746f28cc4800fd`|
