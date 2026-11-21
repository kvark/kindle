# Completed-return actor improves its paired Breakout control

Adding the optional completed-return actor loss raises frozen greedy
**65.208→95.375 (+46.26%)** and sampled **46.458→56.542 (+21.70%)** at the same
200k-action budget. Training takes **17m12.959s versus17m05.486s (+0.729%)**.
Both actual-initial controls match1.417 sampled/2.208 greedy. Every final arm
still has **zero of24 two-wall successes**; the original mean864/90% gate fails.

This is a useful result for the bonus-off recipe on one reused development
seed/cohort. It does not displace the retained action-effects1 recipe, whose
means are70.750 sampled/143.667 greedy in21m27.910s. The actor candidate is
19.80% cheaper in training time but has lower point estimates for both absolute
scores and initial-subtracted score/minute. Retain the existing primary baseline
and this cheaper bonus-off development point. No extension,third learner or
automatic confirmation follows. The five-game goal remains unfinished.

| Recipe | Sampled mean | Greedy mean | Training time | Sampled gain/min | Greedy gain/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired bonus-off control |46.458|65.208|17m05.486s|2.635|3.686|
| Completed-return actor100 |56.542|95.375|17m12.959s|3.202|5.412|
| Retained action-effects1 |70.750|143.667|21m27.910s|3.230|6.590|

Gain/min subtracts each actual-initial mean. All eight new frozen arms complete
24 selected natural episodes,zero truncations and zero learner updates. The
paired recipes share the qualified e3b02163 runtime,runner,teacher,initial arrays
and all learning settings except `replay_actor_scale=100`; both disable model
disagreement. The retained comparison changes runtime/runner and bonus/ensemble
alongside the new loss, so it does not isolate an actor-loss or bonus effect.

## Fixed paired comparison

The [protocol](../experiments/2026-11-03-cdp-episodic-actor-200k.md) fixes greedy
as primary, sampled as secondary, and10,000 paired lane-bootstrap resamples
with seed0. Each of eight lanes contributes its first three natural episodes.
The intervals quantify variation in this reused evaluation cohort; they do not
establish training-seed reliability or erase prior development selection.

| Candidate minus reference | Return difference,95% interval | Gain/min difference,95% interval |
| --- | ---: | ---: |
| Paired control,greedy |30.167 [2.167,60.001]|1.726 [.095,3.462]|
| Paired control,sampled |10.083 [.542,19.333]|.567 [.011,1.104]|
| Retained,greedy |−48.292 [−99.626,2.000]|−1.178 [−3.783,1.364]|
| Retained,sampled |−14.208 [−30.042,.961]|−.028 [−.855,.774]|

Six of eight greedy lane means and five of eight sampled lane means improve
over the paired control. Initial-subtracted score/min rises46.81% greedy/21.50%
sampled. Relative to retained action-effects1, its point estimate falls17.88%
greedy/.87% sampled; neither historical-comparison interval excludes zero.
The prespecified comparison supports development value within the bonus-off
pair, not an accepted improvement over the retained recipe or task mastery.

The final lane means are:

| Recipe / mode | Lane0 | Lane1 | Lane2 | Lane3 | Lane4 | Lane5 | Lane6 | Lane7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control greedy |70.000|53.333|38.667|58.667|75.333|80.000|82.667|63.000|
| Actor100 greedy |76.667|59.000|136.000|102.333|62.667|54.333|118.000|154.000|
| Retained greedy |150.333|129.333|59.667|105.667|238.667|172.667|113.000|180.000|
| Control sampled |57.000|48.333|42.333|54.000|54.333|39.333|38.333|38.000|
| Actor100 sampled |50.000|72.000|35.000|50.333|72.000|64.667|44.667|63.667|
| Retained sampled |89.667|51.000|81.333|49.667|78.000|57.000|81.000|78.333|

## Target activity and training exposure

The candidate receives2,526,690 completed replay rows and1,167,144 positive
rows across12,477 updates. Positive targets start at action1224 and occur in
12,406 updates. Normalized weight sums658,641.525; raw actor loss sums2643.097.
Loss-coefficient accounting passes on every update. These are repeated replay
exposures, not unique experiences or independent observations.

| Action window | Updates | Completed rows | Positive rows | Weight sum | Raw loss sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–50000 |3102|528,017|274,470|241,960.690|1,518.179|
| 50000–100000 |3125|662,206|352,386|202,753.776|644.707|
| 100000–150000 |3125|669,820|288,083|124,344.291|297.944|
| 150000–200000 |3125|666,647|252,205|89,582.769|182.268|

Control/candidate actual overrides are86,538 (43.269%)/91,135 (45.5675%).
Both training trajectories contain258 completed episodes and pass full pixel,
action and corrective-label replay. Final50k teacher cross entropy is.552791/
.545749; changing visited-state distributions prevent a held-out accuracy
interpretation. Guided last50 returns191.08/222.74 and unguided online41.68/38.32
are not the frozen competence results above.

Training maxima are399/415 points,including partial tails. Native actions at
pre-action scores216–431 rise709→1065,only.3545%/.5325% of all actions. Guided
actions in that band are4123/6105. Neither learner visits scores432 or higher.
These score-band exposure counts are not verified wall boundaries or proof
that coverage alone explains the remaining task failure.

## Bounded qualification retained

Kindle now has an optional direct actor loss on completed real experience.
The [implementation](../experiments/2026-11-03-cdp-episodic-actor.md) is off by
default. It uses positive completed-return advantage to imitate executed actions,
without an imagined bootstrap or new trainable parameters. The
[paired bounded check](../experiments/2026-11-03-cdp-episodic-actor-check.md)
passes for both a bonus-off control and the same recipe with actor scale100.

| Bounded arm | Actions / frames | Updates | Training time | Frozen1024 time |
| --- | ---: | ---: | ---: | ---: |
| Bonus-off control |8192 /32724|489|36.677s|.820s|
| Completed-return actor100 |8192 /32729|489|38.277s|.829s|

The candidate adds **4.36%** training time. Its conservative200k projection is
**17m07.959s**, below the30-minute development cap. This is an early projection,
not a measured full-run cost. Neither short frozen arm completes a natural
episode, so these bounded checks alone do not measure gameplay benefit.

The candidate receives56,138 completed target rows and33,105 positive rows over
489 updates. Positive targets first occur at action1,224 and appear in431
updates. Total normalized weight is29,895.926; raw loss sums391.679. These are
repeated replay exposures, not unique experiences or independent observations.
The target is active, but positive advantage does not establish good action
credit or improvement over the teacher.

## What changed

`replay_actor_scale` enables native discounted returns through a known true
terminal. At that terminal, replay walks the retained stream suffix backward,
using the stored scaled reward channels. Earlier eviction does not invalidate
a retained state's known future return. Unfinished/time-limited episodes stay
unlabeled; returns never cross resets or streams and are not checkpointed.

At ordinary replay states, the additional cross-entropy target is the executed
outgoing action weighted by `clamp((G_complete - V_current)/return_scale, 0, 1)`.
It averages over B*(T-1), including masked rows, and detaches features/targets
for this term. Only existing actor parameters receive its derivatives. Teacher,
imagined actor/critic and replay-critic losses remain present. Reports expose
completed/positive rows,weight sum and raw loss.

Model-derived disagreement bonuses are recomputed later, not stored as observed
rewards. The implementation rejects this loss when that bonus is active,
avoiding unlike reward definitions in the completed return and critic baseline.
Both bounded arms therefore disable action-effects1. They have12,375,569
parameters and261 checkpoint tensors, removing96 ensemble parameter/moment
arrays from the retained357-tensor recipe. The optional actor loss itself adds
no parameters. Resident device-target mode is rejected for this experimental
path; AMD Auto already constructs targets on CPU and learns through native GPU
graphs. Stored visitation rewards remain supported.

These changes matter for comparison: the completed pair uses a bonus-off
control. Its historical comparison with the retained recipe cannot isolate
the actor loss because the bonus, ensemble and runtime also differ. The objective is a clipped, normalized
positive-advantage imitation variant inspired by
[Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf),
not an exact reproduction. Completed outcomes can include guidance overrides;
the evaluation must turn guidance off.

## Qualification and execution

All **163+5 Rust CPU tests**, **1006 packaged Python CPU tests** and strict
workspace/binding Clippy pass. Tests cover terminal/timeout/reset/eviction and
stream semantics, reward scaling, prefix sampling, outgoing actions, masks,
positive weights/clipping, default serialization, CLI capability and report
accounting. Initial55 Python checks also pass. Ordinary tests never construct
a native agent; the packaged runner records zero construction attempts.

The guarded native test passes12 F64 loss/derivative cases over zero/partial/
unit targets,unimix0/.01 and expert loss absent/present. It uses the production
behavior graph with controlled parameters, checking actor output-bias gradients,
zero other actor/critic gradients and unchanged parameters. This does not
measure gradient usefulness on trained Breakout states. All221 pins and host
guards pass **15:59:04.487 UTC November3**. Native sourcee3b02163,
SHA`d406629fc67f7b9a4b440675af30c14be5bb7a77cf2fcda1fd6184dc682ba5c8`.

Before training, legacy1024 greedy actions preserve all357 retained tensors and
reproduce128 original semantic events. Both fresh initial checkpoints match
all261 retained non-ensemble arrays and each other exactly. All47 common
pre-update semantic events match. Both learners start at action376,finish489
updates,change CNN/reward/actor/value parameters and pass finite/schema checks.
The candidate's loss coefficient accounting passes on every update.

Each full8192-action training trajectory is replayed once on CPU, verifying
pixels,actions,corrective labels and override counts. Both have21 completed
episodes; actual overrides are3811 control/3853 candidate. Guided training
returns are not unaided competence. Both frozen arms have zero updates and
all261 tensors invariant. All255 bounded pins and338 retained-control pins pass.

Original service`kindle-episodic-actor-check-20261103`,PID1048190/invocation
`12873b06470242aeb15a4fb4bf3ec483`,launches16:05:10 UTC November3/source3018736f
and succeeds **16:07:03**. The controller completes16:07:03.970 in113.332s.
Independent terminal collection completes once **16:08:58.335** in.645s,
checking journal identity,terminal success,checkpoint file hashes,input pins
and raw-log target totals. All five GPU stages pass; no native/learner/replay/
reader stage fails or repeats. Source-edit context corrections occur before
execution and do not change that accounting. Host housekeeping occurs during
the short runs; wall times are not corrected and are not kernel timings.

Root `/x/Temp/kindle-episodic-actor-20261103/check` is closed. Do not rerun its
stages. Runtime,wheels,test executable and build cache remain under the parent
root. No Kindle service was active at that closure. The retained200k70.750/143.667
and400k92.792/173.708 gameplay results remain unchanged,all zero of24 two-wall
successes. The five-game goal remains unfinished.

## Full execution and two preserved CPU failures

The original full pair launches16:24:19 UTC November3/source783852b2.
Control training completes17m05.486s for200k actions/799,575 frames/12,477 updates.
Its100k/6,227 recovery passes16:32:53.148 with261 finite tensors,expected schema,
verified file hashes and read-only history files. Startup matches261 shared
initial arrays,47 pre-update events and first update376. All five control GPU
phases and their full training/frozen replays pass.

After those phases, the original service fails **16:45:24 UTC** in CPU exposure
analysis: a copied helper expects400k/24,977 and an older `expert_metrics` field.
Original `kindle-episodic-actor-200k-20261103`, PID1049147/invocation
`fcc4bc49735445638173a45196bddd44`, and its collector remain failed. The original
preflight passed queues,schemas,loss accounting and imports but did not exercise
that exposure helper against the new full review. Original failure/journal files
are preserved; there is no original passing completion.

Separate repair preserves69 completed-control files and all307 original pins.
Two valid fixtures,six corruption rejections and exact historical metric
reproduction pass in1.128s. Independent teacher metrics,corrected exposure
analysis and previously unstarted control plots finish **16:49:21.889 in3.458s**.
All22 repair pins pass. No GPU,learning,frozen assessment or full replay repeats.

Only the unstarted candidate launches **16:51:24.578 UTC**, sourcedc29240b,
under `kindle-episodic-actor-200k-resume-20261103`, PID1051198/invocation
`2e7e3aa5b60947f199c0d3264da02ebf`. Native,runner,seed1009,scale100,200k/12,477,
30min,100k recovery and all four frozen queues stay fixed. Original absolute
main/collector deadlines are retained. Its100k/6,227 recovery passes17:00:00.910;
training finishes **17:08:43**,17m12.959s for799,595 frames. All five candidate
GPU phases,replays,exposure and curves pass; final initial-greedy replay passes
17:12:47. Both full training audits take110.366/109.735s including CPU replay.

The continuation then fails **17:12:50 UTC** in the final CPU comparison. The
initial-trace reader removed `unix_time` but still compared episode
`elapsed_seconds` across runs. Exactly28 sampled and27 greedy episode durations
differ; all other transition/episode/reset fields match. The continuation and
its collector remain failed, with their original failures/journals preserved.
No completed stage is restarted and no failed assessment is silently replaced.

A separate closure preserves154 completed-candidate files. Two valid fixtures
and eight action/reward/boundary/reset/time corruptions validate the corrected
reader in1.423s. It excludes only episode elapsed time in addition to the existing
wall timestamp removal. All **799 sampled /794 greedy initial semantic events**
match. All261 initial arrays match and every frozen arm preserves261 tensors.
The original307/first-repair22/closure10 pins and retained338 pins pass.

The fixed comparison and standalone figure complete **17:19:25.280 in6.984s**.
Independent terminal verification finishes **17:19:38.264 in2.865s**, confirming
both original main identities/failures,all input/artifact hashes and all eight
paired intervals by a separate sum/quantile calculation. Closure is
`complete_with_cpu_postprocess_repair`; all four original services remain failed.
The successful result is the separate CPU closure, not a rewritten original run.

The pair totals400k training actions/1,599,170 frames/24,954 updates. Eight frozen
arms total153,040 actions/611,832 frames/126.880s withzero updates. There areten
successful GPU stages andzero repeats. No native/GPU/learner/replay failure
occurs. Training times are actual runner elapsed times; startup/recovery
observers (.254/.243s control,.317/.241s candidate),housekeeping and read-only
status checks share the host,with no wall-time correction or kernel timing claim.
Checkpoint checks establish saved weights/moments,not full replay/game/RNG resume.

Root `/x/Temp/kindle-episodic-actor-20261103/full` is closed. CPU repairs are
under `repair` and `closure`. Do not rerun learners,assessments,replays,collectors
or completed analyses. No Kindle GPU allocation or successor is active at
closure; check live state before separately selected work. Keep the original
gate and retained primary recipe. The five-game goal remains unfinished.

## Retained artifacts

The full comparison is `comparison.json`; all lane means are also exported in
`closure/lane-means.csv`. The inspected figure is `closure/comparison.png` with
an SVG companion. Actual logs,checkpoints,replays,videos,curves and failures
remain under the root above. Large evidence stays out of Git.

| Artifact | SHA256 |
| --- | --- |
| Original full declaration |`b48da6eae51aad8fe7761437c77e7a8612db89908b3643612b96756a5c2e5c51`|
| First repair declaration |`cebc98ad879d3734fdd506b846d5ce3a3e98f9bdf6c63ce95c4fa2e4f671541d`|
| Closure declaration |`dc6302f731775d5544b8446860fa0c81d28c63eabe0acf5a5e7b0a09bad93747`|
| Full comparison |`68dd1f937529d139f3779308587b7474f5252d0be1c65045f4a657f4b89c4c92`|
| Repaired completion |`962ab4f70fe80a6d973aaca2080b4cc9ee71b59f8cf6ad92b17777bd2a6f7016`|
| Independent terminal summary |`2e9059e4a502f034a89581bb41c03f2fb39c3bf0ed07d9c6f877520845b00f32`|
| Bounded declaration |`e1b2af39cc225ef2f72bf63117cb5a14d15b68831f48cddfad8f8fbde40a7ad4`|
| Bounded completion |`2c689d078d334c3fe86725f81f6f63bdee3db2345a8720a4c4d702083269ab77`|
| Bounded terminal summary |`9ca2f36456a276e3d4c0290c511dbb95a7f98c01a3dab89f72a07825eb3441eb`|
