# Qbert behavior rate: similar frozen quality with lower observed cost

The fresh200k comparison completes in **20m52s**, versus **28m34s** for the
retained H5/R24 control: **26.92% less training time**. Greedy score is essentially
unchanged and sampled score rises3.00%;both still miss the original15,000 gate.
Retain the qualified faster Python teacher. Doubling actor/critic rate does not
establish a clear quality improvement or replace the better-supported base rate.

| Frozen assessment,24 selected natural episodes | Control | Behavior8e-5 |
|---|---:|---:|
| Greedy mean |14,339.583|14,358.333|
| Greedy first pyramids |24/24|23/24|
| Sampled mean |13,761.458|14,173.958|
| Sampled first pyramids |24/24|24/24|
| Actual-initial greedy mean |1.042|1.042|
| Actual-initial sampled mean |171.875|171.875|
| Training seconds |1,713.661|1,252.376|

Both initial policies clearzero first pyramids. Training is200,000 actions/
799,828 executed frames/18,715 updates. Four frozen arms total128,480 actions/
513,768 frames/109.844s,zero updates or truncations. Selection remains firstthree
natural episodes per lane;the extra completed greedy episode is excluded.
Initial-subtracted return/minute improves37.01% greedy and40.99% sampled on this
reused development seed/cohort. These ratios do not establish seed reliability.

The only learner-config change is behavior rate4e-5→8e-5,affecting actor and
critic together. The separately qualified equivalent Python teacher also changes
bytes and CPU cost. Native447fe50/runneraba4300d remain unchanged. Observed acting/
feedback time falls566.324→184.264s,accounting for82.83% of total time reduction;
learning time falls1,051.583→971.995s and environment time stays60.437→60.647s.
The acting timer includes teacher and native work. This comparison does not
isolate teacher speed,rate effects,actor versus critic,or native/backend speed.
The exact label/pixel replay qualification supports retaining the faster teacher.

All524 pins,357 matching initial/invariant frozen tensors,47 pre-update events,
1,238 sampled/916 greedy actual-initial semantic events,retained100k/9,340 recovery,
18,715 finite supervised reports,full training pixel/action/label replay,frozen
outcome replays/videos,host guards,curves,coverage,exposure and comparison pass.
The comparison plot is visually readable. Actual overrides47,421/200k=23.7105%;
guided online returns and replay imitation fit are not frozen competence.

Main launches15:17:59 UTC November1/source6089e8b,originalPID809245/invocation
`361f29163c1e4b2896fd352f2b905ca8`. Training ends15:38:56,all GPU stages15:41:02,
main15:45:24,postprocessing15:45:34 and original collection15:46:07. All five
original services succeed;no failed or repeated GPU/stage. Full guided CPU replay
takes200.637s. The host is idle at closure. Original CPU co-host timings below
remain without wall-time correction. No extension,rate sweep,confirmation learner
or automatic successor is selected;the five-game goal remains unfinished.

Root`/x/Temp/kindle-qbert-behavior8e5-20261101/full`,trial
`qbert-12m-48-b16-h5-r24-encoder4e5-behavior8e5-greedy-expert100-200k-seed10007`.
The derived`final-summary.json`reads completed audits without repeating collection.

| Completed full receipt | SHA256 |
|---|---|
| Declaration |`03a3ac6234dd23eee005c32c78bbb50af0903a00d1ee84924357931d10fc0650`|
| Learning |`3b8915550c3a85a6cc675f45495ebe0aebd2b7c75c6606c922fdeeabc84c549e`|
| Postprocessing |`c9234999b6eda79b96c5d733b6e6dfc2bac65d86d95f85350c3c94f1c5ab8ce0`|
| Comparison |`c430ffd8bfc2020f83c815b2ed7cd1159259fbba34c1108ae13eaaff0314b084`|
| Full training replay |`46d589f1f91e507c9f8b5e9fb1811c15c280959ea79f76cf942b9e06fd5b2032`|
| Original completion |`aa66f0dd69f66e9ab1ac334d35394721b10cee91c2f1dcd9fa95d8b1cc3c58c9`|

## Bounded execution and cost check

The bounded behavior4e-5→8e-5 check completes8,192 actions/32,761 frames/
733 updates in **47.940s**. Frozen1,024 actions/4,096 frames take.803s withzero
updates and no natural episodes. The conservative200k projection is21m27.462s,
below35min. This establishes execution and early cost,not gameplay benefit.

All496 declared inputs remain unchanged. All357 initial tensors equal the
retained10007 H5/R24 control,47 pre-update events match,357 frozen tensors stay
invariant,and all733 supervised reports are finite. CNN,reward,actor and value
weights change. Full training input-pixel/action/corrective-label replay and both
host guards pass. Actual overrides3,072/8,192=37.5%;guided returns are not frozen
competence. No failure or repeated GPU stage occurs.

The only learner-config change is actor/critic learning rate4e-5→8e-5. Keep
centeredCDP/12M48/Auto/N8/B16/T16/H5/R24,world/base/encoder4e-5,dynamics4e-4,
greedy collection,expert100 and20% episode continuation guidance/cap1920.
Native447fe50/runneraba4300d remain unchanged. The separately qualified equivalent
Python teacher overlay changes source bytes and CPU cost;its full200k label replay
already passes. This timing is not an isolated native speed comparison. Both
actor and critic adapt faster;their individual effects remain unresolved.

Main launches15:13:36 UTC November1/sourceb735644,originalPID808860/invocation
`bd7df223f7a244d3a812138ec70f80d5`,and succeeds15:14:46. Original terminal
collection succeeds15:15:08,1.195s CPU wrapper. The GPU is idle at closure.
Do not repeat the check,collector or successful preflight. The five-game goal
remains unfinished;see the [separately selected full comparison](../experiments/2026-11-01-cdp-qbert-behavior8e5-200k.md).

Root`/x/Temp/kindle-qbert-behavior8e5-20261101/native-check`,trial
`qbert-12m-48-b16-h5-r24-encoder4e5-behavior8e5-greedy-expert100-check-seed10007`.
Evidence:

| Receipt | SHA256 |
|---|---|
| Declaration |`960ee0f3b8041414488bd058987fdd281c689ac5e69153b3abe1b6c596faa41a`|
| Learning review |`bd09eb0af6565a1786df61b2049be638440265c089bddf417efa32106e7b8ada`|
| Full guided replay |`f856d99e8440e9cdfe7221403952691a670ca5ea18a29611685d6138b67d124e`|
| Completion |`8c9637aa595554dfa84fd2b34fdc9c97d0b76a2b63fcdae93c1be1af717a5b37`|

Bounded CPU preflight1.101s validates two CLI paths,18 config cases,357 schema
tensors,442 control inputs and16 overlay files,withzero native/emulator
constructions. The copied protocol constant is corrected by an explicit selector
entry point before declaration;qualified sources remain unchanged. Full CPU
qualification15:15:20 takes3.395s (3.506s wrapper):five CLI paths,28 config,
five teacher-identity,eight mode,16 header,four observer,11 recovery-body,six
retention andtwo actual observer/collector integration cases,51 resolved paths
and17 scripts. All496 bounded/442 control inputs pass. No production/native edit,
build,failure or repeated qualification occurs.

## Full launch and startup

The separately selected200k learner launches15:17:59 UTC November1/source6089e8b
with524 pinned inputs. MainPID809245/invocation
`361f29163c1e4b2896fd352f2b905ca8`;original progress,recovery,postprocess and
collection observers are active. Startup passes15:18:25:357 control/bounded-
matching initial tensors,47 equal pre-update transitions,correct R24 cadence
and314 finite supervised reports through3,728 actions. The1.558s CPU startup
wrapper co-hosts training;retain its time without correction. No frozen result
is available yet. Require retained100k/9,340 recovery and the fixed four arms.
Do not restart this learner,repeat passed checks or allocate another GPU job.

Full declarationSHA256`03a3ac6234dd23eee005c32c78bbb50af0903a00d1ee84924357931d10fc0650`.

The CPU-only [Breakout takeover-state join](2026-11-01-cdp-breakout-takeover-states.md)
co-hosts selection1.507s,analysis2.816s and independent recount2.581s wrappers.
It preserves all524 learner inputs and original process identity,withzero GPU
work or updates. Keep these times as co-host caveats without correction.

## Retained midpoint recovery

The original observer succeeds15:28:35 UTC November1:100,000 actions/9,340
updates,357 finite tensors and9,514 finite reports through101,864 complete-prefix
actions. All four recovery files are retained at`recovery-100k`,179,536,497 bytes,
with copied/source hashes and unchanged source. Retention takes.265s CPU;keep
co-host time without correction. A read-only hash check confirms all four copies.
Weights/metadata omit replay,live belief,games and RNG. The original learner
PID809245/invocation361f29163c1e4b2896fd352f2b905ca8 continues;progress,postprocess
and collection remain active. No frozen result or new allocation follows.

Recovery reviewSHA256`5eef8dfa59b5b3fd6a2d78cb31cd98d9f21617e2f0fd2cd976749a04face17fd`;
retentionSHA256`11e11b50c823cbf04143fca9f5cadd997250e58067ae69bafc758aa908fe1995`.

The unselected [Breakout B8/T32 preparation](../experiments/2026-11-01-cdp-breakout-context32.md)
co-hosts a3.201s CPU preflight wrapper at15:33:44 UTC. Both CLI paths,15 config
cases,357 schema tensors,227 control inputs andfour common-prefix fixtures pass;
all524 current inputs/original identity remain unchanged. No new GPU job is
selected. Keep this co-host timing without correction.
