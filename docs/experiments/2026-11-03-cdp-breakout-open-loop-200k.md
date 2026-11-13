# Fresh200k Breakout comparison with three-step world supervision

Select one fresh comparison after the [bounded H3/scale1 check](../results/2026-11-03-cdp-breakout-open-loop.md)
passes8,192 actions/489 updates in48.603s, with21m45.264s conservative200k cost.
The original service succeeds; its separate CPU collection repair preserves the
failed parent-PID assertion and all602 original pins. No GPU stage repeats.
Execution and cost are qualified; gameplay benefit remains unmeasured.

The intended change from the retained70.750 sampled/143.667 greedy control is
`open_loop_supervision={horizon:3,scale:1}`. Keep actor/world imitation100.
The [bounded protocol](2026-11-03-cdp-breakout-open-loop.md) fixes rollout inputs,
detached targets,500/1/1 component weights, masks and gradient boundaries.
The earlier Qbert H3 comparison regressed; this game-specific test does not
assume that a larger predictive objective improves decisions.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16, base/encoder4e-5,
behavior8e-5,dynamics4e-4,policy1,imagined-value1/replay-value1,lambda.95/discount333,
expert unit weights/world gradients/all-state corrective labels, actor/critic
representation gradients off, sampled collection/posteriors,unimix.01/entropy.0003,
action-effects1,replay100k,warmup1000,20% guided episodes/cap1920,seed1009,
evaluation4000001009,full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64
resize/raw rewards. No imitation decay,crop or centroid loss. No new parameters
or inference inputs are added. Teacher and all other learner settings stay fixed.

Reuse nativec2d627f3/SHA0015dff7 and runner8946f5f3 from the bounded check.
Both differ from retained447fe50/cf4ea238; normalize only the explicit current
`replay_value_bootstrap=imagined` default when comparing configs. Matching
initial values and traces do not establish broad runtime equivalence or an
isolated causal effect. Seed/cohort reuse is development, not confirmation.

Fresh training reuses no weights,replay,belief,RNG,games or experience. Stop at
200,000 actions/12,477 updates or1800s. Preserve the actual initial checkpoint
and100k/6,227 recovery, all four files and357 finite tensors. Require first
update376,residual credit.5,finite reports,declared weighted auxiliary reductions
and valid fractions,constant imitation,raw world/actor expert CE agreement and
changed CNN/reward/actor/value. Report actual overrides separately from competence.

Fixed frozen arms are final and actual initial, each sampled and greedy, first
three natural episodes per eight lanes=24 each. Guidance/labels off; zero updates,
zero truncations and357 invariant tensors. Keep300s/600k actions per arm.
Sampled remains primary; report both modes and initial-subtracted score per
training minute. Compare against70.750/143.667 and own initials1.417/2.208,
all0/24 two-wall. The90% two-wall864 gate is unchanged. No checkpoint or mode
selection, retry, episode-cap increase or outcome-conditioned extension.

Require357 initial tensors matching bounded check/control,47 pre-update events,
both initial frozen semantic traces,full training pixel/action/label replay,
frozen action/outcome/frame/boundary replays and videos,host guards,curves,
exposure and comparison. Ordinary frozen logs omit live pixel hashes; do not
claim unavailable image equality. Keep CPU co-host timing without correction.

Prepare and qualify the actual five CLI paths, audit/comparison config and
auxiliary-report assertions, startup checks, observer wiring, recovery schema
and retention before launch. Reuse passing numerical/legacy qualification;
do not repeat the bounded run. Pin source/runtime/scripts/inputs and this
protocol from a clean commit. Retain one original main/progress/recovery/
postprocessing/collection service, with4500s main,1200s audit,5100s post/progress,
3000s recovery and5400s collection caps. Use the qualified invocation-based
journal checker; a wrapper need not emit its own PID-tagged log message.
Preserve failures and repair CPU processing separately without repeating
successful GPU stages. Independently collect startup,recovery and terminal
evidence once. Declare any additional saved-data reader before reading outcomes.

This selects full-comparison preparation; no service is launched by the protocol
alone. Check live state before launch and do not overlap GPU work. Root
`/x/Temp/kindle-breakout-open-loop-20261103/full`; planned main service
`kindle-breakout-open-loop-200k-20261103`. No extension,coefficient sweep,
confirmation learner or automatic successor is selected. The five-game goal
remains unfinished.

## Original launch and fixed readers — November 3

The declared comparison launches once at **11:50:15.734933 UTC**, source
`ffc4a26dc8d46d3138cbefb2c8f359b4e68fd746`. The private fixed protocol remains
unchanged. All644 declaration pins are retained; declaration SHA256 is
`0bce56751ed8248c09de1cf643a0d3c34ffe0c09008161b8d1257977e4e0366a`.
No training, evaluation or other successful stage repeats.

The CPU preflight completes **11:49:51.764 UTC**, in2.457s body/2.570s wrapper,
with no failures and no native learner or emulator constructed. It checks the
actual five CLI paths,48 config checks,15 launch-command cases,22 header cases,
four observer-parser cases,14 recovery-body/four metadata cases,eight auxiliary
report cases,six retention cases,two collector-corruption cases,44 embedded
paths and19 compiled scripts. All602 bounded-check/338 control pins pass.
Source generation and selection also succeed; selection takes3.300s and the
single five-service launch takes8.503s. Receipts live at the experiment root;
`full/full-preflight.json` records the detailed qualification.

| Original service suffix | PID | Invocation |
| --- | ---: | --- |
| 200k | 1020432 | `7c92d99e93bb4780a04e64d685c6d714` |
| postprocess | 1020435 | `ef5a0ce85b314166a70aa69cb1027719` |
| progress | 1020438 | `ae170a5aa07c453dabfd2f43b7270e80` |
| recovery | 1020574 | `bdc3674c493a407a866a42b64e56c11c` |
| collection | 1020577 | `63ce119ff1b44ea8a5312e55050f6fea` |

Each service is named `kindle-breakout-open-loop-{suffix}-20261103`. These
identities remain unchanged at11:55 UTC. Do not restart them or overlap GPU work.

Startup reading completes once **11:51:07.120 UTC**, in1.495s wrapper:
all644 pins,357 initial tensors matching both bounded check and retained
control,47 pre-update transitions,correct first updates at376/392/408/424/440/
456 and442 finite supervised reports through7,448 actions pass. Auxiliary
weighted-loss checks pass. The4,096-action progress record has233 updates and
residual credit.5. Startup-review SHA256 is
`6f7e843618c349cb49e13df1f34bcbad9976157e943317e31fb8eac37650aa2b`.
This verifies startup only; no full frozen result exists at this review.

Recovery and terminal readers are fixed at **11:51:52.423 UTC**, before the
recovery or frozen outcomes. `readers-provenance.json` pins their sources and
original identity. `review_recovery.py` SHA256 is
`d28902f172e3a25ac106d70bf1fa1a12bbc26b50c4c8978865ae562609e7d9f2`;
run it once after the original100k observer succeeds while the same learner
remains active. `review_full_terminal.py` SHA256 is
`d83f3a0e21285661499a6ff2552529c8d2b30b63047e7dfba5596b5c2090d54a`.
Its preflight completes once **11:51:54.856 UTC**, in2.402s body/2.422s wrapper,
with five score-tamper cases,all644 pins and original process identity passing.
Preflight SHA256 is
`e202f869731033f15da78b5efe4c1adaf47f59a67b6ff1c77177728578531736`.
The reader uses the qualified invocation-based journal helper; it preserves
the earlier bounded collector failure and separate CPU repair. Terminal mode
remains pending until all five original services and completion evidence finish.
Keep these CPU co-host times without correction. No gameplay conclusion,
extension or automatic successor follows from passing preparation/startup.

## Halfway recovery — November 3

The original recovery observer succeeds **12:01:22.587269 UTC**, without stopping
the learner. The fixed independent reader runs once and passes
**12:01:38.194060 UTC**, in1.300s body/1.319s wrapper. It verifies all644 pins,
100,000-action/6,227-update metadata and357 finite tensors, and hashes all four
retained files totaling206,067,280 bytes. The audited complete training prefix
contains100,648 actions and6,268 finite reports. Both the observer and learner
retain their original invocation identities; the main remains active.

`checkpoint-review-100k.json` SHA256 is
`48553be090adca31b6c84a252313e64be42166ea328c4483948738fff185cce2`;
`recovery-retention-100k.json` SHA256 is
`84997973b10d4493c1069995665be889a5601f61c14c92e8313af245bd8f1239`.
`recovery-confirmation.json` SHA256 is
`75873da952bfa52877464aa94ea496fd9aac60b0373d3f9b6ecb5c60de989813`.
The metadata SHA256 is
`c8c0798398b7669f70af83da601016ec211dd2ef6231ab031295798492949962`.
The retained checkpoint contains weights and metadata; replay,RNG,games and
live belief are not included. No restore,learning,evaluation or observer stage
is repeated. Keep the CPU co-host time without correction. Recovery reading is
closed; terminal reading and frozen gameplay remain pending.

## Original terminal result — November 3

The [completed comparison](../results/2026-11-03-cdp-breakout-open-loop.md)
finishes200,000 actions/799,552 frames/12,477 updates in22m06.821s,3.02% slower
than the control. Sampled70.750→60.875 and greedy143.667→104.625 both decline;
all final/initial arms havezero of24 two-wall successes. Own initials1.417/
2.208 match. Initial-subtracted score/min falls16.76%/29.72%. Retain the control
without H3, constant actor/world imitation100 and critic1/1.

All644 pins,recovery,357 matched initial/invariant frozen tensors,47 pre-update
and800/795 initial frozen events,all12,477 finite auxiliary reports,replays/
videos,guards,curves,exposure and comparison pass. All five original services
succeed; main/post/collection finish12:16:22.850/12:16:49.658/12:16:54.736 UTC.
The fixed terminal reader runs once and passes12:17:22.865 in1.404s wrapper;
no full-run failure,GPU stage,collector or reader repeats. Original bounded CPU
collection failure and separate repair remain preserved. No time correction.

Completion SHA256 is
`2f8503990218a6e6604571eb544cdee1fa429b45785f860823ea6ffa2d44474b`;
independent terminal summary SHA256 is
`3c7c8d75bb65d3a3a41fc37bf0d39cbd3823be9116eebc5761ec17076647bbf0`.
The original learner,observers and all readers are closed; do not repeat them.
No allocation remains active at closure. No extension,sweep,confirmation or
automatic successor is selected. The five-game goal remains unfinished.
