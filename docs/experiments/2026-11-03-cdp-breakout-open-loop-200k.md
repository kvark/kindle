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
