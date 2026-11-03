# Fresh200k Breakout comparison with replay-state recomputation

Select one fresh learner after the [context9 execution check](../results/2026-11-03-cdp-replay-context.md)
completes8,192 actions/485 updates in50.564s, with22m37.921s conservative200k
projection. All601 original/631 repair pins,357 matched initial/frozen tensors,
47 common pre-update and128 legacy events,replay and guards pass. The original
queue and CPU audit failures remain; only the previously unstarted frozen stage
continued, with zero training/frozen/replay repeats. Reuse that qualification.

Change `replay_context` from1 to9: one cached posterior, eight observed steps
recomputed with current parameters and detached, then16 learning steps. Prefix
rows supply no loss targets and do not refresh replay caches. The independent
prefix RNG and current world/encoder inference add no trainable parameters.
Sampling eligibility,fresh-window stride,warmup and recurrence work change
together. Cache age is not latent error; this comparison cannot isolate age as
the cause of weak control. The previous replay16k and B8/T32 results stand.

Keep centeredCDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16, base/encoder4e-5,
behavior8e-5,dynamics4e-4,policy1,imagined-value1/replay-value1,lambda.95/discount333,
expert100/unit weights/world gradients/all-state corrective labels, actor/critic
representation gradients off, sampled collection/posteriors,unimix.01/entropy.0003,
action-effects1,replay100k,warmup1000,20% guided episodes/cap1920,seed1009,
evaluation4000001009,full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64
resize/raw rewards. H3,centroid,crop and imitation decay remain off. Teacher fixed.

Use the qualified native65dd69b4/SHA7b15cb69 at
`/x/Temp/kindle-replay-context-20261103/runtime` and the exact bounded runner.
They differ from retained447fe50/cf4ea238. Normalize only the explicit default
`replay_value_bootstrap=imagined`; initial/legacy agreement is not broad runtime
equivalence or an isolated causal comparison. Reused seed/cohort is development.

Fresh training reuses no weights,replay,belief,RNG,games or bounded experience.
Stop at200,000 actions/12,473 updates or1800s. Preserve actual initial and
100k/6,223 recovery checkpoints, all four files/357 finite tensors. First update
is440 and residual credit.5; the control has12,477 updates. Require positive
prefix timings,finite reports,constant imitation,aligned raw actor/world expert
CE,changed CNN/reward/actor/value and exact cadence. Report prefix time separately.

Freeze final and actual initial, each sampled and greedy, first three natural
episodes per eight lanes=24 each. Guidance/labels off; zero updates,truncations
and357 invariant tensors. Keep300s/600k actions per arm. Sampled is primary;
report both modes and initial-subtracted return per training minute against
retained70.750/143.667 and own initials1.417/2.208, all0/24 two-wall successes.
The90% two-wall864 task gate is unchanged. No checkpoint/mode selection,retry,
episode-cap increase or outcome-conditioned extension.

Require357 initial tensors matching bounded/control,47 common pre-update events,
both initial frozen semantic traces,full training pixel/action/label replay,
frozen action/outcome/frame/boundary replays/videos,host guards,curves,exposure
and comparison. Ordinary frozen logs omit live pixel hashes. Online guided
returns are not competence. Preserve CPU co-host timing without correction.

Before launch, qualify the actual five CLI paths, queue expected update count,
audit/comparison config assertions,prefix timing checks,startup cadence,observer
wiring,recovery schema and retention. Pin clean source/runtime/scripts/inputs
and this protocol. Preserve one original main/progress/recovery/postprocessing/
collection service, with4500s main,1200s audit,5100s post/progress,3000s recovery
and5400s collection caps. Use invocation-based journals; a wrapper need not emit
its own PID-tagged log. Independently read startup,recovery and terminal evidence
once; fix additional reader sources before their outcomes are available.

Root `/x/Temp/kindle-replay-context-20261103/full`; service
`kindle-breakout-context9-200k-20261103`. Check live state before GPU work. No
overlap,driver recovery,cap increase or successful stage repeat. Preserve
failures and repair CPU processing separately. No extension,confirmation,
sweep or automatic successor is selected. The five-game goal remains unfinished.
