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

## Original launch and fixed readers — November 3

The declared comparison launches once at **13:20:44.166 UTC**, source
`55f0a46a5af7a5fda099f8a8d47540fd1848352d`, with667 pins. Declaration SHA256:
`32cd53f38f4ccab9dbd13338c0d8c020879b96dfa5583ab22a8b7ebd79100a48`.
The private fixed protocol is unchanged. No training,evaluation or successful
qualification stage repeats.

CPU preflight passes at13:19:20.990 UTC in2.510s body/2.667s wrapper, with zero
native/emulator construction. It checks five CLI paths,50 config assertions,
14 launch-command cases including wrong training/frozen update counts,23 header
cases,four observer-parser cases,14 recovery-body/four metadata cases,eight
prefix-timing cases,six retention/two collector-corruption cases,43 embedded
paths and19 compiled scripts. All601 bounded and338 retained-control pins pass.
Source generation,selection and launch succeed without failure. Selection takes
6.420s and the single five-service launch takes9.074s.

| Original service suffix | PID | Invocation |
| --- | ---: | --- |
| 200k |1029287|`da2979f074bf4fe4b740d734ad33f6cd`|
| postprocess |1029290|`8fa92dbb5aef47a9aee9e1e092e09921`|
| progress |1029293|`5a3c98ba26884f81819f6fb70a78f180`|
| recovery |1029435|`ecca019c954b4a588714594f1cfce4de`|
| collection |1029438|`1145552d80eb4ebbbef91cc6af91c73b`|

Service names are `kindle-breakout-context9-{suffix}-20261103`. Preserve these
original identities; do not restart or overlap GPU work.

Startup reading passes once at **13:21:42.199 UTC**, in1.866s wrapper. All667
pins,357 initial tensors matching bounded/control,47 common pre-update events,
first updates440/456/472/488/504/520 and503 finite supervised reports through
8,488 actions pass. Prefix timings are positive. The8,192-action progress record
has485 updates,50.380s elapsed and residual credit.5. No frozen task result exists.

Recovery and terminal readers are fixed **13:21:07.426 UTC**, before recovery
or frozen outcomes. `readers-provenance.json` pins their exact sources and
original process. Run `review_recovery.py` once after the original100k observer
succeeds while the same main remains active. Run `review_full_terminal.py terminal`
once after all original services and completion evidence finish. Its preflight
passes once at **13:21:31.180 UTC**, in3.171s body/3.218s wrapper, with five score
cases,all667 pins and original identity. No native learner or emulator is built.
Keep these CPU co-host costs without time correction. Source hashes and execution
receipts remain under the experiment root. The five-game goal is unfinished.

## Halfway recovery — November 3

The original recovery observer succeeds **13:32:50.987435 UTC**, retaining the
100,000-action/6,223-update checkpoint while the learner continues. The fixed
independent reader runs once and passes **13:33:51.401802 UTC**, in2.246s body/
2.317s wrapper. All667 pins,357 finite tensors and all four files totaling
206,067,205 bytes pass. The observer's complete log prefix contains103,000 actions
and6,410 finite reports. MainPID1029287/invocationda2979f074bf4fe4b740d734ad33f6cd
remains active; recovery retains its original invocationecca019c954b4a588714594f1cfce4de.

- Observer review: `90880718ae7f019c3cf9ec3f6d9123fd013dfee27210d115f6bd1adef9a9cf4a`.
- Retention: `84fb0b9a78ab630d53f97f7be19572123952981061433a8f98ec2dfc2b99e376`.
- Independent confirmation: `3b392be208b7dc890b8aba186e64b4058aaf67e8e1154c7de665c7c32abfb93d`.
- Metadata: `56da6bf2760a7a40825dd42699b10e628d1f09a3102fa491c4d0284d3896ac49`.

The checkpoint retains weights and metadata; replay,RNG,games and live belief
are not included. No restore,learning,evaluation,observer or successful-reader
stage repeats. Keep CPU co-host time without correction. Recovery reading is
closed. Terminal reading and frozen gameplay remain pending; no successor is
selected and the five-game goal remains unfinished.

## Completed comparison — November 3

The [full result](../results/2026-11-03-cdp-breakout-context9.md) completes200,000
actions/799,606 frames/12,473 updates in **23m30.959s (+9.55%)**. Sampled
**46.583** and greedy **91.167** fall below retained70.750/143.667, all0/24
two-wall. Initial-subtracted return/min falls40.54%/42.60%. Own initials1.417/
2.208 and800/795 semantic events match. Retain context1; no extension,sweep,
confirmation or automatic successor is selected. The five-game goal is unfinished.

All667 pins,100k/6,223 recovery,357 matching initial/invariant frozen tensors,
finite reports,positive prefix timing,full replays/videos,guards,curves,exposure
and comparison pass. Prefix totals101.886s/8.169ms per update. Late CE.559628
and44.342% actual overrides do not establish unaided competence or state accuracy.
All five original services succeed, with main/post/collection terminal at
13:48:19.732/13:48:49.527/13:48:54.418 UTC. Original mainPID1029287 and invocation
da2979f074bf4fe4b740d734ad33f6cd remain the only learner identity.

The fixed independent terminal reader passes once13:49:17.338 UTC in1.706s
body/1.766s wrapper. Summary SHA256:
`624700fe0d397d90f6b5021088364cf4ed9ddd891ba626349ca3fd66a02eaabc`.
Completion SHA256:
`2599a36ed6b57d3919e02a2a62cc58d71d725f621e40c7294dee1ed15bfa60f7`.
The private declaration/protocol and all original pins remain unchanged.

No full-run stage fails or repeats. An ad-hoc observation assertion rejects an
empty invocation after the successful recovery unit is unloaded; the fixed
reader verifies its original identity and success from the journal. No restart
or repair to the experiment follows that poll. Earlier bounded failures and
separate repairs remain preserved. CPU co-host times remain uncorrected;
sampling,warmup,work,native and runner differences prevent an isolated causal
claim. All stages and readers are closed; do not repeat them. No Kindle
allocation remains active at closure. Check live state before future GPU work.
