# Pong12M48 fixed-recipe confirmation on4013

Select one fresh learner after the completed1009 result scores15.625 with24/24
wins and passes all audits. Its500k actions /31,227 updates take41m28s,16.95%
less training time than the strongest compact4M/B8 control. The original gate
passes on one learner; the next question is whether this recipe also works on
4013. See the [primary result](../results/2026-10-25-cdp-cheap-pong.md#larger-capacity-with-less-replay-work-reaches-the-gate).

Keep centered CDP12M/RGB48/Auto/N8/B16/T16/H15/R16, microbatch16, replay100k,
coefficient.01, actor mixture0, original rates and disabled actor/critic gradients.
Full18/sticky.25/repeat4/noops0,100000-frame episode limit, one GPU resize from
native max-pooled frames, no external aid or expert imitation. Actual trainable
parameters14,123,473. Native/CLI79c9a7f and all Python sources stay unchanged.
Only training/environment/RNG seed1009→4013 and evaluation4000001009→4000004013
change.4013 and its evaluation cohort were used in earlier4M development;
this is a fresh fixed-recipe learner, not a held-out test or broad seed campaign.

The fixed budget remains **500k actions /31,227 updates**, first376, residual
credit.5 and initial six updates376/392/408/424/440/456. Training cap **45 minutes**
includes construction. Save/audit100k,200k,300k,400k recoveries at6227,12477,18727,
24977 updates; fixed final31227. Start with new parameters, replay, environments,
RNG and belief. Do not restore1009 or reuse any bounded-check experience.

Fixed final and own actual initial each collect the first three natural sampled
matches per eight streams:24 selected matches, original mean15/90%-wins gate,
zero learning/truncations, all346 frozen tensors invariant. Each arm has600k-action/
300-second caps. Complete recorded-action replay checks rewards, frame counts,
boundaries and outcomes, with reconstructed videos. Ordinary unguided logs do
not contain live pixel hashes, so direct live/replay pixel equality is untested.
Every learning report, config/source/counter, finite checkpoint and host guard
must pass. Startup checks the exact recipe except seed, zero initial counters,
finite346-tensor initial checkpoint and first six updates; different seeds mean
no cross-seed tensor or prelearning-trajectory equality claim.

Retain primary1009 and paired4M1009/4013 controls in the final cost/quality
comparison. Those compact controls score14.958/14.000 with24/24 wins, in49m55s/
49m58s, and both miss mean15. Comparisons across model recipes change capacity,
batch, ratio and older native/kernel together. The two12M learners share the
native/runner/configuration except seed. No checkpoint selection, retry, cap
increase, extension or automatic next allocation is permitted by this declaration.

Primary service is terminal01:51:06 UTC on October27 and CPU postprocessing
completes01:51:35. All42 original pins remain unchanged before preparation.
Three new commands parse without constructing an agent/environment; comparison
readers pass for the primary and both retained compact controls. The new
allocation pins53 files. The Breakout adapter remains an external eight-file
CPU-validated patch; its prepared native check is not selected or launched.

Artifacts:
`/x/Temp/kindle-pong-capacity-20261027/pong-12m-48-b16-r16-intrinsic001-500k-seed4013`.
Owner:`kindle-pong-12m48-b16-r16-500k-seed4013-20261027`.
Controller`run_seed4013.py`, preparation`prepare_seed4013.py`, startup
`check_startup_seed4013.py`, recovery`watch_recovery.py`, comparison
`compare_seeds.py` and completion`finish_seed4013.py` in the parent directory.
Audit cap1200s; main service cap4620s. Monitoring and CPU postprocessing do not
start, stop or extend GPU work. Declaration and launch/startup receipts retain
exact hashes and the launch commit.


## Recorded startup

Service starts01:57:51 UTC on October27 from`4cd7fb5f13294bc692e8b75dbdb341f8711f499e`,
PID71345, invocation`bfe98b4e4ae84b3f996e05f86f28e3df`. Startup passes at5656
complete actions/330 finite reports: zero starting counters/no restore, exact
recipe except4013,14,123,473 parameters, all346 initial tensors finite, fresh
world weights different from1009,47 prelearning transitions and first six
376/392/408/424/440/456 updates. All53 declared inputs remain unchanged. No
cross-seed tensor/trajectory equality is asserted. Startup receipt SHA256:
`49695c5e5676de8d5bd6949c0c1e44cbdf6d645435a1090bf1c95f5b67e49284`;
launch declaration`a0d84ffa794713df966e95a0c5de1cffe4257fe44e72f5846090d91d6ea44d96`.

The initial read-only progress observer exits2 because unsupported `--seconds`
and `--poll-seconds` flags were supplied. Its original journal is retained.
A separate-v2 observer uses the script's existing45-second/3600-second defaults
and runs correctly; the main learner keeps the same PID/invocation and no GPU
stage repeats. Recovery100k/200k/300k/400k and terminal CPU plot/comparison
observers are attached. This is startup evidence, not full-run competence.


## First recovery and independent CPU preparation

The100k/6227-update recovery passes02:06:20 UTC: all346 checkpoint tensors finite,
complete prefix102,264 actions/6368 finite reports, last progress98,304/6121 in
485.781s with residual credit.5. Receipt SHA256:
`7111674bd6490a202fa6e57e2011aff8d961ea7d0c12ae89accfbe5742405626`.
The original PID71345/invocationbfe98b4e4ae84b3f996e05f86f28e3df remains active.
No frozen confirmation score is known yet.

Independent CPU-only analysis of already saved Breakout trajectories supports
keeping full-weight NOOP braking labels for a proposed capped expert trial.
Prefix/parser successful portions total0.630s on one affined low-priority core;
no native agent or environment is constructed. External controllers and both
retained-control readers are prepared, but neither Breakout candidate is selected.
All53 active pins remain unchanged. See [candidate evidence](../results/2026-10-26-cdp-breakout-pixel-reference.md#capped-reference-labels-support-equal-action-weights).


The200k/12,477 recovery also passes02:14:50 UTC: all346 tensors finite,
complete prefix204,600 actions/12,764 finite reports, last progress200,704/
12,521 in996.722s and residual credit.5. Receipt SHA256:
`f28147195462ece6dad4ce2122406d2b25d14e18e73794b1565d184df1e35914`.
A separate one-episode Breakout diagnostic costs0.980s and one CPU reference
variant costs5.572s collection/3.899s replay. Its small152.750 versus144.875 gain
on eight reused episodes, both0/8 two-wall successes, does not replace the staged
teacher. No native agent or live input changes; all53 active pins remain intact.


## Completed result

The fixed final scores14.833/24 wins versus own initial−20.542/0. Mean15 fails;
only one of the two12M learners passes.500k actions /1,999,720 frames /31,227
updates take41m31s,16.90% less training time than the paired4M4013 control.
All53 input pins, finite reports, four recovery checkpoints, config/counters,
346 frozen tensors, replays, guards, curves and comparison pass. Service exits
02:41:08 UTC; postprocessing02:41:23. No extension or third seed follows.
See [complete evidence and hashes](../results/2026-10-25-cdp-cheap-pong.md#capacity-confirmation-improves4013-but-narrowly-misses-the-gate).
The original read-only observer failure remains retained. The later application
of the validated Breakout adapter follows terminal audits and does not change
this run's immutable declaration, artifacts or runtime.

## Prepared frozen action-selection comparison

The completed12M pair motivates a separately prepared four-arm greedy
assessment: fixed final and own actual initial for both1009/4013, no new
training. Preserve original sampled15.625/14.833 and their training costs.
The existing native argmax path and exact original Pong runner require no
source edit or rebuild. The older4M greedy test was negative; no global
default change or broad reliability claim is justified.

Keep full18/sticky.25/native one-resize RGB48, N8, original evaluation seeds,
first3 natural matches per lane=24 per arm,600k-action/300s caps, zero updates/
truncations, mean15/90%-win gate,346 invariant tensors and full replay/video/
guard/source audits. Order is final1009, initial1009, final4013, initial4013,
fixed before any result. Audit cap1200s, total service2460s. No retry, cap
increase, extension, checkpoint selection or automatic successor.

Candidate:`/x/Temp/kindle-pong-greedy-20261027/pong-12m-48-two-seed-greedy-frozen`;
owner if later selected:`kindle-pong-12m-two-seed-greedy-frozen-20261027`.
All58 prepared pins match. Four command parses, five compiled scripts, four
finite source checkpoints, exact sampled score recomputation and refusal while
unselected pass. All47 active Qbert inputs stay unchanged. The declaration
remains prepared/unselected and no service starts. A separate decision and
terminal Qbert/postprocessing receipt are required before GPU work. See
[prepared evidence and limits](../results/2026-10-25-cdp-cheap-pong.md#prepared-greedy-comparison-of-both-retained12m-learners).

## Paired greedy assessment selected after Qbert completion

Qbert R16/20% and all postprocessing finish successfully at05:23:55/05:24:19
UTC. Its complete receipt is SHA256
`eeccfd460ba6afe09618cfd21e250f24bcf44b294ab5c4c68cc84f8b8b6c0515`.
Select the prepared four-arm greedy assessment without changing its58 pins,
original runtime, seeds, fixed arm order, checkpoints, action/time caps or
mean15/90%-win gate. This tests whether greedy inference improves the retained
12M pair at no additional training cost. The earlier4M greedy regression remains
recorded; this is not a global default change. Both original sampled outcomes
and training costs remain. No retry, extension, cap increase, checkpoint
selection or automatic successor. The previous unselected preparation is
historical; this separate decision authorizes only the fixed assessment.

## Paired greedy startup

Service starts05:26:13 UTC/source`edb9ac8`, PID93214,
invocation`cb443e7d9f4e4c2484dd63e197d54051`. Selected declaration SHA256
`9a7c489a213fa7c58720f72921aafe9ea0c8c28677953fbc5ecc6850374130f9`.
All58 pins match; the first completed final/initial1009 headers identify greedy
mode, the exact original native/runner and RX7900XT. Both have zero updates.
The terminal observer starts05:26:20 UTC. An ad hoc read-only probe initially
assumed the first JSON row was run_start; the corrected reader filters by event.
No service mutation or GPU repeat follows that reader error. Fixed caps remain.

A separate.447s affined CPU source-identity receipt checks the optional expert
learning path without loading a native module or environment. All58 Pong pins
remain unchanged. Source-reading wall time is not instrumented; no timing
correction or new GPU work is inferred from this receipt.

## Paired greedy assessment completes

Both retained learners score15.208/24wins under greedy, versus sampled15.625
and14.833. Own greedy initials−20.583/−20.625 have0wins. Both pass the unchanged
mean15/90%-win gate, but only.208 above15 and both conditional uncertainty
intervals cross15. The original sampled outcomes remain. Reused development
cohorts and existing weights do not become new independent learners.

Four arms total202,768 actions/810,927 frames/111.916s, zero training/updates.
All58 pins,346 unchanged tensors per arm, source/config/counters, replays,
videos and guards pass. Main exits05:29:46 UTC after3m32.855s including audits;
postprocessing05:29:51. No cap increase or GPU repeat. Retain as a useful12M
option, with no global default change, extension, third learner or successor
allocation. See [complete result and receipts](../results/2026-10-25-cdp-cheap-pong.md#greedy-evaluation-passes-the-gate-on-both-retained12m-learners).
