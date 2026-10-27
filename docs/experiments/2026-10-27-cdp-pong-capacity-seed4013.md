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


## Active startup

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
