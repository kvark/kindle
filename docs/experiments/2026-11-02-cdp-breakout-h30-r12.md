# Breakout H30/R12 bounded execution and cost check

**Complete:**8,192 actions/367 updates in51.413s;1,024 frozen/.892s,zero updates/
natural episodes. All382 pins,tensors,traces,replay,guards andoriginal services
pass;terminal reader completes07:26:54.448 UTC November2. Conservative200k
projection23m00.725s passes30min. Do not repeat completed stages orreader.
[Result](../results/2026-11-02-cdp-breakout-h30-r12.md);a separately selected
[fresh200k comparison](2026-11-02-cdp-breakout-h30-r12-200k.md) reusesno experience.

Select one fresh bounded check after the lower-dynamics/value1 comparison
closes. Retain its stronger dynamics4e-4 control:70.750 sampled/143.667 greedy
in21m27.910s, withzero of24 two-wall successes. The lower-dynamics combination
regresses both modes and is not extended.

The hypothesis is that longer imagined rollouts can improve sustained control
at a similar budget. In the fixed15-step action-ranking diagnostic, only8/128
origins distinguished plans by actual reward; that test omitted critic bootstrap
and intrinsic rewards. Short H5/R24 worsened both Breakout frozen scores despite
better imitation fit. These results motivate checking a longer rollout, but do
not establish that horizon is the cause or that a learned30-step rollout is
accurate. The earlier compact Qbert H30 trial also regressed and cost more;
retain that negative evidence. No existing result is repeated.

## Recipe and scope

Change H15/R16 to **H30/R12**. At200k actions, world/expert updates fall from
12,477 to9,358; nominal imagined-transition work rises about50%. Temporal
extent, sampling, cadence, action-clock warmup and relative learning work change
together. No isolated horizon or reliability claim is possible.

Keep centeredCDP/12M64/Auto/N8/B16/T16/full BPTT/microbatch16, base/encoder4e-5,
behavior8e-5,dynamics4e-4,replay-value1 withworld gradients, sampled collection
and posteriors, actorunimix.01, replay100k, action-effects1 and expert100.
Keep unit imitation weights, world imitation gradients, all-state corrective
labels,20% guided episodes/cap1920,full18/sticky.25/repeat4/noops0,native pooled
RGB andone GPU resize64. Native447fe50/SHA35ad0db,exact control runnercf4ea238,
teacher and16,334,353 trainable parameters remain unchanged. No build.

Use development learner1009/evaluation4000001009 andoriginal lane offsets.
Begin with no retained weights,replay,belief,games,RNG or experience. No full
learner or independent confirmation is selected by this bounded protocol.

## Check and fixed limits

Train8,192 aggregate actions /367 updates,first update376,residual.375.
Derive100k/4,670 updates/residual.875 and200k/9,358/residual.375 onCPU.
Follow with1,024 greedy frozen actions, guidance/labels off,zero updates.
Require357 finite initial tensors matching control,47 matching pre-update
semantic events,finite aligned learner reports,changed CNN/reward/actor/value,
full training pixel/action/teacher-label replay,357 invariant frozen tensors
andhost guards. This short frozen budget does not measure task competence.

Caps:180s training,60s frozen,180s CPU audit,480s main service. Preserve the
existing bounded collector deadline. Conservative cost gate:
training seconds ×200000/8192 ×1.10 ≤1800s. A pass permits a separate decision
onone fresh200k comparison; it does not launch one automatically. Do not reuse
the check's experience. The original sampled-primary comparison and90%-two-wall
gate remain unchanged for any separately selected full learner.

Before selection, validate both actual commands,configuration rejection cases,
cadence,checkpoint schema,source identities andpreceding terminal evidence on
CPU without native/emulator construction. Require the host idle at launch.
Pin the exact runner/native/teacher andall inputs. Preserve failures separately;
no retry,cap increase,horizon/ratio sweep,extension,automatic promotion or
GPU recovery. Never overlap another active allocation.

Root `/x/Temp/kindle-breakout-h30-r12-20261102/native-check`,trial
`breakout-12m-64-b16-h30-r12-replayvalue1-expert100-check`,service
`kindle-breakout-h30-r12-check-20261102`.

## CPU qualification

Preparation completes07:24:07.354 UTC November2/sourcea5a0246 in1.041s
(1.075s wrapper), withtwo actual commands,20 configuration cases,357 checkpoint
schema tensors andall338 control pins passing. Exact cadence derives367/4,670/
9,358 updates and.375/.875/.375 residuals at8,192/100k/200k actions. Native and
emulator construction attempts arezero. Seven scripts prepare in.028s without
failure; the runner stays byte-identical cf4ea238 andprior CLI qualification
isreused. No native build orsuccessful-stage repeat. Do not rerun preparation
orpreflight; selection andthe sole bounded execution remain next.
