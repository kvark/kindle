# Bounded Breakout check of three-step world supervision

Select a fresh execution/cost check of the existing optional H3/scale1 world
loss on the retained Breakout recipe. The [actor-only imitation fade](../results/2026-11-03-cdp-breakout-actor-only-decay.md)
regresses both frozen modes despite exact pre-fade saved evidence. Keep actor
and world imitation at100. Earlier short action-ranking diagnostics leave
longer-term prediction and survival unresolved. This test addresses a different
part of learning; it does not assume world error is the causal bottleneck.

The [existing implementation and numerical checks](2026-10-29-cdp-open-loop-supervision.md)
are retained. Its Qbert full comparison lowered gameplay and aggregate forecast
quality; no general benefit is established. No Breakout learner with this
objective has been completed. A bounded check can qualify execution/cost only.

Use `open_loop_supervision={horizon:3,scale:1}`. One posterior start at local
time0 per replay sequence rolls forward using recorded actions and sampled
prior latent states. Future observations do not enter the rollout. Detached
centered CNN targets at steps2–3 receive cosine loss; reward two-hot CE and
continuation BCE use steps1–3. Existing component weights500/1/1 and global
scale1 apply. Masks exclude reset crossings and invalid tails, retaining valid
terminal arrivals. Original teacher-forced and expert losses stay intact.
No new parameters, inference inputs or actor/critic objective are added.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16, base/encoder4e-5,
behavior8e-5,dynamics4e-4,policy1,imagined-value1/replay-value1,lambda.95/discount333,
expert100/unit weights/world gradients/all-state corrective labels, actor/critic
representation gradients off, sampled collection/posteriors,unimix.01/entropy.0003,
action-effects1,replay100k,warmup1000,20% guided episodes/cap1920,seed1009,
full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64 resize/raw rewards.
No actor decay,crop or centroid loss. Nativec2d627f3/SHA0015dff7 and the matching
runner/runtime from `/x/Temp/kindle-actor-only-decay-20261103` remain unchanged.
They differ from retained447fe50; the current explicit imagined-bootstrap
default is the only legacy config normalization. Teacher stays byte-identical.

Reuse completed numerical validation and disabled-mode compatibility without
repeating them. Verify the open-loop module against its qualifiedab80486 source,
the current packaged source/runtime hashes, the original guard receipts and
the completed357-tensor/128-event legacy capture. These checks do not establish
broad runtime equivalence. No native rebuild or production edit is selected.

Run one fresh8,192-action/489-update training check, then1,024 greedy frozen
actions at evaluation seed4000001009. Require first update376,residual credit.5,
357 initial tensors and47 pre-update events matching control,exact config,
finite auxiliary reports with weighted total `500*prediction+reward+continuation`,
valid fractions in[0,1],finite imitation metrics,changed CNN/reward/actor/value,
357 invariant frozen tensors,zero frozen updates,full training pixel/action/label
replay and host guards. Do not treat short-run loss or natural episode counts
as task competence. No prior weights,replay,belief,RNG,games or experience transfer.

Fix120s training,60s frozen,180s CPU audit and480s service caps. The conservative
projection is `training_seconds*200000/8192*1.10`; require at most30 minutes
before considering a full comparison. A full learner is not automatically
selected by passing this check. Any later comparison needs its own declaration,
fixed final/actual-initial sampled/greedy24-episode assessments against70.750/
143.667 and the unchanged90% two-wall gate. No extension,coefficient sweep or
automatic successor is selected here.

Before launch, exercise both actual CLI paths without constructing a native
agent/emulator; test actual config and auxiliary-audit negative cases, schema,
source resolution and controller wiring. Pin the clean source, protocol,
runtime, scripts and retained inputs. Check live state immediately before
launch; preserve one original service identity and all failures. Repair a CPU
failure separately without repeating successful GPU stages. Collect terminal
evidence once after authoritative service completion.

Root `/x/Temp/kindle-breakout-open-loop-20261103/native-check`; service
`kindle-breakout-open-loop-check-20261103`. The five-game goal remains unfinished.
