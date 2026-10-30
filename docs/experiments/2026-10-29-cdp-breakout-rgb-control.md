# Bounded reconstruction control for the retained Breakout recipe

Completed October29:8192/489 takes59.857s,frozen1024/.866s,with all execution
checks passing after a separate CPU-only optimizer-name schema repair. The
original failed service is retained and no GPU stage repeats. Fixed projection
26m47s passes35min. See the [complete result](../results/2026-10-29-cdp-breakout-rgb-control.md)
and separately selected [fresh200k comparison](2026-10-29-cdp-breakout-rgb-control-200k.md).

Breakout's trained features carry useful position/motion information, but fixed
linear and RBF action readouts trail its policy. Those diagnostics do not identify
the cause of weak control. The retained matched Qbert RGB control does not answer
whether reconstruction helps Breakout's small moving objects. Prepare one
bounded control using the existing native decoder, without a native rebuild.

Reuse the completed12M64/B16/R16/expert100 CDP200k learner:22m19s training,
sampled13.208 and greedy17.875,zero of24 two-wall successes in both modes.
Its original source/config/checkpoints/evaluation remain fixed; do not retrain
CDP. Keep12M64/Auto/N8/B16/T16/H15/R16,full18/actor-unimix.01,action-effects1,
base/actor4e-5,encoder6e-6,dynamics4e-4,warmup1000,replay100k,seed1009 and
evaluation4000001009. Keep20% episode overrides/cap1920/expert100/all18 weights1/
world gradients/all-state corrective labels,edge-reset teacher with actual-action
feedback,native max-pooled frames and one GPU resize,sticky.25/repeat4.

Change only centered true→false,reconstruction loss0→1 and future loss500→0.
The copied experimental runner retains split rates and exploration; stock RGB
CLI alone would change them. Preserve its Breakout corrective-label path.
Native447fe50/runtime`/x/Temp/kindle-breakout-corrective-20261028/runtime` stays
unchanged. Decoder and predictor heads differ: expect399 tensors and14,457,556
trainable parameters versus357/16,334,353. Require342 shared initial tensors
to match and retain the disjoint head shapes. There is no equal-parameter claim.

## Selected work and bounded gate

Prepare on CPU while Qbert6029 runs. Check its original PID/invocation and263
pins before/after CPU preparation; record co-hosting as a timing caveat. Do not
launch GPU work until its main/observers/postprocessing are terminal and its
completion review passes. Recheck actual live state and host guard at launch.

The next allocation is limited to a fresh8192-action/489-update check,first
update376,residual credit.5,under180s; then1024 greedy frozen actions under60s,
zero updates. Use120s inner/150s outer replay,180s CPU audit and480s service caps.
Require exact config/counters/source,399 finite initial/final and invariant
frozen tensors,342 shared initial tensors,positive reconstruction/zero future
loss,changed CNN/reward/decoder/actor/value,aligned supervised losses/weights,
full training pixel/action/corrective-label replay and host guards. No natural
episode competence is inferred from the bounded evaluation.

CPU preflight must exercise both actual CLI paths before construction,invalid
mode rejection,the actual representation/config assertions,derived decoder
geometry and update accounting. Pin source,protocol,controls,runtime and queue.
No broad backend qualification,driver recovery,retry or check-experience reuse.
Fixed conservative full-cost gate:

`check_training_seconds * 200000 / 8192 * 1.10 <= 2100`.

## Potential full comparison, not allocated

Only after execution and cost pass, separately select one fresh200k-action/
12,477-update life with a35-minute training cap and100k/6227 recovery. Compare
fixed final and own actual initial in sampled and greedy modes;sampled primary
matches the retained control protocol. Each arm takes first3 natural episodes
per8 lanes=24,600k-action/300s caps,zero updates/truncations,guidance/labels off,
399 invariant tensors,complete outcome replays/videos and all usual audits.
Keep the original90%-two-wall gate (864 points); do not replace it with teacher
agreement,finite losses or any score improvement. Report both modes,own-initial
gains,frames and measured wall time separately. No checkpoint/mode selection,
extension,seed sweep or automatic successor follows this comparison.

The known CDP outcome and reused1009/cohort make this development evidence,
not independent confirmation. Reconstruction also changes the objective and
head geometry; even a positive result would not prove lost ball information was
the mechanism. No CDP superiority or five-game completion is inferred.

Root`/x/Temp/kindle-breakout-rgb-control-20261029/native-check`,trial
`breakout-rgb12m-64-b16-r16-expert100-check`,service
`kindle-breakout-rgb-control-check-20261029`. CPU preparation does not allocate
a full learner or enqueue GPU work behind the live Qbert process.

## CPU preparation passes; GPU check pending

Preflight completes17:29:20 UTC October29 in1.398s. Seven actual CLI/config
branch cases and11 actual audit-config acceptance/rejection cases pass; six
scripts compile. Both commands stop before output/environment/native
construction. Capability metadata is stubbed only for these CPU command checks;
no native module or agent is loaded. Breakout corrective-label flags,expert
settings,teacher feedback and fixed update accounting remain intact.

All132 retained control pins verify. Decoder geometry from native447fe50 gives
399 expected tensors/14,457,556 trainable parameters;342 shared tensor shapes
match the357 finite control initial tensors. Shared values and actual native
schema remain requirements of the pending GPU check. The runner changes only
the matched reconstruction option and its necessary validation/config branches;
the production runner and native binary stay unchanged.

All263 active Qbert6029 inputs and original PID318957/invocation
`b53406373693476ca9c08f5a5d05cc0a` remain unchanged before/after preparation.
Retain the17:29:19–17:29:20 CPU co-hosting as a timing caveat without correcting
Qbert wall time. No preparation failure occurs. The bounded queue and controllers
are ready, but no declaration/launch/full learner exists. Selection checks require
Qbert's completed review and an idle host; no GPU job is enqueued automatically.

## Bounded completion and CPU-only repair

The service starts17:55:59 UTC October29/source2261a7a after Qbert's completed
audits. Training8192 actions/32,726 frames/489 updates takes59.857s;1024 greedy
frozen/4096 frames takes.866s,zero updates/no natural episodes. Both GPU stages
and host guards pass. At17:57:08 the original service fails its CPU validator:
38 generated decoder optimizer names have an extra`world.` prefix. Native
weights,shapes and parameter counts are unchanged.

The separate`native-check/cpu-repair` corrects those names against the existing
RGB schema and verifies399 finite actual initial tensors. The original audit
then passes all config/counter/finite/loss/parameter-change/342 shared initial/
399 frozen invariance/pixel-action-label replay/guard checks in7.138s,with no
native import. All156 original inputs and176 repair pins remain unchanged.
Original service,journal,schema,preflight and controller remain preserved.
CPU repair succeeds17:59:54;terminal collection18:01:33. No GPU stage repeats.

The conservative projection1607.487s passes2100s. Actual overrides3898 (47.583%)
and489 supervised reports are execution evidence only. The separately selected
full comparison uses the corrected schema/helper and no check experience.
Completion`bd43647a5445ccda18e9634f9f5c3ec01a1bc1462a3ac087d5f1537945d565f1`.
