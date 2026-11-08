# Breakout: local action-conditioned image prediction

The [transport experiment](../results/2026-11-08-breakout-world-transport.md)
keeps freeflight balls visible but fails precise long forecasts,observation
precision and event timing. It is closed. Select a separately justified local
control path,not an extension or repeat. The existing spatial branch receives two
causal images but actions reach it only through the global RSSM/decoder. Test a
direct submitted-action input to local prediction. This is a hypothesis,not proof
that the global state caused the errors.

Default-off `rgb_spatial_action` requires the ordinary fused categorical spatial
world,standalone `motion_event_rollout`,positive H,and no transport,residual,image
feedback or alternative latent architecture. Concatenate18 incoming one-hot action
planes after the six two-image RGB channels before motion conv0. Broadcast every
action scalar over the image,shared spatially,without object/RAM labels. Use zero
planes at observed reset arrivals. Widen all three motion ConvNorm layers8→16;
the fusion input grows accordingly,its output stays16. No other decoder,encoder,
RSSM,reward/continuation,actor/critic or backend primitive changes.

Observed reconstruction uses the action that arrived at that frame and only two
earlier images. In training use the already masked incoming action;in prediction
use the same `action_inputs` one-hot alignment,including reset zeros. Forecasts
use the submitted action for that transition and generated image history with
full BPTT. No future action enters an earlier output. True future RGB/events/reset
labels remain stopped targets/masks only,never state/history corrections. Event
heads remain RSSM-only;better images cannot replace event qualification.

At RGB64/12M/action18 this adds21,248 scalars to the v13 fusion model:
80 logical/8,893,203 scalars/240 checkpoint tensors. Ten weight shapes change;
70 weights/210 common initial tensors should match cached v13 exactly. The
16 nonimage initial arrays should match too. Keep feature2560/two-image history
24,576. Wider local capacity,direct action conditioning,initialization,work and
gradients change together;do not claim an isolated action-path benefit. Keep
ordinary named initialization and existing thirteen reports. Bind flag,geometry,
objective and scope to v18;defaults/v1-v17 stay. Old native must reject before GPU.

Before allocation,CPU checks cover action-plane layout/lane isolation and
derivatives,default-off/constructor/header restrictions,full geometry/unique names,
causal action/image graph ancestry and finite direct-action response at fixed
latent/history. Strict core/test/binding Clippy and isolated wheel tests. Reuse
unchanged Conv5/RMS/RSSM/event primitive receipts rather than repeat qualification.

Separately declare a900s AMD Auto/host-guarded native composition check:fixed
latent/history action changes alter decoder only in their lane;native values/
derivatives match a fixed CPU reference,new action/history path gradients are
nonzero,and future true targets cannot affect earlier states/images. One full-
shape update has thirteen finite reports,correct total/masks,240 finite/frozen
tensors,210 exact cached common initial tensors,16 exact nonimage arrays and18
initial/trained restore arrays. Retain actual-initial RGB;no old-model query.

Only after reviewed qualification,separately declare one fresh8,192-update learner
(3600s) and final restore(600s). Use original16 fitting/eight development streams,
sealed full4/2/1/1 plan and exposures;no collection or recount. RGB64/12M/B8/T32/
H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01,observed/future RGB100,KL1/.1/
free-nat1,event scales1. Retain actual initial/2048/4096/final,384 dense origins,
all thirteen reports and exact final restore. Reused data/seed are development.

After reviewed learning,declare only the new final84 frozen groups(1800s),history16
primary/32 secondary,eight draws,1/4/8/16/32 endpoints. Reuse cached oracle/H0/
fusion/v13/transport arrays without old-model queries. All original gates stay:
observed>=90%,one>=89/109,eight>=28/107,primary category-balanced survival-weighted
16/32 ball energy<=.12712666431732942,H0.16950221908977256,serve-NOOP false balls
<=10%,ending32 Brier<=.10,ending16<=.01,RIGHT16 risk<=.10,every nonending category
mean<=5%,freeflight8 presence>=90% for every command. Retain all individual alarms
and action/horizon reward/paddle/brick regressions;no best checkpoint/draw.

Precise physics,event timing/calibration,persistent removed-brick memory,late/
tunnel/second-wall coverage,untouched families and fresh confirmation remain
required. Existing oracle has no above-top-brick endpoints;intermediate gates
cannot finish the goal. No unchanged extension,sweep,completed-stage repeat or
automatic successor. Check live identities,stop actual faults/nonfinite state;
no reset,recovery,NVML. Root `/mnt/data/kindle-breakout-world-local-action-20261108`.
No GPU allocation or full learner is declared at selection.
