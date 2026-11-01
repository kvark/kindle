# Breakout: the full-width red mask yields no accepted paddle comparisons

The fixed full-width measurement has**zero accepted paired observations**.
Widening the existing red detector to columns0:160 introduces additional red
pixels or components touching the image boundary. The declared filters reject
every row,so this result does **not** quantify the old crop's position bias.
Keep the existing teacher unchanged. A simple full-width replacement is not
qualified by this experiment.

| Primary24-episode population | Rows |
| --- | ---:|
| Recorded decisions |42,303|
| Initial/reset observation, no raw pair |24|
| Extra red pixels outside the selected component |37,960|
| Component touches image edge |4,285|
| Nonstandard component width |34|
| Accepted same-time pooled comparisons |0|

Across all208,254 raw frames,the exclusions are186,864 extra-red,21,179
image-edge and211 nonstandard-width components. The selected episodes contain
169,178 raw frames:151,874 extra-red,17,130 image-edge and174 nonstandard.
The all-trajectory52,072 decisions comprise32 initial/reset,46,702 extra-red,
5,299 image-edge and39 nonstandard rows. All denominators reconcile.

The recorded examples explain why an unrestricted red centroid is unsafe as
a position measurement. Lane0/episode0/raw frame1 retains the same unique
16-pixel component center106.5, but the mean of all red pixels on that full row
is122.833. At raw frame24,the old cropped component is13 pixels wide with
center145.0,while the full component is21 pixels wide,center149.0,and touches
the image edge. Neither example establishes the actual paddle extent outside
the original crop or ground-truth collision state. No detector thresholds are
relaxed after seeing these outcomes.

The prior pooling/contact reports excluded cropped components and therefore
left edge measurement unresolved. This new diagnostic rules out accepting the
naive full-width mask under the fixed screen;it does not establish that the old
crop is correct,that all full-width detectors would fail,or that crop error is
the cause of missed contacts. A separately justified component/width tracker
would need its own validation before teacher or learner integration. None is
selected here. No response refit,gameplay comparison,loss or learner follows.

## Fixed replay and preserved active learner

The [declared diagnostic](../experiments/2026-11-01-cdp-breakout-paddle-crop.md)
replays all52,072 recorded actions/208,254 frames of the existing teacher.
Every original action,debug observation,pooled pixel hash,reward,boundary,
episode and partial tail reproduces. All208,254 prior cropped raw annotations
match exactly,as does the aggregate raw RGB hash. The source remains225.083
mean andzero of24 two-wall successes;there is no new gameplay score.

Eight CPU fixtures and3.309s preflight pass. Selection takes1.268s. The new
annotation replay takes53.457s and its controller56.159s,completing
**18:41:41 UTC November1**,source7433689. All47 diagnostic inputs and528 active
Qbert inputs stay unchanged,before/after preserving originalPID831942/
invocationbae3fe2dd4a24070ab2845a7828145fa. The low-priority single-core CPU
work co-hosts with Qbert training;retain these times without wall-time correction.
No native import,agent construction,GPU allocation,learner update,policy change,
failure or repeated diagnostic stage occurs. The completed replay has no
eligible paired sample;its complete status means execution and accounting
passed,not successful paddle measurement or task competence.

Root`/x/Temp/kindle-breakout-paddle-crop-20261101` retains all raw components,
pooled observations,exclusions and receipts. Complete;do not rerun this replay
or qualification. The original Qbert learner continues,and the five-game goal
remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`ab09c1298f0ed15f6bcf5a85076be12c93062bb8e41f0295be2d780adc565ad9`|
| Review |`8fe410c268bfa218fd2c57a2f5750067c8087fccffd108d8621c5fe27bfd85b2`|
| Raw components |`1ddbad15398fded48d74ffe65b1170609f868fc4a17e5f84ce784048c57b606e`|
| Pooled observations |`801790c84883becde41779737d6ba35fa6a2fc56e302017a1ed8029047a482bc`|
| Completion |`e26f3723cfc6dcf1dbc51655bfb971b5b3023318454e9a6374052d22c492592f`|
