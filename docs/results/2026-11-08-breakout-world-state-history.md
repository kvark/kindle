# Breakout: correcting image history repairs many frozen visual readouts

The separately repaired diagnostic is complete. On the local-action H16 model,
replacing its two generated past images with true past images raises eight-step
ball accuracy46/108→96/108;replacing recurrent h alone gives44/108. At32,the
corresponding counts are8/108→74/108 with history and8/108 with h. The H32 model
needs more joint correction:at8,prior28/108,history58/108,h28/108,history+h92/108.

This identifies a large **readout sensitivity to autoregressive image history**
in these models. It differs from the older nonspatial [state-swap result](2026-11-07-breakout-world-state-swap.md),where h substitution repaired much of the
visual rollout. It does not assign a causal percentage to image history or prove
a trainable fix. No learned forecast improved. The full world-model goal remains
active and unproven;actor/critic and CDP comparisons stay parked.

[Protocol](../experiments/2026-11-08-breakout-world-state-history.md).
Root `/mnt/data/kindle-breakout-world-state-history-20261108`;repaired outputs
under `repair/`. Original attempt remains **protocol-failed**,not relabeled.

## Intervention and data

Frozen local-action H16 final8192 and H32 final4096,one at a time. Both v18,
RGB64/B8,80 logical/8,893,203 scalars/240 tensors,categorical RSSM,two-image local
history and direct action planes. Saved T/full BPTT,world/dynamics rates,budget,
optimizer age and H16/H32 differ;this is not an isolated-horizon comparison.
No optimizer step,actor/critic,emulator,collection,new loss or architecture.

Take the same first32 of128 dense development origins,in original B8 order,and
all10 supported terminal-centered origins with game over at future17. No source
exclusion was needed;six padding lanes are unscored. Four seeds83001/83003/83009/
83023. Reused related development windows,not untouched families or independent
confirmation. RGB64 comes from the original one-GPU resize;no camera or
CPU downscale/upscale change. RAM/positions/lives/phase/outcome labels stay CPU
metadata,never network inputs or training targets.

A fixed prior trajectory and a separate true-image teacher supply h,z and
history. All eight readouts use the same incoming command. Teacher h has seen
true images through the previous arrival;teacher z also sees the current true
image. Teacher history contains only the two earlier true images,with production
reset masks. Those later images are privileged future information relative to
the original forecast. No hybrid feeds either path. Swaps can be off-manifold;
full teacher is a diagnostic reference,not an optimal upper bound.

## Ball positions: all eight readouts

Within one resized RGB64 pixel. Visible targets pool four model draws per origin:
108 means27 visible origins×4,not108 independent episodes. Missing/ambiguous
balls fail. Endpoint support varies. These32-origin tables do not replace the
original128-origin dense or84-group frozen retention assessments.

Local H16,natural origins:

|Readout|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|
|Prior forecast|95/108|57/108|46/108|18/112|8/108|
|Future h|95/108|59/108|44/108|12/112|8/108|
|Future z|100/108|54/108|43/108|14/112|9/108|
|Future h,z|100/108|57/108|43/108|14/112|8/108|
|True past images|95/108|88/108|96/108|71/112|74/108|
|True past+h|95/108|92/108|100/108|88/112|88/108|
|True past+z|100/108|88/108|100/108|80/112|88/108|
|Full teacher|100/108|96/108|100/108|92/112|88/108|

Local H32,natural origins:

|Readout|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|
|Prior forecast|84/108|36/108|28/108|12/112|5/108|
|Future h|84/108|31/108|28/108|4/112|7/108|
|Future z|89/108|32/108|20/108|12/112|4/108|
|Future h,z|89/108|24/108|23/108|10/112|4/108|
|True past images|84/108|76/108|58/108|31/112|28/108|
|True past+h|84/108|88/108|92/108|88/112|80/108|
|True past+z|89/108|92/108|74/108|46/112|42/108|
|Full teacher|89/108|87/108|94/108|86/112|82/108|

H16 history-only correction rescues50 prior misses and harms0 successes at8;
h-only rescues2/harms4. At32,history rescues66/harms0,h0/0,and history+h80/0.
H32 history-only at8 rescues38/harms8;history+h rescues64/harms0. These are paired
sample counts,not additive causal error fractions or seed reliability.

Terminal-centered H16:

|Readout|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|
|Prior forecast|32/40|28/36|8/36|—|—|
|Future h|32/40|21/36|8/36|—|—|
|Future z|36/40|28/36|8/36|—|—|
|Future h,z|36/40|24/36|8/36|—|—|
|True past images|32/40|36/36|24/36|—|—|
|True past+h|32/40|36/36|32/36|—|—|
|True past+z|36/40|36/36|31/36|—|—|
|Full teacher|36/40|36/36|32/36|—|—|

Terminal-centered H32:

|Readout|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|
|Prior forecast|33/40|25/36|8/36|—|—|
|Future h|33/40|16/36|13/36|—|—|
|Future z|34/40|13/36|7/36|—|—|
|Future h,z|34/40|13/36|16/36|—|—|
|True past images|33/40|28/36|20/36|—|—|
|True past+h|33/40|28/36|32/36|—|—|
|True past+z|34/40|26/36|18/36|—|—|
|Full teacher|34/40|28/36|35/36|—|—|

No visible balls support terminal-slice16/32 position scores. Terminal arrival17
is supported for events;later unknown/reset arrivals are unscored. All paddle,
brick,detection,error,paired harm/rescue and latent-drift statistics remain in
`analysis-review.json`;the ball table is not the entire result.

## Event readouts remain weak with true-image conditioning

History substitutions leave all reward/positive/continuation heads exactly
unchanged,because they read RSSM features only. Correcting decoded image history
therefore does not directly repair these frozen event heads. This is a verified
structural invariance,not an inference that changing training cannot help them.

The selected cohort has no true ending through16 and all10 terminal arrivals at17.
Cumulative values multiply conditional survival along each readout sequence,
then average draws. Instantaneous17 removes the configured continuation discount.

H16:

|Readout|Cumulative end by16|Cumulative end by17|Instantaneous terminal17|
|---|---:|---:|---:|
|Prior forecast|0.361894|0.409595|0.190837|
|Future h|0.378492|0.439292|0.231157|
|Future z|0.322575|0.380585|0.170246|
|Future h,z|0.335223|0.402440|0.205339|
|True past images|0.361894|0.409595|0.190837|
|True past+h|0.378492|0.439292|0.231157|
|True past+z|0.322575|0.380585|0.170246|
|Full teacher|0.335223|0.402440|0.205339|

H32:

|Readout|Cumulative end by16|Cumulative end by17|Instantaneous terminal17|
|---|---:|---:|---:|
|Prior forecast|0.210135|0.262558|0.092721|
|Future h|0.173955|0.220413|0.072755|
|Future z|0.248950|0.302308|0.107195|
|Future h,z|0.207761|0.257725|0.086247|
|True past images|0.210135|0.262558|0.092721|
|True past+h|0.173955|0.220413|0.072755|
|True past+z|0.248950|0.302308|0.107195|
|Full teacher|0.207761|0.257725|0.086247|

Even full-teacher instantaneous terminal means are.205339(H16)/.086247(H32)
on these ten examples. Earlier cumulative risk is also present. Thus replacing
wrong rollout features with true-image features does not make event timing
accurate on this slice. This does not identify head capacity,representation,
optimization or irreducible observation ambiguity as the cause. Outcome-selected
examples do not establish natural-frequency calibration;no assisted score replaces
an original gate. Latent cosine/KL/RMS comparisons are checkpoint-specific,not a
shared competence metric.

## Native controls,failures and cost

Source `22cf176a88ab75d1503dba52d95b70d7632189d8`,test executable SHA256
`091c06614357073f911b04847f08253cca408dcb66805d7cba93dc4360edc93c`.
Only a cfg(test) registration and new orchestration differ from qualified native
source7f797a2b. Production graphs,math,older diagnostics and wheel stay unchanged;
no primitive qualification repeats or runtime overlay. Auto on RX7900XT/RADV,
existing host guard,no reset/reload/recovery/NVML;other host GPU work may exist.

First compilation fails eight age comparisons from usize versus u64/u32.
Separate explicit-u64 repair passes three CPU checks;strict core/test release
Clippy passes. Cached data preparation passes eight exact shared reference joins.
The first CPU packaging assertion falsely matches `thiserror` in a successful
Clippy log;its separate parser repair reuses the completed check. All failures,
source/logs and the short-lived helper prototype remain retained.

Original native attempt then fails the final canonical teacher API call,which
requires at least one future batch. Its first trace/common-prefix/history/head
controls run,but no per-query export is completed. Preserve the unsaved work and
9.72s native-test cost,the empty query log,before-H16 snapshot and27 sealed failed
guard pins. The observer correctly stops on that failed service. No device fault
or model update is reported.

The separate repair supplies one unused complete NOOP future batch to a
48-observation canonical predictor. Its posterior is computed before that future
prediction,which is never scored or fed to a hybrid. Final CPU checks/strict
Clippy pass;original source/executable/guard stay unchanged. This completes only
missing new diagnostic outputs,not a repeat of a completed assessment/collector.

All48 bundles/eight modes,four full movies,64 cached shared arrays,bitwise prior
transition/RGB/head replays,prefix/first-step/history/head-pair controls and48
final teacher-feature/posterior controls pass. Initial/final snapshots reproduce
960 tensors;the original failed snapshot separately matches240. Both optimizer
ages stay8192/4096,with zero new updates. All targets/masks/padding are retained.

Repair execution verifies2,370 pins. Native function wall66.212892s,
Rust test wall66.70s,CPU analysis25.826287s;build,preparation,
original failures,guard/poll spans,figures and publication are additional. These
wrapper times do not isolate kernels or certify exclusive-GPU speed.

Four first-origin/draw lossless clips retain every supported decision and all
eight modes,with98 decoded frames matching exactly. All raw arrays,weights,
source/executables,figures and failures remain. Disposable736,178,565-byte build
cache is released after successful native/CPU review. Closure verifies
2518 pins,successful guard9/failed guard27,and both service identities
terminal. Original failure remains failed;the repair is a completed diagnostic.

## Consequence for the next learning decision

The current local decoder can recover many positions when supplied accurate past
images. Replacing only h or z leaves its corrupted generated-image history in
place and usually repairs little. H32 shows additional state/history interaction.
This directs the next separately justified learning change toward stable visual
transitions and coherent state/event fitting,rather than another unchanged long
extension or a claim that more h supervision alone is sufficient.

No successor learner is selected here. Prior H16/H32 forecasts and their failed
retention gates remain unchanged. Precise physics,events,persistent removed-brick
memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation
remain unproven. The full world-model and five-game goals stay unfinished.
