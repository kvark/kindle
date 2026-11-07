# Frozen Breakout recurrent/categorical state substitutions

The [closed latent-overshooting comparison](../results/2026-11-07-breakout-world-overshooting.md)
improves ending endpoints,but fails tracking and predicts some endings too early.
Before another objective,measure whether replacing the recurrent or categorical
block can repair decoded trajectories and event timing. This is a new frozen
intervention/output query,not another learner or a repeat of a closed assessment.
Root `/mnt/data/kindle-breakout-world-state-swap-20261107`.

## Fixed checkpoints and data

Use coverage and latent-overshooting final8,192 checkpoints,one at a time,with
their saved RGB64/12M/B8/Auto configurations. Keep all204 tensors,including moments,
bitwise unchanged. No optimizer step,emulator,new collection or behavior learner.
Use one pinned native test executable with unchanged production forward graphs;
no wheel rebuild or overlay mixing. Verify retained baseline arrays where present.

Take exactly the first32 origins from the retained128-origin dense development
selection,in its original B8 batch order. Four seeds83001/83003/83009/83023 are fixed.
The first seed's five endpoint RGB/feature arrays and32 reward/continuation arrays
must match each original dense export bitwise. These controls do not constitute
fresh evaluation; intermediate frames/features are previously missing outputs.

Separately take all original-development natural terminal arrivals with at least
32 preceding arrivals and no intervening reset. Set each start to terminal_index−32:
after history16,the terminal is future step17. This is label-selected diagnostic
coverage,not a natural-frequency calibration sample. Exclude only unsupported
source windows,record all exclusions,and pad the last B8 batch using its first
origin. Never count padding. No fitting or untouched-test family is introduced.

Reuse cached RGB64 from the original single GPU resize. No resize/downscale/
upscale or emulator reconstruction. Native inputs are RGB and incoming actions.
RAM/lives/episode/frame/terminal/reward labels remain CPU analysis metadata; no
labels train or enter the world network. Preserve actual masks and action timing.

## Interventions

For each batch/seed,call the canonical prior predictor once from history16 under
recorded future commands. Keep all32 prior features,RGB and head outputs. In a
separate branch,observe the real48-frame trajectory using the same posterior
sampling stream. Prefix features must match exactly; after the prefix,the teacher
uses real future images and is therefore explicitly not a forecast.

At each future step decode four combinations without feeding them back:

| Readout | Recurrent block h | Sampled categorical block z |
|---|---|---|
|prior|prior|prior|
|teacher_h|image-informed|prior|
|teacher_z|prior|image-informed|
|teacher|image-informed|image-informed|

Replay each cached prior transition only to expose its missing prior logits;
require exact recurrent states and sampled z under the matched RNG stream.
Keep image-informed posterior logits too. Neither replay consumes a hybrid.
At step1,recurrent states and paired readouts must agree at1e-5 absolute plus
1e-5 relative tolerance across the different observe/transition graphs; teacher_h
must reproduce prior and teacher_z must reproduce teacher. Re-decoded prior RGB/reward/continuation must
match the canonical predictor at every step. Check all values finite.

Save all32 prior/teacher features and logits,all32 head traces and five endpoint
images for every readout/draw. Save full32 RGB traces for draw0 of the first batch
of each group,for both checkpoints. Export true/predicted clips for the fixed
first origin of each group,ending at its first unsupported arrival. Do not choose
the closest draw or a favorable context after results. Mark diagnostic future
information and readout-only swaps on clips.

## Analysis and limits

Report observed/perception ceiling and all1/4/8/16/32 readouts on the actual valid
support,ball detection/within1px/error,paddle and bricks. Count paired rescues and
harms relative to prior,per readout and checkpoint,without treating draws as
independent episodes. Compare recurrent-state drift and mixed categorical
KL(q_teacher||p_prior) within each checkpoint;latent geometry differs across
checkpoints and is not a common competence metric.

Report instantaneous reward/termination errors on supported arrivals,plus
cumulative events only where the complete real prefix exists. Terminal arrival
is supported; following reset/unavailable arrivals are not. For terminal-centered
windows,report cumulative ending by16 and17 and instantaneous terminal hazard
at17. Do not turn unknown post-terminal teacher states into32-step event evidence.
All raw masks,labels and traces remain retained.

Swapped states may be off-manifold. A successful substitution identifies a
possible readout repair with privileged future information,not a causal fraction
of model error,a learnable correction or a valid policy/model forecast. A failed
substitution does not prove a block irrelevant. This cannot qualify the complete
world goal,calibration,reliability,late-game/second-wall coverage or CDP superiority.
No new loss,capacity,budget extension or successor learner is selected here.

## Execution

Run the focused CPU block-layout check plus existing world tests and strict
Clippy. Pin source,executable,configuration,batches,baseline arrays,checkpoint
bytes and analysis protocol before GPU construction. Recheck live process/host
identity. One900s worker under the existing AMD host guard;Auto precision,no NVML,
reset/reload/reboot or retry. Stop on numerical/real GPU faults. Verify all204
checkpoint tensors before/after and original input hashes on CPU. Preserve any
failure and separate repair. Production packages/defaults remain unchanged.
