# Breakout: fixed-contact fitting with the current motion objective

The image-feedback world improves recorded eight-action precision but fails the
physical/event gates. Fitting-stream forecasts remain weak too. Earlier eight-clip
capacity checks used uniform squared RGB error; the later motion-weighted H16
objective was tested on broad data. Select a small, fixed fitting diagnostic to
determine whether that objective can accurately fit the retained contact clips.
This is a distinct data/objective combination, not a repeat of a closed learner.
It does not reduce the full world-model goal or establish generalization.

Reuse the eight original standard sticky-.25 fitting clips, one per independent
environment, and their first-contact starts `[43,27,43,43,28,49,43,67]`. Each has
32 arrivals and a candidate paddle-height vertical reversal after the16-arrival
prefix. These are selection predicates, not exact collision labels. Keep native
max-pooled frames and the already saved single GPU RGB64 resize. No collection,
emulator actions/frames, object/RAM training labels, teacher, actor or critic.

Fresh2,048 updates on the same B8/T32 batch, RGB64/size12M/full BPTT/Auto,
seed41001/rates4e-4, categorical mixing.01, observed and causal H16 motion RGB
cross-entropy100, KL1/.1/free-nat1, observed reward/continuation1. Use the qualified
85dbe150/SHA5b112866 runtime unchanged:68 logical/8,852,675 scalar weights/204
tensors. No spatial decoder or image feedback. Changed loss formulation and
runtime versus the early MSE contact test are disclosed together; initial204
tensors match the retained early and broad-data H16 initial controls exactly.
No isolated effect of motion weighting, CE, data diversity or update budget follows.

CPU preparation verifies all eight source hashes, original selection/manifest,
window alignment, episode/reset boundaries, incoming actions and exact saved RGB
and outcome targets. Only RGB/actions/reward/terminal enter native learning.
Archive runner/config/data/protocol/native identities before execution. Reuse
passed native motion-loss/derivative/mask/restore qualification; do not rebuild or
repeat primitive fixtures. One separately declared guarded learner,900s cap,
stops on host faults or nonfinite reports. Check all eleven loss reports, weighted
total, suffix validity, every update, actual initial204 tensors and final frozen204.
No automatic retry, extension or successor.

Retain actual initial,512,1024,2048 checkpoints, all loss rows and four fixed
draws83001/83003/83009/83023 at0/512/1024/2048. Export every future image and event
head at all sixteen actions, with observed reconstruction and actual targets.
Only past16 images and recorded future actions enter predictions; no future
image correction. Save one final untouched204-tensor copy after frozen queries.
Separately guarded final restore,300s cap, must reproduce all eight prediction
arrays for all four draws exactly without updates, including RGB and event heads.

The fitting diagnostic passes only if each of four final draws gets every
supported observed/1/4/8/16 ball within one resized pixel, and each draw gets>=90%
of all supported1..16 endpoints within one pixel. Missing predicted balls fail.
Report every individual endpoint, missing/unsupported ball, paddle/brick error,
reward and continuation prediction. The conservative below-brick detector leaves
unsupported positions untested; it cannot certify a full-field world model.
Compare cached early H16 MSE and new actual-initial arrays without old-model
queries. Show fixed lane0/draw0 full forecasts, never best matches.

This experiment probes fitting ability, not causal action intervention, natural
event calibration, held-out physics or seed reliability. All original full-corpus
physical/event gates remain unchanged and are not reevaluated using this easier
support. Precise all-horizon physics, brick memory, late/tunnel/second-wall
coverage, untouched families and fresh confirmation still govern the full goal.
Root `/mnt/data/kindle-breakout-world-contact-fit-20261108`. AMD host guard/Auto;
no reset, recovery or NVML polling. No GPU allocation has run at declaration time.
