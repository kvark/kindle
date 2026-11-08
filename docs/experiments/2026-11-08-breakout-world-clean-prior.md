# Breakout: train causal readouts at every observed transition

The suffix-event learner remains closed/failed. A new CPU count in
`/mnt/data/kindle-breakout-world-clean-prior-20261108/support-review.json` finds
65,454 valid first-suffix forecasts from real history, including623 terminals,
among2,097,152 observed arrivals. The existing data offers2,021,344 valid causal
one-step transitions/10,826 terminals/33,188 positive rewards. These are replayed
exposures, not independent samples or proof of a bottleneck. Later suffix steps
already have direct supervision but condition on imagined states.

Select explicit standalone objective `motion_clean_prior`, v14. Keep the same
categorical fusion architecture and inference. Replace posterior RGB/reward/
continuation readouts in the observed training pass with readouts of its causal
prior: h_t uses observations only through t-1; sample z_t from p(z_t|h_t), before
q_t sees the current image. The observed teacher trajectory still uses q_t to
condition its next transition. No clean-prior prediction corrects that trajectory.

Decode from the same preceding-two-image history, with ordinary observed reset
controls. Mask reset/first arrivals, whose incoming state is unavailable. Genuine
terminal arrivals remain valid. Use motion-weighted RGB CE100, reward CE1 and
continuation soft-label BCE1, all averaged over full BxT. No extra decoder pass,
weights or heads. Posterior KL1/.1/free-nat1 remains; the encoder/posterior learns
through subsequent transitions and KL rather than current-image reconstruction.
This changes conditioning, gradients, reset weighting and sampling; it is not an
isolated frequency intervention. New prior samples consume additional RNG draws.

Keep the v13 H16 free rollout and its RGB100/reward1/continuation1 losses,
generated-image history/full BPTT, and no generated-image re-encoding. No future
image/event/reset target enters a forecast. No labels, pretrained encoder, teacher,
actor/critic, collector or backend primitive. API reports keep the13 positions,
with indices1/4/5 explicitly renamed clean-prior RGB/reward/continuation; append
clean-prior valid fraction at13. Total formula stays, with fraction excluded.
Bind v14/objective/architecture; defaults/v1-v13 stay and old natives reject early.

CPU checks cover headers/scopes/CLI rejection before GPU, causality of actual
main-pass RGB/event readouts, no current/future-image or target ancestry, past
image dependence, required gradients, one shared decoder and unique inputs.
Use the unchanged previously qualified sampling, CE/BCE, motion/history and
fusion math; do not repeat their completed native numerical fixtures.
Separately declare one900s native qualification: actual-main-pass target/image
perturbations leave causal readouts unchanged; past-image changes affect only
their lane; zero masks zero the relevant losses. One full-shape update must have
14 finite reports, exact totals and both valid fractions, required changed
families,all240 cached v13 initial tensors/18 initial arrays,240 own frozen
tensors and18 initial/trained restore arrays. Reject v14 under retained v13
native with GPU discovery disabled. No old-model query or stage repeat.

After reviewed qualification, separately declare fresh8,192 updates from0
(3600s), final restore (600s). Keep RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,
rates4e-4,mixing.01,80 weights/8,871,955 scalars/240 tensors,original full4/2/1/1
plan,16 fit/eight dev streams. Retain actual initial/2048/4096/final,384 dense
origins,all reports/exposures and exact final restore. Reused seed/families are
development, not independent confirmation. Do not claim a speed benefit without
useful frozen quality; count wall/native costs separately.

After reviewed learning/restore, declare only the new final's84 frozen groups
(1800s), histories16 primary/32 secondary,eight draws,all1/4/8/16/32 endpoints.
Reuse cached oracle/H0/fusion/suffix-event arrays. Keep every original gate:
observed>=90%,one>=89/109,eight>=28/107,primary16/32 category-balanced survival-
weighted ball energy<=.12712666431732942,serve NOOP false balls<=10%,ending32
Brier<=.10,ending16<=.01,RIGHT16 ending<=.10,non-ending category means<=5%,
freeflight8 presence>=90% on each command. Keep all individual alarms and
per-action/horizon reward/paddle/brick regressions. No best checkpoint/draw,
easier metric,unchanged extension,sweep or automatic successor.

Full causal physics,event timing/calibration,persistent brick memory,late/tunnel/
second-wall coverage,untouched families and fresh confirmation remain required.
The oracle has no above-top-brick endpoints. Intermediate gates do not complete
the goal. AMD/Auto/host guard;stop real faults/nonfinite state,no recovery/NVML.
Root `/mnt/data/kindle-breakout-world-clean-prior-20261108`;no GPU allocation at
this declaration. Original failed learners and collectors remain closed.

## User-authorized resume after pause

The user paused the run at2,998 reported updates, then resumed the goal. The
latest saved checkpoint is2,048;950 later updates have no saved weights. Preserve
the original interrupted queue,unsealed guard,logs and pause receipt. Do not
relabel them complete or restart that queue. Qualification remains separately
passed; no build or completed native fixture repeats.

Use the unchanged native/runtime35c090d0/SHA d60f2888. Separately declare one
600s guarded restore check of checkpoint2,048 against its cached18 forecast
arrays,and verify all240 weights/moments are unchanged by inference. Only after
review declare6,144 additional updates using exactly original plan rows2,048:8,192,
then final restore. This is a warm start: optimizer state is retained but RNG
restarts. It is not uninterrupted learning or an independent confirmation.

Use separate resume outputs and service identities. The Python runner may skip
the completed initial evaluation; default fresh-run behavior stays. Keep the
original0/1,024/2,048 evaluations and actual-initial dense baseline, rather than
querying them again. New periodic evaluations begin at3,072. Checkpoint4,096 and
an extra6,144 milestone follow the existing relative-step save schedule; only
final8,192 is assessed. Validate the suffix plan and counts before execution,
all14 finite reports/totals/fractions afterward,240 frozen tensors and18 final
restored arrays. Include both retained-lineage cost and actual spent cost with
the discarded950 updates. No new collection or easier gate.

After reviewed continuation, separately declare only the new final's original
84 frozen groups. All original competence requirements remain. User resumption
authorizes this bounded continuation; no automatic successor or extension.


The first resumed restore check stopped before forecasting on an over-strict
Python config equality: the native getter emits short decimal float32 values,
while checkpoint metadata contains their expanded float64 JSON representations.
A separate CPU repair proves exact float32 identity for every numeric config
field, compares the getter against the cached original getter output, and keeps
the original failed guard/script. The repaired native check matches all18 cached
arrays and all240 unchanged weights/moments in13.049s. No learning occurred in
these checks. The1,431-pin continuation is separately declared under `resume`.
