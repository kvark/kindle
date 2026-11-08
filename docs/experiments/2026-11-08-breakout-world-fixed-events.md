# Breakout: future events under a fixed observation filter

The [fixed-filter phase](../results/2026-11-07-breakout-world-fixed-teacher.md)
improves primary ball energy .169502→.141405 and retains zero nonending alarms
above5%, but fails eight-action precision, RIGHT freeflight presence and ending32
Brier. Its imagined trajectories have latent/recurrent targets without explicit
future event supervision. [Joint suffix events](../results/2026-11-08-breakout-world-suffix-events.md)
improve fusion's event/ball scores but still fail. The later
[RGB-weight1](../results/2026-11-08-breakout-world-rgb1.md) comparison also fails.
Neither gradient balance nor a frozen filter alone establishes physical competence.

Select one distinct composition: keep the original fixed-filter phase and add
future reward/continuation losses to its two existing recurrent rollout draws.
After each causal h transition, sample its ordinary categorical prior z and read
the existing reward and continuation heads from h/z. Average the two draws and
all BxH rows using existing stopped-target two-hot CE and soft-label BCE, scales1.
Retain terminal arrivals and mask reset crossings. Reuse the recurrent mask;
do not introduce duplicate named inputs. Add only the final-step prior samples
needed by the event readouts; preceding transition/sample paths remain the same.
Future labels/masks never enter the predicted states or immutable filter.

New explicit `motion_fixed_teacher_events` objective and
`start_fixed_teacher_events_phase` API; checkpoint v16 binds objective/scope.
Ordinary construction must reject before GPU. Preserve all original defaults,
v1-v15 paths, old14-report fixed-filter objective and optimizer/checkpoint layouts.
New API reports16 values: original14 plus future reward/continuation. No new
parameters, RGB future loss, encoder, backend primitive, object/RAM training
label, collector, actor or critic. Derivatives remain categorical ST surrogates.

Keep the immutable H0 observation filter, fixed36 visual weights and32 active
dynamics/head weights. Source is the same retained H0 final8192, with all204
weights/moments and optimizer age preserved. This new warm phase starts from
that anchor, not the closed fixed-filter final. The extra filter remains39 logical/
6,203,136 scalars; student68/8,852,675/204 is not the total footprint. Preserve the
embedded anchor/digest, fixed teacher targets at declared seeds and original
observed outputs. Do not restart any old learner or qualifier.

Before allocation, run affected CPU graph/gradient ancestry, unique-input,
header/scope/report-total and Python API/runner checks, plus strict Clippy.
Reuse unchanged CE/BCE/GRU/decoder/filter/derivative receipts rather than repeat
their native numeric fixtures. Separately declare a new composed qualification
(900s): target/mask perturbations leave rollout states unchanged, zero mask
zeroes new losses, event targets respond and lanes stay isolated. Require204
exact inherited tensors,18 cached original-phase initial arrays, all original14
first-update components within rounding tolerance,16 finite new reports/total,
32 changed active weights,36 exact visual weights and18 trained restore arrays.
Teacher targets/anchor/GPU equality must remain exact before/after update/restore.
Old native must reject v16 with its GPU loader disabled. Preserve any failures.

Only after reviewed qualification, separately declare4096 new updates (3600s),
global8192→12288, followed by final exact restore (600s). Keep RGB64/12M/B8/T32/
H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01,base RGB100,KL1/.1/free-nat1,
latent overshooting1,recurrent energy100 and all head scales1. Same original
first4096 plan rows,16 fitting/eight development streams and all exposures.
Reuse the sealed sampling/exposure receipt, no recount or new collection.
Keep initial/intermediate10240/final,384 dense initial/final origins,all reports,
visual/filter invariance and inherited plus incremental learning costs. Warm
initialization/RNG restart and reused seed/families are not fresh confirmation.

After reviewed learning/restore, separately declare only the new final84 frozen
groups (1800s),history16 primary/32 secondary,eight draws,1/4/8/16/32. Reuse oracle,
H0,original fixed-filter and suffix-event arrays; no old model query or stage
repeat. All gates stay: observed>=90%,one>=89/109,eight>=28/107;primary ball energy
<=.12712666431732942;serve-NOOP false balls<=10%;ending32 Brier<=.10,ending16<=.01,
RIGHT16 mean risk<=.10;every nonending category mean<=5%;freeflight8 presence>=90%
for every command. Retain individual alarms and all reward/paddle/brick regressions.

Full precise physics,event timing/calibration,persistent removed-brick memory,
late/tunnel/second-wall coverage,untouched families and fresh confirmation stay
required. Existing oracle has no above-top-brick endpoints. Intermediate gates
cannot complete the goal. No unchanged extension,sweep or automatic successor.
AMD Auto/host guard,check live identities,stop real faults/nonfinite values;
no recovery/reset/NVML. Root `/mnt/data/kindle-breakout-world-fixed-events-20261108`.
No GPU allocation or full learner at selection.
