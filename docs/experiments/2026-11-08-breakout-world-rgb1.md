# Breakout: test pixel loss weight 1 from a fresh start

The [fixed-weight diagnostic](../results/2026-11-08-breakout-world-loss-gradients.md)
measures future-RGB gradients 22–87 times KL in the prior and 336–884 times KL in
the recurrent core. Observed and future image gradients often oppose in the
encoder. These measurements do not establish a physical cause or useful weights.
They justify one distinct, bounded test of the existing RGB coefficient.

Change only `loss_scales.reconstruction` from 100 to 1 relative to the completed
v13 suffix-event learner. This changes both observed and imagined RGB losses.
Keep KL dynamics 1 / representation .1, free nat 1, reward and continuation 1,
including their existing H16 suffix supervision. No balancing algorithm, sweep,
new architecture, object labels, RAM inputs, actor/critic or collection.
The coefficient changes all relative gradients and their optimizer history;
it does not isolate a particular physical error or just one gradient conflict.

Reuse exact v13 native/source a8efffc6, SHA f40555b5, and its matching runner.
No native rebuild or repeated primitive/derivative/legacy qualification is needed
for this exposed scalar. Before learning, separately declare a guarded preflight
(600s): all 240 actual-initial tensors and 18 cached initial arrays must match
v13; one new full-shape update has 13 finite reports, correct new weighted total
and unchanged raw reports within rounding tolerance. All expected weight families
must change, and initial/trained restore must reproduce 18 arrays exactly.
Reuse original causal-event and derivative receipts without querying old models.

Only after reviewed preflight, separately declare fresh 8,192 updates (3600s)
from seed 41001, followed by exact final restore (600s). Keep RGB64/12M/B8/T32/
H16/full BPTT/Auto, rates 4e-4, mixing .01, spatial fusion on, image feedback and
pixel residual off, 80 logical weights / 8,871,955 scalars / 240 tensors. Use the
unchanged full 4/2/1/1 plan: 16 fitting and 8 development streams, 2,097,152
replayed arrivals, 975,941 valid suffix rows / 9,570 terminals / 16,008 positive
rewards. Reuse the sealed plan/exposure count; do not recollect or recount it.
Keep actual initial, 2048, 4096 and final checkpoints, all reports and 384 dense
origins with actual-initial controls. No warm start or selected checkpoint.

After reviewed learning/restore, separately declare only the new final's 84
frozen branch groups (1800s), histories 16 primary / 32 secondary, eight draws,
endpoints 1/4/8/16/32. Reuse cached oracle, H0, motion-rollout, fusion and v13
arrays. The RGB detector and original unsupported endpoints remain unchanged.
No old model query or completed-stage repeat.

All gates stay: observed ball >=90%, one-step >=89/109, eight-step >=28/107;
category-balanced survival-weighted 16/32 ball energy <=.12712666431732942
(H0 .1695022191); serve-NOOP false balls <=10%; ending Brier at32 <=.10,
at16 <=.01 and RIGHT-at16 <=.10; every nonending category mean <=5%; freeflight
presence at8 >=90% for each command. Keep every individual false alarm and all
per-action/horizon reward, paddle and brick regressions. No best draw, relaxed
gate, unchanged extension, automatic successor or seed campaign.

This is a reused development seed/family test. Full precise physics, event
timing/calibration, persistent removed-brick memory, late/tunnel/second-wall
coverage, untouched families and fresh confirmation remain required. The oracle
has no above-top-brick endpoints. Passing these intermediate gates cannot prove
the complete world model. Goal remains active and unproven.

Root `/mnt/data/kindle-breakout-world-rgb1-20261108`. Check live process and
service identity before allocation; host guard / AMD Auto; stop real faults and
nonfinite values. No recovery, reset or NVML. No GPU allocation at selection.
