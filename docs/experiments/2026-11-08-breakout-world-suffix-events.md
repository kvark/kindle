# Breakout: supervise imagined events without image feedback

The [event diagnosis](../results/2026-11-08-breakout-world-event-diagnosis.md)
finds weak observed terminal fitting and increasing false ending risk during
rollout. Fusion improves recorded short ball forecasts but fails full retention.
The older image-feedback learner also adds future event losses while changing
recurrent conditioning. Select a distinct test of those losses without feedback.

Add explicit standalone objective `motion_event_rollout`. Require categorical
RGB with the existing spatial fusion decoder and a positive suffix. Preserve
ordinary `motion_cross_entropy`, old headers and defaults. Bind the new objective
to v13; old natives must reject before GPU allocation. No new architecture flag,
weights, labels, teacher, backend primitive, actor/critic or collection.

At each imagined step use the same causal RSSM prior state for RGB, reward and
continuation. Add existing stopped-target two-hot reward cross-entropy and
soft-label continuation BCE, each at its ordinary configured scale1. Average
over full BxH, retaining terminal arrivals and masking reset crossings. Event
targets/weights never change state, predicted RGB or history. No generated-image
encoding or posterior correction; generated images only shift image history.
Event gradients reach RSSM and heads; they do not traverse the image decoder.
RGB losses retain full BPTT through generated-image history. Reports grow11→13;
the exact total includes both future heads. No class reweighting or scalar fit.

CPU checks cover objective/header/scope/CLI rejection before GPU, full-shape
parameter and input geometry, causal ancestry, gradient connectivity, report
order and totals. Existing qualified event-loss math is reused unchanged; do
not repeat its completed native numeric fixture. New guarded qualification
(900s) checks target/mask perturbations leave imagined states/RGB unchanged,
event losses respond and zero masks zero all suffix losses, and lane isolation.
Use one full-shape update with all13 finite reports and mask/total identity,
all240 exact fusion initial tensors,18 initial/trained restore arrays, required
changed families and no added parameters. Compare the new actual-initial output
with cached fusion output, never query an old model again. Reject v13 under the
retained v12 native with the GPU loader disabled.

Only after reviewed qualification separately declare fresh8,192 updates from0
(3600s), then final exact restore (600s). RGB64/12M/B8/T32/H16/full BPTT/Auto,
seed41001, rates4e-4, mixing.01, original full4/2/1/1 plan, observed/future motion
RGB100, KL1/.1/free-nat1 and observed event scales1 stay. Keep80 logical weights,
8,871,955 scalars/240 tensors, RSSM2560 plus24,576 image-history scalars. Recount
975,941 valid suffix rows/9,570 terminals/16,008 positive rewards. Sixteen fitting
and eight development streams, actual initial/2048/4096/final checkpoints,
384 dense origins, all reports and exposures stay. Reused seeds/families are
development, not an independent confirmation or untouched generalization test.

After reviewed learning/restore, declare only the new final's84 frozen groups
(1800s), histories16 primary/32 secondary, eight draws and1/4/8/16/32 endpoints.
Reuse cached oracle/H0/H16/fusion arrays; no old query, collector or stage repeat.
All gates stay: observed>=90%, one>=89/109, eight>=28/107, primary category-
balanced survival-weighted16/32 ball energy<=.12712666431732942, serve NOOP
false balls<=10%, ending32 Brier<=.10, ending16<=.01, RIGHT16 ending<=.10,
nonending category means<=5%, freeflight8 presence>=90% on every command.
Keep all individual alarms and per-action/horizon reward/paddle/brick regressions.
No best checkpoint/draw, easier metric, unchanged extension, sweep or successor.

Full physics, event timing/calibration, persistent brick memory, late/tunnel/
second-wall coverage, untouched families and fresh confirmation remain required.
The oracle has no above-top-brick endpoints. Intermediate gates cannot prove
full competence. AMD/Auto/host guard; stop real faults/nonfinite state, no reset,
recovery or NVML. Root `/mnt/data/kindle-breakout-world-suffix-events-20261108`.
No GPU allocation exists at this declaration.

Preserve the initial new CPU-fixture compilation failure: two microbatch fields
were assigned integers instead of `Some` values. The separate fixture correction
passes105 world CPU checks;33 native checks remain ignored. A pre-execution
source review also distinguishes13 API reports from38 reserved/debug graph
outputs. No production math correction or native work repairs this fixture.
