# Breakout: self-generated image feedback and future event supervision

The closed spatial-decoder recipe fails physical/event gates:energy .478844,
187/304 nonending cases>5%,development87/112,81/109,24/107. It feeds image
history only into decoding; recurrence and event heads remain latent-only.
Select a distinct joint-world hypothesis: couple generated images back into
belief and supervise reward/continuation along the resulting imagined states.
This changes recurrence,sampling/gradients and future losses together. No unique
cause,external-method reproduction or benefit is established.

Default-off `world_image_feedback` requires the standalone categorical RGB
spatial decoder and explicit `motion_image_feedback` objective,positive suffix.
Keep v10's77 logical/8,857,723 weights/231 tensors and all geometry. For each
future step: advance GRU once from preceding belief/action;sample prior z;decode
RGB from prior state and two-image history;encode that generated RGB using the
learned CNN;compute/sample posterior z at the SAME already-advanced h;use this
self-observed belief for heads and next GRU step. Shift generated RGB into image
history. No second time advance,no true future image or outcome/reset label in
belief. Full BPTT through decoder,generated-image encoder,posterior and recurrence.

The legacy `prior_features` readback now means self-observed imagined state,
not raw prior state. RSSM feature2560 still omits two-image memory24,576 scalars.
The new feedback inference session reuses encoder/posterior parameters; no new
logical weights or backend primitive,but extra graph/activation/session work.
Fresh named initialization should match all231 spatial actual-initial tensors,
plus201 common H0 tensors. Verify own initial/restored controls; no warm start
or old forecast rerun. First generated RGB may match cached spatial first-step
control; later states,heads,RNG consumption differ and need not match.

Keep existing observed RGB100,KL1/.1/free-nat1,reward/continuation1 and H16 motion
RGB100. Add future reward two-hot CE1 and future continuation soft-label BCE1
on self-observed imagined states. Targets/masks are stopped,including discounted
nonterminal continuation targets. Preserve terminal arrivals; mask all after a
reset crossing. Average over full BxH,not number of valid targets. Reuse existing
arrival-aligned reward/continuation inputs without duplicate names. Report13
values:old11 plus future reward/continuation. Total identity includes both new
terms. Future targets never influence recurrent state or reset imagined history.

v11 persists both architecture flags/objective; old native rejects before GPU;
old versions cannot disguise feedback. Reject original decoder phases/whole
agents. Defaults and v1-v10 paths stay unchanged. Restore preserves geometry/
objective; checkpoint omits replay/live beliefs/image history/full RNG.

CPU checks:config/header/objective/shape/counts,unique inputs,generated-image
feedback ancestry,one time advance,exclusion of actual future images/labels/
reset masks from state,full gradient connectivity,first-step controls where
applicable. F64 future CE/BCE values/analytic derivatives with terminal,reset,
all-zero masks and stopped targets. Packaged Python and strict Clippy. Reuse
qualified Conv5/RMS/categorical/resize/history primitives. Separately declare
one900s guarded native qualification:new loss values/derivatives,feedback at
fixed h changes with generated-image input without lane mixing,full-shape one
update,finite13 reports/total/masks,all expected changed families,231 spatial
initial/201 H0 common tensors,231 own initial/frozen tensors,18 initial/trained
restore arrays and v11 old-native rejection. Preserve failures and separate repairs.

Only after reviewed qualification separately declare fresh8,192 updates(3600s)
and final restore(600s). RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,
original full8,192-row4/2/1/1 plan,sixteen fitting/eight development streams and
all old source exposures stay.975,941 valid suffix rows/9,570 suffix terminals;
independently recount future positive rewards and new loss target exposures.
No new data,collection,object/RAM/phase labels,teacher,pretraining or actor/critic.
Retain initial/2048/4096/final,384 dense actual-initial/final origins,13 finite
reports,full exposures,231 frozen tensors and18 final restore arrays. Record all
new time and extra feedback work; no isolated speed or learning-per-hour claim.

After reviewed learning/restore separately declare only new final's84 frozen
oracle groups(1800s),history16 primary/32 secondary,eight draws,all1/4/8/16/32
endpoints. Reuse oracle/H0/categorical H16/spatial cached arrays without old-model
queries. Unchanged gates:observed>=90%,one>=89/109,eight>=28/107,primary16/32
survival-weighted ball energy<=.12712666431732942,serve NOOP falseball<=10%,
ending32 Brier<=.10,ending16<=.01,RIGHT16 ending<=.10,nonending category means<=5%,
freeflight8 presence>=90% each command. Retain every individual alarm and all
per-action/horizon reward,paddle,brick regressions. No easier support/best draw.

Full all-horizon physics,event timing/calibration,persistent brick memory,
late/tunnel/second-wall coverage,untouched families and fresh seed confirmation
remain required. Existing oracle has no above-top-brick endpoint. Intermediate
gates do not complete the goal. No unchanged extension,sweep,completed stage
repeat or automatic successor. AMD/Auto/existing host guard;inspect live identity,
stop real faults/nonfinite state,no reset/recovery/reboot/NVML. Root
`/mnt/data/kindle-breakout-world-image-feedback-20261108`. No GPU allocation yet.
