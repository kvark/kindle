# Frozen representation and actor comparison after the encoder4e-5 gain

The completed [encoder4e-5 learner](../results/2026-11-01-cdp-breakout-encoder4e5.md)
improves compact Breakout sampled25.292→56.292 and greedy42.625→104.667 in
21m28s versus22m27s. Select a short frozen diagnostic to determine whether
accessible ball/motion information and teacher-action agreement improve on the
same states. This is architecture diagnosis within the existing five-game goal;
no learner,loss variant or new gameplay evaluation is selected.

Reuse the original6e-6 control's exact unaided final-greedy corpus:20,960 recorded
actions/83,799 frames,with teacher labels added offline. Preserve every-eighth-row
sampling and954/509/601 train/validation/test rows on lanes0–3/4–5/6–7. Retain
all targets,feature variants,missing-value masks,per-lane and136 near-descending
test-row results. No rows,targets,penalties or decoder variants are selected
using the new test results. Reuse the original and2e-5 saved captures,predictions
and fits; do not repeat their GPU stages or refit their readouts.

The new explicit`--compare-encoder4e5` profile permits only source encoder6e-6
versus checkpoint4e-5; every other config field must match. Profiles are mutually
exclusive,including the retained2e-5 and centroid100 options. Restore must exactly
match the new checkpoint configuration. Native447fe50,runtime and F64 actor
probability/metric functions stay unchanged. No teacher target enters the agent
or replay. The selected checkpoint is the fixed200k final,not a chosen interim.

Run three distinct guarded stages:128-action feature compatibility under120s;
128-action native actor-probability check under120s; one complete capture under
300s. Each must preserve357 tensors and zero learner updates. Require exact
input-pixel/action/reward/reset/frame replay,finite readbacks and unchanged
state/probabilities/counters at existing checks. The actor check requires128
matching greedy choices,maximum CPU/native probability error1e-4,and exact
equality to16 prior selected feature rows. Reuse qualified control actor evidence
after exact source/provenance checks; do not repeat completed native work.

CPU state and actor analyses keep training-only normalization,training-derived
kernel bandwidth and validation-only ridge selection. Each saved learned actor
is applied to its own recurrent states. Reproduce the original control metrics
exactly. A third CPU comparison aligns the new and saved2e-5 features by lane,
index,split,phase and targets,requires unchanged control/pixel metrics and reports
all three rates together. Compare position on exactly shared pixel-supported
rows. Preserve state predictions,actor probabilities and action-readout fits.
Each CPU analysis has120s; whole service900s.

CPU preparation passes81 scoped tests plus two independent pure regression
fixtures,with zero native/emulator import attempts. Preflight5.237s verifies
three actual command parsers,six combined-profile rejection cases,seven actual
config cases,55 original representation/289 actor/358 prior encoder/227 learner
pins,exact checkpoint/native identities and unchanged splits. Historical source
pins resolve to exact Git bytes without changing the original declarations.
The three F64 actor functions and two numerical constants match the previously
qualified source exactly. No native build or broader qualification is needed.

Stop on a failed stage and preserve evidence;no GPU retry,cap increase,training
extension or automatic successor. Decoder error does not prove information
absent,and improved readouts do not establish causal policy use. Pixel-teacher
labels are estimates; teacher agreement is not optimal-action truth. Scalar
posterior draws differ from original vector acting. These reused control states
do not measure the stronger policy's own-state coverage or independent learner
reliability. Gameplay improvement comes from the completed full comparison.

Root`/x/Temp/kindle-encoder4e5-representation-20261101`,service
`kindle-encoder4e5-representation-20261101`. New checkpoint metadata identity
`638cc43afd39fef14f6943242f6ee59dfd9d0e20c034c96df0880c794048f26b`.
Use`/x/Temp/kindle-breakout-corrective-20261028/runtime`;check live state and
host boot/driver pins before selection. No new learner is selected.
