# Qbert: bounded replay-value scale1 comparison

Completed November2:8192 actions/733 updates in57.202s;1024 frozen/.855s,
zero updates. All815 pins,357 initial/frozen tensors,47 pre-update events,replay
andguards pass; original main/collector succeed01:15:11.906/01:15:18.703 UTC.
Conservative projection25m36.196s clears30min; gameplay benefit remains unmeasured.
[Completed result](../results/2026-11-02-cdp-qbert-replay-value1.md). No full learner
or automatic successor is selected; do not repeat completed stages.

Selected November2 within the unfinished five-game goal. One bounded execution
and cost check is selected; a full learner requires a separate decision after
this check closes. No old experience or completed stage is reused.

## Question and fixed intervention

Does increasing the existing real-replay value loss scale from0.3 to1.0 improve
short-budget Qbert learning? The completed seed14009 control scores14,770.833
greedy and14,254.167 sampled, both24/24 first pyramids, in25m09.906s. It misses
the fixed15,000 mean gate. Earlier longer training worsened Qbert; this test
adds no parameters or scheduled updates.

The native447fe50 behavior objective weights replay value cross-entropy and slow
value regularization with this coefficient. The world objective uses the same
coefficient through a frozen critic, with `replay_value_gradient=true`. Thus
critic fitting and representation training both change. Targets still bootstrap
imagined returns; this is not new Monte Carlo supervision or an isolated critic
intervention. Prior critic diagnostics do not prove a missing-loss defect.

Only `loss_scales.replay_value` changes. New optional fresh-only CLI
`--replay-value-scale 1.0` preserves native defaults when omitted and rejects
restore/evaluation overrides. The private control runner receives exactly the
same ten added lines as production; no other runner behavior changes. Reusing
seed14009 makes this a development comparison, no longer held-out confirmation.

## Fixed recipe and budgets

- 12M64/Auto/N8/B16/T16/fullBPTT and microbatch16/H5/R24;16,334,353 parameters.
- Base/encoder/effective behavior4e-5; dynamics1e-4; greedy collection and sampled
 posterior; actorunimix0; intrinsic/extrinsic1; replay100k and warmup1000.
- Expert100, all-unit action weights, world gradients, all-state corrective
 labels;20% continuation-guided episodes, cap1920. Actual override fraction is
 measured; guided online scores are not frozen competence.
- Full18 actions, sticky.25, repeat4, noops0, pooled native RGB with one GPU resize.
- Training seed14009, frozen seed4000014009; fixed eight lane seed offsets.
- Bounded8192 actions/733 updates, first update376, residual credit.75,180s cap.
- Frozen greedy1024 actions/zero updates,60s cap. No completed episodes required;
 this arm establishes execution and invariance, not gameplay quality.
- Audit180s; main service480s; terminal observer600s. Original failures remain;
 no GPU repeat, cap increase, coefficient sweep, extension or automatic successor.

Native447fe5013b6442c477fc7bfef7eab4122082bc54, binarySHA35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2;
qualified runtime `/x/Temp/kindle-qbert-color-masks-20261101/runtime`,
teacherSHAfa261c53991c329236bcc167745f309fe6e568547fe2d8c6e29da3d5bc7c3e5d.
Control runner e161bf38 becomes e304af89 solely through this CLI patch.
No native build or backend optimization is selected.

## Required checks and decision

Require exact declared configuration and cadence,733 finite reports, aligned
world/behavior expert reports, changed CNN/reward/actor/critic parameters,
357 finite initial/final tensors,357 control-matching initial tensors, matching
pre-update semantic events, and357 unchanged frozen tensors. Require full
recorded action/pixel/label replay, exact package/source pins, host guards and
original terminal identities. No benefit follows from finite loss alone.

Conservative cost gate: `training_seconds * 200000 / 8192 * 1.10 <= 1800`.
A passing result permits consideration of a separately declared fresh200k /
18,715-update learner with100k/9,340 recovery and final/actual-initial sampled
and greedy24-episode assessments. It does not itself select that learner. Keep
Qbert's15k/90% first-pyramid gate and report cost and initial-subtracted return
per minute against14009. One reused learner cannot establish seed reliability.

CPU qualification:27 focused tests pass in production and the exact private
runner; two actual CLI commands,ten config checks,357 shapes and the full cadence
derivation pass. All768 control pins remain unchanged. Initial test fixtures
reused an output path; that failure and separate fixture repair are retained.
The first source-copy script stopped on a two-line anchor containing an optional
centroid flag absent from the old runner; the ten completed controller copies
remain, and a separate unique-anchor runner assembly completes. No GPU work
occurs during either failure or repair.

Evidence root `/x/Temp/kindle-qbert-replay-value1-20261102/native-check`;
trial `qbert-12m-64-b16-h5-r24-dynamics1e4-replayvalue1-expert100-check-seed14009`;
main service `kindle-qbert-replay-value1-check-20261102`. Check live state before
work. Do not launch another GPU allocation while this service is active.
