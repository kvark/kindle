# Fixed visual functions preserve perception but do not solve Breakout forecasting

The [bounded staged experiment](../experiments/2026-11-07-breakout-world-frozen-visual.md)
is closed and **fails retention**. It preserves recorded observed/one-step accuracy and
improves four/eight-step tracking over the inherited H0 control, but remains weak
at longer horizons. Keep the coverage control; no extension, repeat or automatic
successor. The full world-model goal is unproven. Actor/critic and CDP comparisons
remain parked.

## What changed

Warm-start the retained H0 coverage world at 8,192 updates, preserving all 204
weight/moment tensors and optimizer age. Train 4,096 additional native updates
(global 12,288), using existing H16 motion-weighted RGB cross-entropy and fixed
encoder/posterior/decoder weights. All 32 dynamics/reward/continuation weights
change; all 36 visual weights stay exact at the intermediate and final checkpoints.
Their 72 moment tensors change. Derivatives through fixed functions remain;
their recurrent inputs can change, so this does not freeze each observation's
latent or reconstruction.

RGB64/12M/B8/T32/full BPTT/Auto, original 16 fitting/eight development streams,
rates/scales, seed 41001 and the first 4,096 rows of the original4/2/1/1 sampler are
retained. This combines warm initialization,H16 and freezing, and resets phase
RNG. It is neither uninterrupted training nor an isolated estimate of visual
drift, an independent learner or a matched-cost comparison with fresh H16.
No new loss, data, object/RAM targets, actor or critic enters learning.

The plan covers 32,833 of 32,850 stored arrivals, 488,356 valid H16 suffix targets,
5,419 observed terminals, 4,755 suffix terminals and 17,425 positive-reward rows.
No emulator action/frame is added; replay exposure is not new experience.

## Same original recorded cohorts

First fixed model draw, within one resized pixel on supported visible balls.
Related origins and varying horizon support are not independent episodes or a
calibrated distribution score. Fitting and development tables match the originals.

|Measure|Inherited H0 dev|Fresh H16 dev|Staged dev|Staged original fit|
|---|---:|---:|---:|---:|
|Observed|108/112|99/112|108/112|163/163|
|One-step|94/109|84/109|94/109|150/177|
|Four-step|33/109|47/109|51/109|77/180|
|Eight-step|14/107|25/107|24/107|40/176|
|Sixteen-step|3/114|10/114|4/114|18/171|
|Thirty-two-step|0/95|4/95|0/95|10/179|

Observed and one-step gates pass; eight-step misses 28/107. Four-step 51/109 exceeds
H0's 33/109 and fresh H16's 47/109. This is useful local progress, not accurate long
futures. Observed development brick balanced error is .224305 versus .223954 H0;
accurate ball reconstruction does not establish persistent brick memory.

## Same-origin alternative-action forecasts

All 84 new-final-only groups use the original oracle and fixed history 16 (primary)/
32 (secondary), eight model draws and all 1/4/8/16/32 endpoints. No old model is
queried again. Primary seven-category 16/32 survival-weighted ball energy is
0.204741, versus 0.169502 H0 and
0.311977 fresh H16; this is 20.79% worse than H0. The existing retention threshold
is .127127. Energy measures survival and observable uncertainty jointly; point
accuracy/detection cannot replace it.

Supported fixed-plan ball-pair accuracy also changes: 8: 22.45→45.12%, 16: 20.22→28.12%, 32: 5.90→2.04%.
These related model/oracle pairs omit survival weighting and cannot replace the
distribution score or ending gates.

At 8 in the eight retained freeflight histories, detected model balls for
NOOP/LEFT/RIGHT/REVERSE are 49/49/43/46 of 64 per plan;
the existing threshold is 90% for every plan. Oracle ball paths are identical
across those plans through 8. Detection alone does not establish correct location.

Ending 16/32 Brier is .011062/.179440; the RIGHT 16 false ending probability is
0.177495. Non-ending origin/plans above 5%
ending probability: 0/304; maximum
4.53%. The separate 5% category-mean false-alarm
gate and every other original gate remain in `branch-comparison.json`.

These are repeatedly examined development families. They cannot establish
population calibration, untouched generalization, full late/tunnel/second-wall
coverage or fresh-seed reliability. No above-top-brick endpoint exists in this
oracle. The goal remains open regardless of any intermediate gate.

## Execution, cost and preserved failures

Native source 7ed07b2b, SHA256
`3032c0a6c1c2c4f1fef64dbb5619fa3cd84af38600f99c060014ab7591b69e79`;
Python runner 442a2b45. Distinct `start_frozen_visual_phase` API and v3 checkpoints
preserve scope/age; default/all-weight v2 and ordinary restore behavior remain.
The old qualified native rejects v3 before GPU construction.

The original CPU graph fixture failed from an incompatible whole-agent replay
scale; corrected 20 CPU checks/four GPU checks ignored and strict core/binding
Clippy pass. Packaged 87 CPU checks pass. First native qualification loaded the
world then failed Python dictionary equality before prediction/save/update:
shortest F32 getter JSON differs from checkpoint F64-valued JSON. Preserve that
failed declaration, guard and log. The Python-only exact-F32 comparison repair
rejects one-ULP changes; 92 CPU checks pass. Saved phase metadata configuration
also matches source metadata exactly.

Separately declared corrected qualification completes in 21.571s: 353 pins, 204
exact inherited tensors, 18 initial/restore arrays,two finite updates including
restored scope,inference invariance and legacy rejection. Learning verifies 375
pins, all 4,096 eleven-loss/mask/total rows, 108 inherited initial evaluation arrays,
204 frozen inference tensors, 36 fixed visual weights,exact source plan and 18
restore arrays. All four successful host guards use Auto/AMD without resets,
recovery or NVML; original failed qualification remains failed.

New phase native calls 383.411s, measured train/evaluation
loop 535.628s; inherited H0 training adds 529.515s native/
796.900s loop. New phase cost does not include its 21.571s qualification or the
separate 259.785s branch assessment. Do not present the incremental cost as total learning
cost. Branch time and final seal are recorded in external receipts.

Root `/mnt/data/kindle-breakout-world-frozen-visual-20261107` retains protocols,
config/native/runner bytes,source tar,wheel,first failures,checkpoints,curves,
exports,guard evidence and CPU reviews. `comparison.png`,`learning-curves.png`
and three causal APNGs use retained arrays only, with fixed first origins/draws,
not best-match selection. Native oracle and RGB64 forecasts are resized only for
display. The packaged disposable 3.24GB build cache was released separately;
original build receipts remain unchanged. Initial figure generation lacked
Matplotlib; the separate isolated CPU analysis environment completes the figures
without changing the learner/runtime, as recorded in `analysis-invocation-notes.json`.
