# Saved forecasts separate launch blur from premature ending risk

The CPU review closes without another model or emulator query. Serving still
loses coherent moving detail; selected ending forecasts accumulate substantial
risk before any real game over. Longer observed history does not repair either
problem. This is evidence about where forecasts fail, not their unique cause.
All ordinary physics/event gates remain failed and the full goal remains unproven.

[Protocol](../experiments/2026-11-10-breakout-world-rgb-event-heads-timing.md).
Root `/x/Temp/kindle-world-rgb-event-heads-timing-20261110`.

## Ending timing

The table uses primary history16, averaging eight origins and eight related
draws. A crossing is the first submitted action reaching that mean cumulative
risk; “never” means it does not cross within32 actions.

| Ending plan | Model reaches5% /50% | Oracle reaches5% /50% | Model /oracle risk at16 | Model /oracle risk at32 |
|---|---:|---:|---:|---:|
|NOOP|5 /11|17 /never|.709560 /0|.959818 /.375|
|LEFT|6 /28|17 /never|.234626 /0|.577934 /.375|
|RIGHT|8 /12|17 /17|.818743 /0|.986459 /1|
|RIGHT then LEFT|8 /28|17 /never|.181668 /0|.593592 /.375|

Every RIGHT origin reaches50% predicted risk at10–15, before the oracle's17.
The mean curve crosses at12. Under NOOP, the five origins that never end in
the32-step truth nevertheless cross50% predicted risk at9. Per-origin curves,
all crossings and never cases remain retained. The largest origin's mean
conditional hazard before17 is.373501 under RIGHT,.248923 under NOOP,.040704
under LEFT and.090679 under reversal: premature cumulative risk is not solely
an accumulation of numerical-sized probabilities. Conditional hazards are not
the same statistic as a survival-weighted marginal hazard.

History32 gives16-step mean risks.777253/.314174/.828578/.199701 for
NOOP/LEFT/RIGHT/reversal, versus zero truth. It does not correct the timing.
There is no newly discovered discount, label or scorer bug: every stored
survival, reward aggregate and endpoint ending/Brier summary reconstructs
exactly using the original float32 gamma convention.

The short ending images can be substantially right while survival is wrong.
For example, NOOP and LEFT have512/512 supported model/oracle ball pairs
within1px at1 and4,192/192 at8 and320/320 at16. RIGHT has512/512 at1 and4
and192/192 at8; the fixed instrument has no supported ball pairs at16.
These related pair counts are neither independent trials nor whole-scene
physics/calibration proof.

## Serving and input history

For FIRE1, primary-history detection is512/512 supported pairs at1, with only
116 within1px and detected RMSE6.440px. At4,8 and32 it is0/512 detected.
RIGHTFIRE11 has0/512 at1,4,8 and32. Both have only8 supported pairs at16,
also with no detections; low16-step aggregate image-feature error must not be
read as successful launch physics. Missing support can be a ball outside the
fixed instrument's region. The FIRE and RIGHTFIRE fixed examples show diffuse
ball-like fragments rather than one sharply advancing ball.

FIRE1 mean32-step ending risk is.039886 (history32:.054894), RIGHTFIRE11
.075017 (.087485), against zero oracle. The latter's mean first5% crossing
is27/25 for history16/32. All18 commands, both histories, every draw/origin and
five endpoints remain numerically retained; these two images are preselected
examples, not the aggregate serving assessment.

All eight serving origins contain a single exact native RGB frame repeated
for their entire16/32-input history. Ending origins have11–13 distinct frames
within16 and27–29 within32, with longest identical runs4–6 and lives2→1.
Thus these ending inputs are not entirely static. This metadata does not prove
information absence, establish a hidden-clock cause or show that differences
survive RGB64 resizing. Lives/positions are diagnostic labels only.

All six fixed parent0/modeldraw0/oracledraw0 figures are reviewed. RIGHT32
oracle RGB is black padding after terminal; it is not evidence that the physical
screen becomes black. Numerical point comparisons correctly exclude invalid
and post-terminal truth. Image titles' alive counts describe the aggregate,
not the validity of every displayed pixel. No image is rerendered or selected
after inspecting its apparent quality.

## Integrity and next question

Source`17ae22c914372afff2c8137d778dd3393c7e5db0` declares226 inputs. Analysis
takes5.070215s, retaining22 plans/44 plan-history groups,352 complete origin
curves,1,760 endpoint rows,220 probability/Brier joins,32 prefix records andsix
figures. Maximum reconstruction error iszero, bound1e-12. All226 hashes pass
before/after and at closure. The separate13-pin closure declaration summarizes
existing tables and seals23 files. Constructor traps/Vulkan-disabled analysis
performs zero model/GPU/emulator queries or optimizer updates. No failure,
retry or old stage rerun occurs. Original service completion is established by
earlier sealed invocation-specific journals; current transient units are unloaded.

The retained [launch-coverage audit](2026-11-08-breakout-world-transition-support.md)
already finds missing clean-wait aliases in the unchanged natural plan, and the
[corrective launch learner](2026-11-08-breakout-world-launch-corrective.md) fails
broad retention. Do not recount or repeat them. The [sampled-prior loss review](2026-11-08-breakout-world-prior-pixel-loss-review.md)
already explains an averaging incentive; it does not identify every present error.

Select a separately qualified [cumulative-survival loss](../experiments/2026-11-10-breakout-world-survival-loss.md)
as a targeted event-objective experiment. Existing suffix BCE scores each
continuation; the proposed additional Brier term scores their cumulative product.
It introduces no oracle corrections to forecasts and may not repair images.
No new learner, collection, extension or automatic successor is declared here.
Full physics/serve/collisions, natural calibration, brick memory, late/tunnel/
second-wall/above-top coverage, untouched families and fresh confirmation remain.
