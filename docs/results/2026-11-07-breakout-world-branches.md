# Frozen Breakout action branches expose drift, serving and ending failures

The [motion-weighted pixel world](2026-11-07-breakout-world-motion-attention.md)
learns useful short motion and paddle responses, but **does not pass the world
goal**. Under commands fixed before the future unfolds, ball forecasts drift,
waiting-to-serve behavior is wrong, and last-life game-over is severely missed.
Doubling observed history16→32 gives no consistent rescue. No actor/critic or
world learner ran in this challenge. All frozen checks pass; retain the model.

The [predeclared development protocol](../experiments/2026-11-07-breakout-world-branches.md)
is complete and closed. Do not repeat its collectors/evaluations or extend the
learner. Next address behavior/episode coverage and causal forecast supervision;
this report does not select another learner or establish an architecture cause.

## What was measured

Seven groups each use eight seeded sources: serve, upward free flight, descending
paddle approach, upward brick approach, side-wall approach, descending multi-life
state and descending last-life state. Serve holds each of18 commands for32
decisions. Other groups hold NOOP/LEFT/RIGHT, or RIGHT4 then LEFT28. Commands do
not consult future observations. Eight oracle repetitions and eight fixed model
draws per origin/plan, horizons1/4/8/16/32. Keep sticky.25/repeat4/full18/native
pooled RGB/one GPU resize to64, batch8, exact native/runtime and frozen weights.

The actual-initial model uses history16; the final8,192-update motion model uses
primary history16 and secondary32. The model sees RGB and submitted commands,
without RAM, categories, emulator snapshots or future images. RAM selects/audits
sources. Source replay reconstructs ALE's private previous sticky action before
installing each desired physical/RNG snapshot; full original ALE state is checked.
Repeat0 uses original RNG, later repetitions continue the preceding oracle RNG.

These are fresh **development** sources, not an untouched test or natural-frequency
sample. Distinct native RGB contexts by group are **1/8/6/8/7/1/2**:33 across
groups, with duplicates retained rather than replaced. Eight seeded streams do
not imply eight distinct situations; repeated oracle/model draws are not extra
independent origins. This is an early-game challenge with few brick layouts.
Exact hidden-state conditioning in the oracle is stronger than finite RGB history.

## Learned behavior and remaining failures

Paddle control is broadly learned over these short branches. At8 decisions, the
RIGHT-minus-LEFT expected visible/alive paddle-x contrast is52.77 resized pixels
versus52.90 oracle for free flight; mean per-origin contrast error.291 pixels.
Errors are.267 for brick approach,.867 wall approach,.160 multi-life approach,
and1.956 paddle approach. This observable moment includes visibility/survival;
it is not a conditional-on-survival position estimate. Wrong ending predictions
therefore also corrupt the late last-life contrast.

The following secondary diagnostic compares every model draw with every supported,
alive oracle ball endpoint at the same origin. Missed detections remain failures.
It is a **draw-match frequency**, not a calibrated distribution score or a count
of independent examples. Oracle leave-one-out draws match within1 pixel100% for
free-flight/brick/wall cases and97.5% for supported paddle-approach cases at8 steps.

| Origin category |1-step within1 pixel|4-step|8-step|16-step|
|---|---:|---:|---:|---:|
| Free flight |100.0%|68.0%|59.0%|7.4%|
| Paddle approach |92.2%|53.1%|31.9%|21.3%|
| Brick approach |99.2%|18.4%|11.7%|3.1%|
| Wall approach |96.9%|40.6%|25.0%|15.0%|

Detected pair error for free flight grows.988 pixels at8 steps to3.379 at16;
wall approach grows2.216→3.386. At32 steps, free-flight detected error is18.801
pixels, with only64 supported oracle endpoints remaining. That small surviving
support cannot be generalized to every physical state. Paddle-approach detections
are only35.8% at8 steps, recovering to100% at16 with5.306-pixel error: both
visibility and trajectory timing/location can fail. Do not discard those misses.

Primary finite-mixture energy scores include an absorbing game-over atom, with
mass from the continuation head. Lower is better for the reported observable
forecast. Model samples are decoded mean RGB images, not independently sampled
Bernoulli pixels; these scores do not evaluate full image likelihood.

|8-step ball feature energy, equal plan mean|Initial16|Trained16|Trained32|Naive constant velocity|
|---|---:|---:|---:|---:|
| Serve |1.3153|.6093|.5484|.6499|
| Free flight |1.5552|.0095|.0139|.0353|
| Paddle approach |1.2315|.3309|.3092|.4554|
| Brick approach |1.6323|.0243|.0224|1.2970|
| Wall approach |1.4997|.0990|.1196|1.0821|
| Multi-life approach |.9926|.2211|.4011|.0000|
| Last-life approach |1.2629|.2381|.2423|.0297|

The detector measures a supported below-brick red sprite. Entering/leaving its
region causes a discontinuity: a zero constant-velocity score where both labels
are unsupported/absent is not perfect physical prediction. Brick-contact boundary
scores especially need the rendered images and supported-position diagnostics.
Initial values and all plan/origin scores remain in the retained JSON; no favorable
draw or checkpoint replaces the fixed endpoint. History32 helps some cases,
worsens others, and does not establish a memory-capacity explanation.

**Serving is wrong.** NOOP oracle branches do not launch. The model detects a
below-brick ball in95.3% of its8-step NOOP forecasts and98.4% at16. Conversely,
FIRE has no detected model ball at1 step, despite supported oracle launches.
The source history is31 NOOPs. A separate CPU corpus audit finds zero fitting or
development ball-absent sequences of16 non-FIRE decisions: fitting max11, with
only25 NOOP commands from a ball-absent state and max8 consecutive absent NOOPs.
Thus this is a demonstrated coverage gap, not proof of an architecture limit.
RAM absence is an analysis proxy, not an exact serving-state classifier.

**Game-over is badly underestimated.** From the multi-life sources, RIGHT loses
a life in64/64 branches but never ends the game; NOOP/LEFT avoid life loss. From
last-life sources, RIGHT ends64/64 by32 decisions, while NOOP/LEFT/reversal each
end24/64. These counts include repeated/duplicate origins. On RIGHT, the model
assigns mean game-over probability **.581% versus100% oracle**, Brier.988414.
Across the four last-life plans its mean probability is.989% versus53.125%
oracle. Predicted paddle movement does not imply learned life/ending dynamics.
This does not isolate missing lives information, forecast-state error, terminal
imbalance or head fitting; those remain distinct hypotheses.

Reward timing also needs work: positive-event Brier at8 steps is.21316 for free
flight; at32 it is.52451 for paddle approach. Alive-only brick Brier at8 steps is
.000009 free flight,.004178 brick approach and.002826 wall approach, on mostly
full early walls. These cell averages do not establish generalization to removed
bricks; the preceding dense-development balanced brick error remains.206631.
No stochastic calibration claim follows from eight repetitions or one class mean.

## Cost, identity and preserved failures

Root `/mnt/data/kindle-breakout-world-branches-20261107`, corrected execution under
`root/repair`. CPU oracle collection **298.770s**:

|Work|Actions|Emulator frames|
|---|---:|---:|
| Source collection |3,789|15,156|
| Oracle branches |83,976|335,784|
| Exact source replay |149,024|596,096|

Frozen initial evaluation131.115s/42 groups; final252.232s/84 groups. Other project
processes were present; these are elapsed costs, not isolated speed benchmarks.
Native/source3fa37b6d, SHA256
`eafe8359ae5019ea0a5a7517b7acfe75aacaf64b4e84a3bcd9ec4548cf939dc5`;
matching motion runtime/checkpoints unchanged. Python source780db35e, archive
`source-780db35e.tar.gz`, SHA256
`d7b582eccb282914ec4db3e828a5d2eb0edb319fabc81c442b98d0b98f1f16cf`.

Nine CPU tests pass. Corrected collection/audit checks all393 manifest files,
branch masks, terminal cutoffs, horizon frames and action accounting. All431
evaluation input pins pass at closure. Both guards and their nine-pin sealed
audits pass with no unsealed files, host recovery or new GPU/numerical failure.
All204 parameter/moment tensors remain exact in each before/after checkpoint;
update counters remain0 and8,192. No learning or repeated completed GPU stage.

Preserve three original failures: sticky-fixture raw-difference exposure assertion,
initial collector feedback without a proposal, first CPU audit's duplicate report
key. Separate CPU reviews verify16 replay-corrected trace pairs and repair the
audit report. Corrected collection uses separate outputs; the first attempt
completed no oracle branch and serialized no emulator state before its exception.
None of those failed attempts is relabeled as passed.

`closure-review.json`, `oracle-audit-repaired.json`, `forecast-analysis.json`,
`visual-diagnosis.json`, `serve-coverage.json`, per-origin arrays, forecasts,
`ball-energy.png` and `fixed-plan-forecasts.png` remain outside Git. The forecast
panel uses fixed oracle/model draw0, lane0, not matched RNG realizations; ended
oracle endpoints are marked explicitly. Native/RGB source artifacts are retained.

All phases close. The overall world-model goal stays active and unproven. Retain
pixel/motion H0 for diagnosis. A next experiment must separately declare how it
addresses the now-measured coverage/forecast failures; do not resume actor/critic,
repeat this challenge, silently train on it as an untouched test, or infer that
more warmup alone solves the world.
