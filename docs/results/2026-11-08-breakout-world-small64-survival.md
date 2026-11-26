# Small RGB64 world: ending weights do not explain all forecast error

The CPU diagnostic is complete. Holding decoded paths fixed and substituting
empirical future ending labels changes primary ball energy **.172366→.130336**.
Serve, paddle-contact and ending-category image errors remain. This is an
oracle-assisted sensitivity, not a learned repair or a causal fraction of error.
The original small-world trial still fails retention; the full world-model goal
remains unproven. No successor learner or GPU allocation is selected.

[Protocol](../experiments/2026-11-08-breakout-world-small64-survival.md).
Root `/mnt/data/kindle-breakout-world-small64-survival-20261108`.

## Fixed forecasts, four survival choices

Use only the new1M/RGB64 final16,384 forecasts: all42 chosen plans, histories16/32,
eight model/oracle draws, five endpoints and both ball/paddle features. Reuse the
unchanged four-term energy formula and its sealed independent scalar fixtures;
no old model or completed diagnostic is rerun. No model, gradient, emulator,
collection or GPU call occurs. Native weights are unnecessary.

For each unchanged rendered sample, ordinary survival weights retain image/head
association. Same-mean makes those weights equal while preserving the ending
marginal. Oracle-mean uses realized future labels. All-alive ignores game overs.
Only the ordinary forecast is the model's actual output.

Primary history16, equally weighted seven categories and16/32 endpoints:

|Survival weights|Ball energy|Paddle energy|
|---|---:|---:|
|Ordinary forecast|0.172365704612|0.104890151016|
|Same mean on every draw|0.172372450457|0.104897744547|
|Oracle future-label mean|0.130335589942|0.026660532116|
|All alive; ignores endings|0.168282018513|0.085280825813|

The same-mean ball change is only+0.000006746. Changing this draw association does
not explain the ordinary gap. Oracle ending marginals change the score more,
but do not turn the fixed paths into correct physics. Oracle labels are neither
deployable heads nor an optimal population forecast; no ordinary gate is applied
to these altered scores. All-alive must retain its errors at true endings.

## Where fixed-image errors remain

Category-balanced16/32 ball energy, same primary history:

|Origin category|Ordinary|Oracle future-label mean|
|---|---:|---:|
|serve|0.371740969|0.336614274|
|freeflight|0.058677426|0.030169009|
|contact|0.209184094|0.203290924|
|brick|0.044468313|0.008202375|
|wall|0.066790657|0.041325553|
|life_loss|0.067190429|0.025405091|
|ending|0.388508044|0.267341903|

The altered score is concentrated in serve, contact and ending origins; freeflight,
brick and wall paths have smaller residual energy here. Categories describe
origin predicates, not guaranteed future events. These values do not identify a
missing architecture, data or optimization cause. Small energy also does not
establish one-pixel precision or full game coverage.

Across the predefined nonending categories, ball energy changes
.136341981435→.107501204445 with oracle labels. History32 gives all-category ball
.174074361555→.132782394234, so longer conditioning does not remove this pattern.
Every action/horizon/lane/history/feature and four algebraic terms remains in
`cases.npz`, `case-index.json`, `group-values.json` and `result.json`.

## Checks, support and consequence

All6,720 cases/840 groups reconstruct the canonical ordinary per-origin/group
scores and original ending probabilities/Brier within1e-12; maximum observed
error is4.441e-16. The ordinary primary is exactly
.17236570461161804. Analysis passes313 input pins in4.804316s;
closure verifies325 pins in1.635648s. Reused math
source matches its existing24-fixture receipt; those fixtures are not rerun.
CPU plotting uses retained aggregates only and shows no retention threshold.

There are1,680 distinct target origin/plan/horizon rows, not independent episodes.
All336 rows per horizon through16 have eight alive oracle outcomes. At32,
319 have eight alive and17 have eight dead; no mixed outcome appears among the
retained eight repetitions. This does not establish population determinism or
natural-frequency calibration. Detection is restricted and no above-top-brick
endpoint exists.

The next decision should inspect **conditional training support for serve and
contact transitions**, alongside event fitting. The paired fitting collector
already includes FIRE1, LEFTFIRE12 and RIGHTFIRE11 for serve-labelled parents;
absence of all launch data is not established. Category names and global action
counts cannot demonstrate coverage of a specific transition.
[Existing paired-data result](2026-11-07-breakout-world-paired-actions-data.md),
[collector plans](../../python/examples/breakout_world_paired_actions.py).

No new loss or learner follows automatically. The original short/long/event gates,
265/304 nonending alarms and every regression remain authoritative. Precise physics,
event timing/calibration, persistent brick memory, late/tunnel/second-wall states,
untouched families and fresh confirmation remain required.
