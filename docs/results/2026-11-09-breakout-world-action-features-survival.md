# Action-feature world: survival is not the only failure

Holding the new rendered forecasts fixed, substituting empirical oracle survival
changes primary ball energy **.224881281067→.185872721847**. This is an
oracle-assisted diagnostic using future labels,not a learned repair or a new
passing score. The original failed assessment and all its gates remain unchanged.

[Protocol](../experiments/2026-11-09-breakout-world-action-features-survival.md),
[failed broader assessment](2026-11-09-breakout-world-action-features-branches.md).

## Fixed-image score substitutions

History16,category-balanced16/32 endpoints. Same-mean preserves the marginal
ending probability while removing association with image draws. Oracle-mean
uses realized future endings;all-alive deliberately ignores endings.

|Feature/scope|Original|Same mean|Oracle mean|All alive|
|---|---:|---:|---:|---:|
|ball/all_categories|0.224881281067|0.224832959867|0.185872721847|0.223819150419|
|ball/nonending|0.191209183623|0.191155993212|0.176639642951|0.176639642951|
|paddle/all_categories|0.085008099815|0.085020073124|0.015038813151|0.073678978990|
|paddle/nonending|0.052369835233|0.052383693925|0.017177089362|0.017177089362|

The ending distribution affects the score,but substantial image-feature errors
remain. Changing draw association alone has little effect here. These algebraic
differences do not identify causal fractions of a training failure or show
natural-frequency calibration. No acceptance gate is applied to substituted
weights,and no oracle label enters a model input or loss.

## Which categories remain weak

The following values keep each category's16/32 average and compare the new
oracle-assisted diagnostic with the already sealed one-hot diagnostic. The old
raw forecasts and fixtures are not rerun.

|Category|New ordinary|New oracle mean|Cached one-hot oracle mean|
|---|---:|---:|---:|
|serve|0.323263514|0.336614274|0.336614274|
|freeflight|0.056613727|0.039151545|0.030169009|
|contact|0.072289247|0.053369811|0.203290924|
|brick|0.088547388|0.061911223|0.008202375|
|wall|0.152206112|0.142236985|0.041325553|
|life_loss|0.454335113|0.426554020|0.025405091|
|ending|0.426913866|0.241271195|0.267341903|

The category named `life_loss` has the largest residual, with wall and brick
interactions also worse than the one-hot diagnostic. The subsequent
[edge-rebound audit](2026-11-09-breakout-world-edge-rebound.md) establishes that
NOOP, LEFT and reversal actually survive this slice; only RIGHT loses a life.
The earlier life-loss/respawn attribution is therefore unsupported. Contact improves strongly.
These are functional error locations,not proof of a data-coverage,representation
or optimizer cause. Supported-position counts condition on visible balls;they
can improve while disappearance/respawn or other image-distribution errors worsen.

All endpoints,plans,origins,ball/paddle features andboth histories remain in
`result.json`, `cases.npz`, `case-index.json` and `group-values.json` outside Git.
History32 is secondary and does not replace the primary history16 result.

## Integrity and scope

The CPU analysis reads only the new frozen exports,with312 pinned inputs.
All6,720 cases/840 groups and420 ending checks reconstruct ordinary per-origin,
group and primary scores within1e-12;maximum error is4.44e-16. The unchanged
four-term energy math and its sealed scalar-fixture receipt are reused without
rerunning completed fixtures. Exact source/helper identities and all input
hashes pass. Execution takes4.760s. No failure or repair in this stage.

Model constructors are trapped and the Vulkan loader is disabled. There are
zero new model queries,training updates,emulator actions orframes. Old-model raw
forecasts are not reread. The one-hot table above uses its existing aggregate
result only;it is a related reused development comparison,not confirmation.

Root `/x/Temp/kindle-world-action-features-survival-20261109`.
No new learner,collector,loss,backend work or automatic successor is selected.
The full Breakout world-model goal remains active and unproven,including events,
physical transitions,brick memory,late/above-top coverage,untouched families and
fresh confirmation. Further work must address phase transitions as well as
ending calibration;these evaluation substitutions alone repair neither.
