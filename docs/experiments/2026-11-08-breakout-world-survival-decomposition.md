# Breakout: saved-forecast survival decomposition

The [local-action world](../results/2026-11-08-breakout-world-local-action.md)
improves supported ball-position accuracy but fails primary error,event timing,
observation precision and other predictions. Select a short CPU-only examination
of the saved score,holding images fixed. No new learner,native query,gradient,
emulator,collection or GPU allocation. Existing closed experiments stay closed.

Use exactly the retained final coverage H0 and local-action worlds,all42 fixed
plans,histories16/32,eight model/oracle draws and1/4/8/16/32 endpoints. Match cached
truth arrays exactly;retain every lane/category/horizon and ball/paddle result.
Use cached model features/survival,with truth features from the pinned existing
RGB64 detector. No detector,model-input,source or outcome-mask change. Native
weights and dense training exports are unnecessary. Root
`/mnt/data/kindle-breakout-world-survival-decomposition-20261108`.

Before analysis,pin exact scripts,helper/runtime/source bytes,source result and
closure receipts and all read NPZ/JSON files. Use the installed CPU NumPy/Pillow
stack;disable the Vulkan loader. Source inspection must show no constructor or
model/emulator call. Verify small independent scalar fixtures,normalization,
endpoint behavior and joint draw permutation invariance. Stop on disagreement;
preserve failure and any separately justified repair. Do not adjust tolerances
or settings after seeing the result.

For each origin,let x_j be the unchanged live image-derived ball/paddle feature,
s_j its saved survival and D=8. Add absorbing ending atom e=(1,0,...). The ordinary
forecast assigns w_j=s_j/D and t=1−mean(s). Reconstruct the energy as four terms:
image/truth cross distance,ending/truth cross distance,minus half the weighted
image/image spread,and minus weighted image/ending spread. Require agreement
within1e−12 with the canonical energy function and every retained per-origin and
aggregate ordinary score. The reconstruction must reproduce H0 primary
.16950221908977256 and local action.24056568347464044 within1e−12 before interpreting
counterfactuals. Keep terminal arrivals and the original absorbing truth rule.

Compare exactly four survival choices without changing x or truth:

- Original s_j,retaining any draw-wise geometry/survival association.
- Same mean:every s_j=mean(s),preserving the marginal ending probability while
  removing its draw association with rendered geometry.
- Oracle mean:every s_j=the empirical oracle alive fraction for that origin and
  horizon. This uses realized future labels and is an oracle-assisted diagnostic.
- All alive:every s_j=1. It deliberately ignores endings. Report all categories
  and the predefined six nonending categories separately;never present it as a
  deployable head or silently exclude ending failures.

Report original→same-mean and same-mean→oracle-mean differences as algebraic score
changes,not percentages of a causal training failure. Four-term contributions
must sum to their energy within1e−12. Verify same-mean ending probabilities and
Brier match the original;report sample support and oracle outcome variability.
Save every case and per-category/action/horizon/feature/history aggregate. Keep
category-balanced16/32 primary aggregation unchanged;also report all endpoints
and the predefined nonending subgroup. Do not select models,draws or thresholds.

A figure may compare these diagnostics but must label oracle assistance and
ignored endings clearly. Do not draw the original passing threshold over the
counterfactual scores. Preserve the local-action original failed gates and every
alarm/regression;these rescored values cannot replace any retention criterion,
show natural-frequency calibration or prove full physics. Sampled sticky-action
futures and related origins remain limited development evidence.

Publish a compact result in docs/results,update the roadmap/handover and PR31.
No new loss,learner,automatic successor or change to original requirements follows
from this diagnostic. Full physics,event timing/calibration,persistent brick
memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation
remain required. No reset,recovery or NVML work.
