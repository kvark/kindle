# Breakout: survival weighting explains a score gap, not the full forecast failure

The CPU-only diagnostic is complete. Ending probabilities impose a large numerical penalty on the local-action score, especially for the paddle. Removing that penalty with future oracle labels does not establish a usable world model: ball geometry remains inaccurate, and the original failed gates remain failed. No new learner or GPU allocation is selected.

[Protocol](../experiments/2026-11-08-breakout-world-survival-decomposition.md).
Root `/mnt/data/kindle-breakout-world-survival-decomposition-20261108`.

## Images held fixed

Read only the retained final [H0](2026-11-07-breakout-world-coverage.md) and [local-action](2026-11-08-breakout-world-local-action.md) forecasts. For each unchanged live image feature x_j, ordinary weight is survival_j/8; remaining mass belongs to an absorbing ending state. Compare saved per-draw survival, a shared mean preserving total ending probability, the empirical oracle mean, and all-alive weights. Oracle mean uses realized future outcomes; all alive ignores endings. These are diagnostic substitutions, not learned forecasts.

History16 is primary and history32 is secondary. Tables keep the original equal-category16/32 averaging. Lower finite-feature energy is better. No retention threshold is applied to a counterfactual score.

| History / feature / scope / model | Original | Same mean | Oracle mean: assisted | All alive: ignores endings |
|---|---:|---:|---:|---:|
| 16 / ball / all7 categories / H0 reference |0.169502219|0.169500875|0.159389839|0.197336267|
| 16 / ball / all7 categories / Local action |0.240565683|0.240564146|0.161657998|0.199604427|
| 16 / ball / predefined6 nonending / H0 reference |0.183738553|0.183736978|0.184266569|0.184266569|
| 16 / ball / predefined6 nonending / Local action |0.218725002|0.218722926|0.154725987|0.154725987|
| 16 / paddle / all7 categories / H0 reference |0.021643457|0.021643749|0.004713006|0.063335391|
| 16 / paddle / all7 categories / Local action |0.147579095|0.147581851|0.002989523|0.061616100|
| 16 / paddle / predefined6 nonending / H0 reference |0.005424692|0.005425100|0.005412491|0.005412491|
| 16 / paddle / predefined6 nonending / Local action |0.134833312|0.134836534|0.003380189|0.003380189|
| 32 / ball / all7 categories / H0 reference |0.181091624|0.181128086|0.169840693|0.207515472|
| 32 / ball / all7 categories / Local action |0.251191417|0.251199194|0.158731655|0.196678084|
| 32 / ball / predefined6 nonending / H0 reference |0.195632876|0.195664282|0.196424185|0.196424185|
| 32 / ball / predefined6 nonending / Local action |0.226270277|0.226279176|0.151305817|0.151305817|
| 32 / paddle / all7 categories / H0 reference |0.024162475|0.024163272|0.004887677|0.063510200|
| 32 / paddle / all7 categories / Local action |0.168436065|0.168434389|0.005038273|0.063664750|
| 32 / paddle / predefined6 nonending / H0 reference |0.005631388|0.005632247|0.005615787|0.005615787|
| 32 / paddle / predefined6 nonending / Local action |0.154249234|0.154247284|0.005774050|0.005774050|

All-alive and oracle-mean scores coincide on the predefined nonending categories because every oracle sample stays alive there. The categories are fixed by the original challenge, not chosen from model outcomes.

## What the substitution shows

Changing the association between survival and image draws while preserving their mean has a tiny effect. Changing marginal ending mass to oracle values has a much larger effect. These are algebraic changes to this finite score, not percentages of a causal training failure.

| History16 / feature / scope / model | Same mean − original | Oracle mean − same mean |
|---|---:|---:|
| ball / all7 / H0 reference |-0.000001344|-0.010111037|
| ball / all7 / Local action |-0.000001537|-0.078906148|
| ball / nonending6 / H0 reference |-0.000001575|0.000529591|
| ball / nonending6 / Local action |-0.000002076|-0.063996939|
| paddle / all7 / H0 reference |0.000000292|-0.016930743|
| paddle / all7 / Local action |0.000002756|-0.144592328|
| paddle / nonending6 / H0 reference |0.000000408|-0.000012608|
| paddle / nonending6 / Local action |0.000003222|-0.131456345|

For local action, primary ball energy changes.240566→.161658 with oracle survival; H0 changes.169502→.159390. The gap between models shrinks, but corrected ending weights alone do not produce precise ball forecasts. The retained oracle entropy reference is.004813 for ball and.000488 for paddle across these endpoints. It is a finite-sample reference, not a population calibration result or a new gate.

On predefined nonending cases, local-action ball energy changes.218725→.154726 and paddle energy.134833→.003380. H0 oracle-weighted values are.184267/.005412. Thus the local-action paddle score regression is highly sensitive to ending mass despite fixed rendered geometry. This does not prove that changing the continuation head alone would produce those images or achieve those scores after training.

The original local-action world still has observed precision98/112, primary error.240566, ending16/32 Brier.094168/.208852, RIGHT16 risk.507002 and244/304 nonending alarms>5%. Those results and all action/horizon regressions remain unchanged. Neither model is reclassified as passing.

## Oracle support and limits

There are1,680 unique oracle origin/endpoints across42 plans,8 lanes and5 horizons. Related plans and repeated histories share origins; these are not independent episodes. All336 endpoints at each of1/4/8/16 have8/8 alive oracle draws. At32,319 have8/8 alive and17 have0/8 alive; none has a mixed ending outcome among the eight recorded repetitions. This does not establish deterministic population futures or natural-frequency calibration. The finite challenge shows no sampled ending-outcome ambiguity that could explain the early false endings.

Ending probabilities read RSSM state, while local-action images also read local history. That information split is a structural fact, not a causal diagnosis. This diagnostic leaves both event prediction and physical image forecasting as training targets. It does not select a new loss or learner. Precise full-game physics, event timing/calibration, persistent removed-brick memory, late/tunnel/second-wall coverage, untouched families and fresh confirmation remain required. Existing oracle has no above-top-brick endpoints.

## Reconstruction and artifacts

All523 pinned inputs pass, including420 cached forecast files matched to the closed local-action receipt. Twenty-four independent scalar fixtures pass at1e−12; their maximum error is2.22e−16. Ordinary per-origin/group reconstruction maximum error is4.44e-16; ending probabilities agree exactly. H0.16950221908977256 and local-action.24056568347464044 primary scores reproduce. All four component terms sum to their scores; same-mean ending probabilities and Brier are unchanged.

`cases.npz` and `case-index.json` retain all13,440 cases, four survival choices and four signed terms; `group-values.json` retains1,680 action/horizon/feature/history aggregates. `result.json` contains every category/horizon and the original failed gates. `interpretation.json` retains signed score changes and oracle support. `comparison.png` labels the assisted substitutions and applies no passing line.

Analysis wall time is41.259s, including initial/final input checks; declaration hashing, plotting and publication are separate. No model/emulator construction, native query, training update, new action/frame or GPU allocation. CPU environment:Python3.13.13,NumPy2.5.2,Pillow12.3.0;offline figure usesMatplotlib3.11.2/NumPy2.5.2. Source/runtime identities are pinned, Vulkan loader disabled. No failed diagnostic stage or repair occurred. Earlier failures remain in their original reports.
