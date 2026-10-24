# Qbert's 2M world models predict useful rewards, including beyond the first pyramid

**All three 2M checkpoints beat zero, calibration-mean, persistence and unrelated
actions on h15 reward MAE and RMSE.** Their positive-event AUC is .951/.967/.973.
The retained500k models are much weaker on these same trajectories, especially
after the first pyramid. This supports useful short-horizon reward modeling,
not complete world-model quality or solved Qbert. Survival prediction and the
policy/value system remain open; no five-game quality target is closed.

[Fixed declaration](../experiments/2026-10-11-cdp-qbert-forecasts.md) ·
[Compact data](2026-10-11-cdp-qbert-forecasts.json) ·
[Full metrics/audit](../../runs/cdp-qbert-forecasts-repair-20261011.vzrMzTOu/forecast-review.json) ·
[Whole source videos and learning results](2026-10-11-cdp-qbert-budget.md).

## Same recorded actions, frozen models

The six captures finish October11 at17:16:46 UTC; the independent CPU review
passes at17:17:19. Six guards, identical real action/reward/boundary/frame traces,
paired pixel hashes and exact346 saved tensors/model pass. There are386,496
replayed diagnostic actions, **zero actor/readout updates**,24m16.84s capture
time and12 recorded allocation warnings. The CPU audit takes3.84s, apart from
waiting for capture. No worker remains from this diagnostic.

Each500k/2M pair re-observes its seed's entire recorded2M frozen-policy log,
including excess episodes and tails. Of the72 selected natural source episodes,
the first per stream supplies24 calibration episodes; the next two supply48
test episodes. At h1 there are91,679 test transitions with3,282 positive rewards;
h15 has5,693 endpoints with207 positives. No negative rewards occur. These are
three learner seeds, not thousands of independent learning replicates.

| Seed | h1 MAE,500k→2M | h15 MAE,500k→2M | h15 RMSE,500k→2M | h15 positive AUC,500k→2M | h15 positive events |
| --- | ---: | ---: | ---: | ---: | ---: |
|1009 |2.858→.647 |2.218→.900 |17.880→7.948 |.583→.951 |56 |
|2017 |3.941→.870 |5.137→2.283 |31.603→26.744 |.586→.967 |78 |
|3019 |1.935→.551 |2.614→.844 |33.675→14.213 |.916→.973 |73 |

Seed-mean h1 MAE falls2.911→.689; h15 falls3.323→1.342, paired change
**−1.981 [−2.854,−1.318]**. H15 RMSE falls27.719→16.302, paired
−11.417 [−19.462,−4.859]. Intervals use10,000 bootstrap resamples of the three
learner seeds and are coarse. Lower error does not establish a learning cause.

The stronger controls matter because most transitions have zero reward:

| Seed |2M h15 MAE | Unrelated actions | Zero | Calibration mean | Last reward | h15 summed-reward MAE,actual/unrelated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|1009 |.900 |1.449 |2.041 |5.298 |3.070 |12.445/15.914 |
|2017 |2.283 |3.146 |5.232 |8.444 |5.148 |16.781/21.448 |
|3019 |.844 |3.088 |3.493 |6.523 |4.526 |7.887/27.400 |

The paired h15 MAE difference versus unrelated actions is
−1.219 [−2.244,−.549]. Positive-event MAE is9.00/26.09/15.39 versus zero's
55.80/79.49/142.47; this is not merely better prediction of empty frames.
H15 summed-reward MAE and RMSE also beat all four controls in every2M seed.
At h1, actual actions do **not** beat unrelated actions on every measure:
1009's unrelated-action MAE is slightly lower. Do not overstate immediate
action sensitivity. Posterior reward estimates remain separate, easier outputs:
their h15-endpoint MAEs are.353/.730/.605 after seeing the real target image.

## Later-stage improvement, limited survival evidence

After the first pyramid, h15 positive-event AUC rises
.349/.443/.838→.916/.956/.964. Before that milestone, the old3019 model is
already strong; its MAE worsens slightly.324→.375 and AUC.9998→.9914.
The result is not uniform improvement of every phase. The common trajectories
come from the2M policies and include later states the weak500k policies rarely
visited; this is not an evaluation on each checkpoint's own state distribution.

Continuation has a separate interpretation correction. Native targets are
`discount*(1-terminal)`, with F32 discount1−1/333. Thus the originally declared
`1-head` score includes a discount floor. The original scores remain in the
full audit. The [separate CPU correction](../../runs/cdp-qbert-forecasts-repair-20261011.vzrMzTOu/terminal-calibration.json)
uses `1-clip(head/discount,0,1)`, with no fitting, new actions or changed model.
The public summarizer now requires that discount explicitly. Its corrected
results agree with the independent calculation; reward summaries are unchanged.

Corrected h1 game-over AUC is.944/.982/.858, but each seed has only16 test
terminals, below the declared20-event threshold. Brier improves over the
constant baseline in1009/2017, not3019. At h15 there are only3/2/2 terminal
endpoints, AUC.891/.567/.548; **survival forecasting remains unresolved**.
These are game-over labels, not every lost life or cumulative15-step death risk.

## Retained failure, runtime and scope

The first full attempt stops after40,789 actions at a diagnostic lifecycle
assertion: an unused actor episode was opened after a final natural reset,
then begun again for the next stream. The corrected first full capture
reproduces that entire saved prefix exactly and passes all eight streams.
Failed source/data/guard remain in
`runs/cdp-qbert-forecasts-20261011.NOdhNC3b`; the failed process did not reach
its final frozen-tensor comparison. No GPU fault is reported.

The original128-action smoke and failed40,789 actions are additional cost:
**427,413 total diagnostic actions**, zero actor updates. They add two and one
allocation warnings respectively, for15 across all attempts. Completed training
and frozen competence results are unchanged. The collected source, original
terminal scores, corrected source and independent CPU receipts are retained;
no numerical GPU gate was relaxed. The collection commit's CI335 passes.

Native7311547d/Meganeura31026833/Blade56f0565 is unchanged. The October11 upstream
review finds no identified fix for this ordinary F32 CDP path. Historical
training pins remain historical. Scalar re-observation is **not** the original
vector actor's categorical trajectory. Forecasts use one sampled latent path,
paired actual/unrelated draws within a checkpoint, not a predictive likelihood
or shared draws across checkpoints. Recorded future actions are prescribed
inputs, not future actions available to the live actor. No RAM/task labels,
future frames or rewards enter the model; phase labels are analysis-only.

## Next decision

Do not replace the representation or automatically extend training on the basis
of these results. The world model is useful on the tested reward task, while
Qbert's control/reliability gap remains. Next use the already implemented greedy
action mode in one bounded, three-seed frozen Qbert comparison, including actual
initial controls. Retain the sampled evaluations as the standard result. This
tests whether action sampling explains part of the gap before another learning
change; it is not permission to substitute greedy scores for online DreamerV3
parity or to relax a mastery gate. All five quality targets and the matched RGB
budget question remain open.
