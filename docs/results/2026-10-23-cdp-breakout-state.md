# Breakout: learned state is useful; longer training does not sharpen forecasts

The12M CDP model retains ball/paddle information and learns action-dependent
predictions. The8M warm-start lineage is not clearly better than the2M source.
This rejects the simple diagnosis that spatial information is entirely missing;
it does not prove sufficient precision for good control or isolate the plateau.

Held-out readout R², shown as **ball x / ball y / paddle x**:

| State | Actual initial | Source2M | Warm-start8M |
| --- | --- | --- | --- |
| CNN |.500/.555/.876 |.616/.687/.931 |.689/.611/.901 |
| Posterior |−.389/−.307/.242 |.796/.751/.843 |.772/.748/.848 |
| One-step prior |−.289/−.308/.222 |.766/.753/.821 |.672/.760/.858 |
|15-step prior |−.398/−.271/.253 |.269/.590/.739 |.227/.551/.772 |

Source15-step coordinate RMSE is35.75/17.11/25.62; final36.78/17.90/23.95.
These are RAM-coordinate units, not rendered64×64 pixels. Both beat the fixed
training-mean control41.82/26.80/50.27 and decoded-posterior persistence.
Using unrelated actions degrades paddle RMSE to56.34/55.88. Ball differences
are much smaller; do not claim strong action sensitivity for every target.
The actual-initial prior has no useful ball R², unlike the trained models.

All three pixel readouts are identical: R² approximately0/.010/.979. The simple
pixel MLP underfits the ball, so this is not evidence that the GPU image loses
it. The CNN readouts recover substantial ball information from those images.
Trained-state training-set R² is near1 versus weaker held-out performance:
readout generalization and random-trajectory coverage limit causal conclusions.
These are not independent actor seeds or competence scores.

## Scope and checks

Each frozen model sees the same eight4096-action random trajectories with fixed
train/validation/test splits. Five native GPU heads/model use hidden128/batch64,
2048 updates/head and validation-only selection. RAM is diagnostic only, never
actor input, reward or actor training. Checkpoint-derived widths are2048
deterministic,1024 CNN and2560 latent; all2048 h1 deterministic coordinates
match exactly in each captured check, within the unchanged2e-5 gate.

All98,304 action/label/reward/reset traces agree across models. Three guards,
three exact346-tensor frozen comparisons, source/file receipts and independently
recomputed held-out RMSE/R² pass.30,720 readout updates, zero actor updates.
Captures take140.45/139.69/146.28s. GPU ends October23 03:51:03UTC; CPU audit
ends03:51:15. Service finishes successfully in7m36s; no probe worker remains.

The preceding384-action smoke/160 readout updates also passes. Its original CPU
auditor imported the historical helper; preserve that failed service and source.
A corrected import-order CPU audit passes without GPU repetition, and the full
controller uses that correction.80 CPU probe tests pass. No native rebuild,
backend optimization, numerical relaxation or probe architecture sweep.

## Training decision

Keep CDP and the current encoder. Do not extend the failed Breakout lineage
again unchanged or blame absent pixels. Future Breakout work should test a
specific learning hypothesis—such as longer temporal context—against the fixed
2M control, retaining sampled evaluation and checking both forecasts and score.
This report does not allocate that experiment or establish its benefit.

The next active training block is the fixed Freeway exploration replication
on1009/2017, addressing the remaining reliability hole across the five games.
All published targets and broader supremacy remain unmet.

- [Full audit](../../runs/cdp-breakout-state-20261023.PLPVSIid/full-review.json)
- [Source metrics](../../runs/cdp-breakout-state-20261023.PLPVSIid/source/result.json)
- [Final metrics](../../runs/cdp-breakout-state-20261023.PLPVSIid/final/result.json)
- [Untrained metrics](../../runs/cdp-breakout-state-20261023.PLPVSIid/initial/result.json)
- [Fixed plan](../experiments/2026-10-23-cdp-breakout-state.md)
