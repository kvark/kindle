# The largest state-readout gap is inside the RSSM

[Configuration, curves, controls and audits](2026-10-04-rssm-belief-probes.json)
· [Predeclared protocol](../experiments/2026-10-04-rssm-belief-probes.md).

## Decision

**Useful visual state reaches the adapter, but horizontal position becomes
poorly readable from the actor's posterior belief on held-out trajectories.**
Vertical position survives better. This localizes a weakness under the fixed
probe recipe; it does not prove that information is absent or that JEPA cannot
work. No gameplay competence or online efficiency advantage is established.

Keep learned RGB as the qualified 2D default, target standardization opt-in and
`ac_grads` default-off. Before more joint gameplay, propose one frozen-corpus
ablation: predict normalized current Tiny features from the full posterior,
alongside the existing future-prediction objective. This tests a direct
observation-learning signal without reconstructing RGB. **Not launched here.**

## Where state remains readable

Three frozen standardized RSSMs, seeds 1009/2017/3019, replay the same saved
eight-trajectory corpus used in the preceding target-scaling test. Eight small
native GPU readouts per model fit current state or causal h1/h15 forecasts.
The current-state table compares identical held-out labels; the Tiny readout
is reused. Higher R² is better; zero is test-mean prediction quality.

| Current representation | Width | Player x R² [95% CI] | Player y R² [95% CI] |
| --- | ---: | ---: | ---: |
| Tiny features | 3,136 | .784 [.750,.804] | .833 [.788,.860] |
| Learned adapter | 196 | .585 [.501,.707] | .765 [.699,.820] |
| Full RSSM posterior | 640 | .003 [−.189,.088] | .554 [.466,.614] |

Posterior minus adapter is **−.582 [−.749,−.466]** for x and
−.211 [−.277,−.164] for y. All three models show the horizontal gap.
However, posterior x training R² is .278/.331/.215 versus held-out
−.006/.011/.004: there is a generalization gap, not proof of complete erasure.
One head architecture, initialization and budget cannot exclude a stronger
probe. Different input widths also imply different probe parameter counts.

These are current-state estimates, **not forecasts**. The full posterior
contains 512 deterministic values and 32×4 sampled categorical values.
The future-feature predictor consumes only the deterministic prior.

## Forecasts, transfers and controls

| Forecast representation, independently fitted head | Horizon, actions | Player x R² | Player y R² |
| --- | ---: | ---: | ---: |
| Deterministic prior | 1 | −.033 | .525 |
| Full prior | 1 | −.057 | .473 |
| Predicted Tiny features | 1 | .025 | .507 |
| Deterministic prior | 15 | −.027 | .141 |
| Full prior | 15 | −.069 | .057 |
| Predicted Tiny features | 15 | −.092 | .081 |

**The earlier transferred Tiny readout understated surviving vertical
information.** On the same h1 predicted features, its y R² is .046, versus
.507 for a stage-fitted readout: paired gain +.461 [.427,.501]. Horizontal
readability remains weak. At h15, transferring the posterior head onto full
priors gives y R² −.459 versus .057 for a fitted prior head. Transfers alone
confound representation quality with distribution shift. The previous report's
numbers remain valid for its transferred readout, not as proof that predicted
features contain no useful state.

There is also **some action signal**: h15 deterministic-prior fitted readouts
have actual/unrelated-action MSE ratios .862 [.835,.886] for x and
.900 [.868,.929] for y. Nevertheless, their errors are **2.562×**
[2.392,2.696] and **2.144×** [1.681,2.595] Tiny persistence. Persistence has
h15 x/y R² .599/.599; real future Tiny features score .793/.837. Thus the
model is not wholly action-blind, but its absolute state forecasts remain poor.
All controls use identical forecast origins and paired categorical draws.

Reward-event readouts are inconsistent across seeds; no robust reward forecast
is established. Test slices contain 95 rewards/21 terminals among 12,288 current
outcomes, 27/3 among 3,072 h1 origins and 18/7 among 2,991 h15 origins. These are
the same events across models, not independent repetitions. Missing position
labels are masked. RAM is diagnostic-only; bullets are not labeled.

## Scope, cost and integrity

Each model replays 32,768 recorded actions: three training, two validation and
three test trajectories, 4,096 actions each. Heads use hidden128/B64, 2,048
updates, seed20261003, Adam1e−3/.9/.999/1e−8, training-only normalization and
weight penalty `1e-4 / train_examples`. Validation-only selection occurs every
128 updates, with no test tuning or refitting. The JSON retains learning
curves, selection steps, all-target metrics and player trajectory slices;
complete per-trajectory/chunk metrics remain in the raw per-model reports.
Intervals use 10,000 crossed paired learner-seed/whole-test-trajectory bootstrap
draws. Only three models and three shared test trajectories limit inference.

The full queue takes **6m39s**, including guards: about 354s collecting states,
36s fitting all 24 readouts, and the remainder in restore/save/guard overhead.
It adds **zero gameplay and zero actor updates**:
98,304 recorded transition replays and 49,152 small diagnostic-head updates.
Restores use fresh belief/RNG, not the previous forecast's exact random samples.
This is diagnostic cost, not a learner-throughput or GPU-utilization result.

All 252 saved parameter/optimizer tensors per actor match the source on restore
and remain bitwise unchanged afterward. All six before/after checkpoints and
captured arrays are finite; categorical states are one-hot. Every h1 prior
deterministic state exactly matches the next posterior's deterministic part
(maximum error 0). The canary also preserves live belief, all five RNG streams,
counters and legacy forecast outputs. The excluded smoke replays 384 recorded
actions and fits eight 32-update, two-coordinate heads; no extra gameplay.

All five native guards/seals pass: canary, smoke and three complete jobs.
No new kernel warning/fault, unfinished native child, NVML polling or host
recovery. Minimum sampled Vulkan estimated headroom is 15.094GiB, not physical
free or peak VRAM; utilization is unmeasured. Native jobs use one CPU quota,
4GiB host memory and zero swap. Meganeura13b19d33/Bladee349cddf remain pinned;
the October4 upstream recheck found main6268ea5 unchanged. No compile overlaps
the full queue.

All 108 Rust CPU tests, 1,051 Python tests, formatting and both strict Clippy
checks pass; [CI272](https://github.com/kvark/kindle/actions/runs/37176447728)
passes implementation `ed61028`. An initial Clippy type-complexity error was
fixed before native qualification. Two CPU report-printing failures occurred
after summary generation; a shadowed variable was fixed and the read-only
audit rerun. Publication also corrects the global regularization metadata to
the sample-count formula above; actual per-head penalties are retained.
No full native job was retried or discarded. Raw evidence:
`runs/rssm-belief-probes-20261004.JKZvbG`.

## Next hypothesis, not a result

The RGB path supplies a reconstruction gradient directly to the full posterior;
our JEPA recipe disables that loss and predicts future features from the
deterministic prior. Replacing RGB targets need not remove the posterior's
observation-learning signal. A bounded current-latent prediction ablation can
test this specific omission, using the saved corpus and existing controls.
Measure held-out posterior readability, causal forecasts versus persistence
and unrelated actions, and added update cost. Capacity, regularization and
probe generalization remain alternative explanations. No new RL matrix,
Phase3 campaign, asynchronous learner or swarms start automatically.
