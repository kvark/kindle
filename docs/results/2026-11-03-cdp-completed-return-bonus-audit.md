# Completed-return actor targets remain active against the combined critic

Completed observed returns exceed the retained bonus-inclusive critic in
**390 of480 saved states (81.25%)**, across all eight lanes. Their clipped,
normalized weight sums94.779770, with228.457 effective weighted rows and no
saturated weights. This supports investigating a conservative combination of
the completed-return actor and retained exploration bonus. It does not measure
gameplay improvement or select a new learner.

The [fixed CPU protocol](../experiments/2026-11-03-cdp-completed-return-bonus-audit.md)
uses every state from the two previously captured B16/T16 batches and the
original35,552-action sampled frozen trajectory. No states, coefficients or
new trajectories are selected after inspecting results. All480 states have a
subsequent true terminal in that log. Future lengths range133–1592 actions,
median680. This is retrospective availability, not availability when an online
learner would have sampled the state.

| Statistic | Completed observed return against combined critic |
| --- | ---: |
| Completed rows |480 /480|
| Positive rows |390 /480|
| Batch0 /batch1 positive rows |205 /185|
| Weight mass |94.779770|
| Effective weighted rows |228.457|
| Saturated weights |0|
| Executed-action teacher agreement |53.125%|
| Teacher agreement weighted by actor target |49.597%|

Each lane contributes60 rows; positive counts are60,60,50,49,30,60,36,45.
The fixed ±.001 raw-advantage sensitivity leaves390 positives in both cases,
with weight masses94.768886–94.790654. It is not a proved error bound. Effective
weighted rows describe weight concentration, not statistical independence.
Teacher agreement is descriptive; these returns cannot identify whether an
individual action caused success. In particular, reinforcing completed outcomes
would not simply duplicate the corrective teacher labels.

## Why the combination is coherent as a conservative target

The [completed-return actor pair](2026-11-03-cdp-episodic-actor.md) improved its
bonus-off control from65.208 to95.375 greedy, at nearly unchanged cost, but
trailed the retained bonus-on143.667. That historical comparison also changes
runtime and ensemble parameters, so it does not isolate the bonus's effect.
Current native configuration rejects completed-return learning with the bonus.

Source inspection of `exploration.rs` finds action-effect ensemble disagreement
followed by ReLU and positive averaging. `agent.rs` scales the bonus and checks
it is finite and nonnegative before adding it to imagined and replay rewards.
The ensemble's state/action inputs and prediction targets are detached; its
loss is not direct auxiliary supervision of the CNN. Removing that ensemble
must not be described as removing such a representation gradient.

For fixed current value V, positive normalization S, and any nonnegative finite
completed bonus sum B, monotonic clipping gives:

```
clip((G_observed - V)/S, 0, 1)
    <= clip((G_observed + B - V)/S, 0, 1)
```

Using observed return alone therefore gives a lower bound on the actor weight
from a hypothetical completed bonus-augmented return. This requires no extra
critic or invented historical bonus. It deliberately uses the combined critic
as a conservative threshold. The inequality holds even for an inaccurate
critic; it does not establish a bound on true action value, an unbiased policy
gradient, or improvement over the current policy. Changing the critic or
normalizer between the two sides would invalidate this particular comparison.

The saved values were approximately recovered from rounded native targets in
the [earlier audit](2026-11-03-cdp-breakout-replay-actor-targets.md). They use
fixed-batch posterior states, not original online replay samples. Completed
returns here use stored observed rewards through the terminal, no lambda
mixture or value bootstrap, native gamma.9969969987869263, and F32 return
rounding. The recorded stored intrinsic channel is zero. No model bonus sum is
reconstructed or assumed to be an observed reward.

## Validation and decision

All15 CPU fixtures pass before execution, including episode/stream boundaries,
timeouts, terminal rewards, retained suffixes, outgoing alignment, signed
rewards, discount endpoints, clipping and10,000 fixed monotonicity cases.
The audit verifies512 arrival action/reward pairs and480 outgoing actions and
teacher labels. Backward recurrence and independent forward sums differ by at
most3.553e-14 before F32 rounding. All26 logged episode lengths/returns and all
480 later terminal returns are checked; no excluded row is silently replaced.

The audit completes **17:41:03.986 UTC November3**, source4d2463e0, in**.445s**
CPU. Independent terminal recount completes **17:41:24.245** in**.332s**.
All17 current,20 prior-audit and158 original-capture pins remain unchanged.
There are no failures, model forwards, emulator actions, GPU allocations or
learner updates. No completed prior stage is repeated. Root
`/x/Temp/kindle-completed-return-bonus-audit-20261103` is closed; do not rerun it.

The next design candidate is an explicit optional conservative mode that keeps
the retained bonus and the existing actor target arithmetic. Before a full
learner, it still needs native configuration/CLI validation, default checkpoint
compatibility, gradient qualification and a fresh bounded activity/cost check.
No implementation or GPU job is selected by this CPU audit. Retain the existing
gameplay results; Breakout and the five-game goal remain unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration |`4b67bfce81e4ddb5bfa782041002c5583a679f789f19e66f6e587377c1733b51`|
| Analysis |`360a8e2205c69b2800175d75235b9a5b39a7a3334999443e460be39986c326c7`|
| Row arrays |`8e00f5fb818919452a7d56b124ce9f3e86e077fd7621e9bea93dab741db18997`|
| Completion |`7dd5a205e9079162bf1a9dc47c93f48433fef2d6d6b146063026acde7e337d10`|
