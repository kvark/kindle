# A fixed timing correction improves current paddle-position estimates

The causal estimate `pooled_center + observed_action_displacement / 8` passes
its fixed prediction screen. On the reused episode2 holdout, MAE falls
**1.435→0.975 pixels** overall and **1.470→0.963** near descending balls, with
lower overall MAE in all eight lanes. RMSE also improves in both subsets.
No coefficient is fitted or searched; the factor follows from half a raw frame
of pooling delay and four raw frames per action under constant velocity.

| Subset | Rows | MAE, pooled → corrected | RMSE, pooled → corrected |
| --- | ---: | ---: | ---: |
| Episode2, all | 12,863 | 1.435127 → .974565 | 1.855747 → 1.357019 |
| Episode2, near descending | 2,606 | 1.469877 → .962826 | 1.880011 → 1.329591 |
| Episode2, contact within one action | 684 | 1.692251 → 1.145194 | 2.017739 → 1.454681 |
| All selected episodes | 35,610 | 1.455695 → .988930 | 1.867918 → 1.365783 |
| All selected, near descending | 7,509 | 1.445932 → .951575 | 1.861221 → 1.320054 |

The overall held-out median absolute error falls2→.875 pixels and the95th
percentile3→2.875. Exact matches fall3,564→2,561: the correction is an aggregate
improvement, not a pointwise improvement. Acceleration, reversals and sticky
actions can violate its constant-velocity approximation. Current command,
future pixels and raw target positions are excluded from the prediction inputs.

This supports separately investigating a response model that uses the corrected
position. It does not select a teacher change or gameplay test: the existing
response model predicts the next pooled position, so simply substituting a
latest-frame estimate would mix input and target conventions. The ball's timing,
collision physics and closed-loop behavior remain untested. No native input,
architecture, loss or learner changes, and no pooling-induced gameplay failure
is established.

The retained teacher trajectory remains225.083 points with0/24 two-wall successes.
All52,072 actions/208,254 frames are aligned by lane,episode and decision index;
42,303 decisions belong to the fixed24 episodes. Preserve the prior exclusions:
24 initial/reset observations,6,545 cropped components,88 detector disagreements
and36 nonstandard widths. All35,610 accepted current observations match their
saved pooled centers and have a four-frame preceding action interval. Targets
are retained raw pixel components, not exact emulator collision state. The
episode0–1 and episode2 results are both retained; no fitting occurs, and the
heavily reused holdout is not an independent cohort.

Eight CPU fixtures qualify alignment/reset handling, duplicate/misaligned-row
refusal, target/future-input isolation, subset boundaries and the fixed screen.
The old pooled metrics reproduce to1e-10. All14 pins pass; the independent
JSON recount verifies35,610 rows and400 aggregate metrics. No old collection,
replay, analysis or qualification repeats, and no emulator or native module is
imported. No GPU work or learner update occurs. There are no failed stages.

Source `1cc31e3a`, [fixed protocol](../experiments/2026-11-03-cdp-breakout-paddle-timing.md).
Preflight takes.194s; analysis takes.785s and completes **08:55:43.996 UTC
November3**. Controller completion is **08:55:44.059**, .888s; independent recount
finishes **08:56:25.832**, .200s. The root
`/x/Temp/kindle-breakout-paddle-timing-20261103` is closed; do not repeat its stages.
No Kindle allocation is active at closure. No response-model fit, gameplay test
or automatic successor is selected. The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `d256a0b5a7ecde0707cc1753efcd11b7a346f7c2a5381414d99c4d79233fe8e4` |
| Review | `7edba75e8167948f072f32f6dbfa765728da46a3bfb3da7851a5861615f8cb10` |
| Completion | `469e8e1c6a4378aa1fddf74c4362557aedf6fd8a8ed3110791d500a4be582d47` |
| Independent recount | `13ab40d7bb5ddd37cb5a0b2b6be8089ade6b1fd46e852d1747058a436df1c9a1` |
