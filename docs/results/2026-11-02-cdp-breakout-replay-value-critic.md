# The replay-value gameplay gain leaves weak Breakout critic calibration

Increasing replay-value weight0.3→1 improved retained frozen gameplay, but
the stronger model still predicts complete sampled-policy returns poorly:
correlation **.231**, R² **−.439**, and critic RMSE **9.231**, versus **7.855**
for a constant fitted on earlier episodes. The0.3 model also trails its constant
baseline. This identifies an unresolved value-prediction weakness; it does not
establish that critic error causes the remaining gameplay gap or that another
loss-weight increase would help.

The [fixed diagnostic](../experiments/2026-11-02-cdp-breakout-replay-value-critic.md)
captures **151,384 actions / 605,381 frames in 171.911s**, with zero updates.
All four original trajectories reproduce exactly and all357 tensors per arm
remain unchanged. Original scores, training times and zero-of24 two-wall
outcomes stay fixed. The preceding1,024-action check adds1.025s, also with zero
updates. No new learner, loss change or automatic successor is selected.

## Complete-return calibration

Targets use actual arrival rewards plus the frozen native intrinsic bonus,
the same F32 discount1−1/333, complete natural episodes and zero terminal
bootstrap. No critic prediction enters the target. The first episode in each
lane fits a constant; the next two per lane give16 test episodes. Rows within
an episode are correlated, so row counts are not independent sample counts.

| Replay-value / mode | Test rows | Target mean | Predicted mean | Bias | Correlation | R² | Critic RMSE | Earlier-episode constant RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 sampled |18,726|15.565|17.251|+1.686|.218|−.421|9.891|8.384|
| 1.0 sampled |20,765|16.527|12.282|−4.245|.231|−.439|9.231|7.855|
| 0.3 greedy |22,779|27.417|23.831|−3.586|.413|.154|26.611|29.067|
| 1.0 greedy |23,563|29.685|16.420|−13.265|.371|−.268|23.708|21.152|

Sampled arms are primary because the imagined actor samples actions. Both
sampled critics lose to their earlier-episode constant in squared error and
have negative R² against the hindsight test mean. Weight1 has lower absolute
RMSE but larger mean underprediction, and it still trails its own constant.
The target distributions differ; these numbers do not establish a causal
calibration improvement or regression from changing the weight.

Greedy changes the policy and visited states. Its larger value underprediction
does not by itself show that a critic trained for the sampled policy is defective.
The0.3 greedy critic beats its constant; the1.0 greedy critic does not. Realized
returns are noisy samples of expected values. The two checkpoints share seeds
but not state distributions, so this is not a matched-state comparison.

The sixteen held-out episode starts have predicted/realized mean returns
10.957/12.679 and10.502/12.516 for0.3/1.0 sampled, and10.958/13.316 and
10.512/13.973 for greedy. Their MAEs are1.920/2.142 sampled and2.639/3.597
greedy. Individual start predictions and decomposed returns remain in each
analysis artifact; these are descriptive aggregates, not independent seed tests.

| Replay-value / mode | Mean external return | Mean intrinsic return | Intrinsic fraction of total | Per-action bonus mean |
| --- | ---: | ---: | ---: | ---: |
| 0.3 sampled |14.059|1.506|9.676%|.006171|
| 1.0 sampled |15.171|1.356|8.204%|.005405|
| 0.3 greedy |25.805|1.612|5.881%|.006185|
| 1.0 greedy |28.307|1.378|4.642%|.005322|

These bonus fractions do not establish gradient dominance or harmful exploration.
The final50k training means remain in the preparation receipt: imagined reward
.066328/.057007, scaled bonus.006033/.005788, entropy.683574/.696046 and
normalized absolute advantage.015926/.019345 for0.3/1.0, each3,125 reports.
They describe training outputs rather than complete realized returns or gradients.

## Execution and retained evidence

Both models retain12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/
dynamics4e-4/expert100 and sampled training collection. Native447fe50/SHA35ad0db
is unchanged, as are their exact runners6c1b69fb/cf4ea238. Runtime is
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. The live posterior probe
reads after act and before the environment transition, without another posterior
sample. Guidance and labels are off. No native build occurs.

| Capture | Actions | Frames | Seconds | Matching semantic events | Retained score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.3 sampled |31,848|127,346|33.266|4,030|59.000|
| 0.3 greedy |40,560|162,206|49.065|5,122|117.292|
| 1.0 sampled |35,552|142,169|37.560|4,496|70.750|
| 1.0 greedy |43,424|173,660|52.020|5,478|143.667|

All74 pins, checkpoint counters/config/native/own-runner identities,357 invariant
tensors per arm, finite values and nonnegative bonuses,24-episode cohorts,zero
truncations,host guards and source-log hashes pass. Existing source replays are
reused after full semantic equality and hash checks, with no emulator replay
repeat. Source logs lack live pixel hashes; unavailable direct image equality
is not claimed. Qbert phase labels are excluded from the unchanged return helper.

CPU preparation passes08:18:24.992 UTC November2/sourcefe139dd in5.834s:
five commands,both schemas,282/338 source and422 preceding H30/R12 inputs.
Nine passing arithmetic tests are retained after verifying unchanged test/helper
sources, without repetition. No native or emulator is constructed in preparation.
The1,024-action check matches128 source events and preserves357 tensors.

Original main916432/invocation628c655e44284a00ab7ecd46b019125d launches
08:20:22.182 UTC November2/sourced982167 and succeeds **08:23:41.562**.
Original collector916435/4fe85dbc943542098062105aa841e1d4 succeeds
**08:23:43.655**. The read-only terminal reader passes once **08:24:19.771**,
in1.074s (1.092s wrapper), verifying both original identities, all pins and files,
four analyses, retained scores and return decomposition. No preparation,
capture or collector fails; no original stage repeats.

Root `/x/Temp/kindle-breakout-replay-value-critic-20261102` is closed. Do not
repeat captures, qualification, collection or terminal reading. No learner or
Kindle GPU allocation is active at closure; check live state before new work.
The five-game goal remains unfinished. Keep the gameplay gain while investigating
the critic weakness; no new coefficient, target, architecture or learner is selected.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `a533cbd83dfc7a23eefca0b483a06287915278ccabea7987bdb06875708f9a58` |
| Comparison | `d7b901a6fc686a58dfe5bcf38130ac31cf57c4ea69500664c264c0a37614a4d8` |
| Original completion | `7d26970f938432d0c8282fe9c6853c379e4a0f9e688543301549b520fd039b15` |
| Terminal summary | `4f127fcdf87c2755bd8f3ead17aa804fa75c53a405b7571758ccbe3c04364b94` |
