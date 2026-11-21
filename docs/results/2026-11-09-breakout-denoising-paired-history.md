# Matched-noise history diagnosis: forecast errors accumulate

The goal **prove the world model training on Breakout** remains unproven.
The [frozen paired-history check](../experiments/2026-11-09-breakout-denoising-paired-history.md)
shows a substantial loss from replacing real past images with the model's own
past predictions, even with identical sampling noise within every comparison.
No training or source/native change occurs.

| Draw | Real history correct /110 | Generated history correct /110 | Both correct | Real only | Generated only | Both wrong |
|---|---|---|---|---|---|---|
|0|77|40|39|38|1|32|
|1|86|41|40|46|1|23|
|2|77|35|33|44|2|31|
|3|77|33|32|45|1|32|

The generated branch loses38–46 real-history successes per draw and gains only
1–2. It also reduces supported detections from77–87 to56–65 of110. By horizon16,
real-history one-step predictions get4–6 of8 positions right; the recursive
branch gets0 in every draw. This supports training for imperfect history rather
than treating the earlier gap as an inference-noise artifact. It does not show
that history is the only remaining bottleneck: real-history one-step predictions
also remain below their original accuracy gate.

The controlled branch resets each draw's seed at every step, matching the cached
real-history one-step calls. Production forecast code starts the same Gaussian
stream for the same seed and8×3×64×64 shape. Only past images change after step1;
actions, weights, sampler, coefficients and noise match. Temporal noise is therefore
correlated across steps. This is a controlled diagnostic, not the ordinary
production rollout distribution. The original ordinary matches35/38/34/33 of110
remain retained; this reset-noise recursive branch scores40/41/35/33. Neither
result replaces a failed fitting gate.

The generator receives only the observed16-image prefix and submitted actions;
future real images, object locations, rewards, endings and emulator state never
enter it. Its own generated RGB is appended recursively. The API's separately
seeded observed-denoising output is never fed back. All20 first-step arrays are
bit-exact to the cached original first forecast, and the four first predictions
also match the cached real-history target16. All135 frozen tensors remain exact.

Reuse source7871f08b/nativeaf13a03e and the sigma_data.05 checkpoint at age2048;
diagnostic source8826416b.64 native one-step queries take4.934s on the shared host;
the CPU analysis takes1.135s. No build, learning, new emulator actions or frames.
All22 declared/22 CPU pins pass. One terminal successful service and a sealed
nine-file host guard pass; invocation014c4ff5f9ee49b6b9659be36b0ddbd8. Missing balls
count as failures;18 unsupported endpoints per draw remain unscored.512 paired
endpoint rows and all horizon ball/paddle/brick metrics are retained. Event
outputs from the generated branch are retained, but no paired event comparison
is claimed because cached real-history event outputs are absent.

Four draws share one reused development learner and eight related clips; they
are not independent learner seeds or held-out game evidence. This diagnosis
does not prove natural event calibration, changed-brick memory, late/tunnel/
second-wall/above-top coverage, untouched-family generalization or fresh-seed
confirmation. Actor/critic, CDP comparison and five-game work stay parked.
No learner, extension, collection or automatic successor is selected here.

Root `/mnt/data/kindle-breakout-denoising-paired-history-20261109` retains the
declaration/source archive, scripts, raw prediction/event arrays, frozen weights,
paired rows, metrics, host guard and terminal service receipt. All earlier
experiments, failures and repairs remain closed and retained.
