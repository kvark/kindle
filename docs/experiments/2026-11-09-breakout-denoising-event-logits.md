# Frozen reward-logit diagnosis

The [lower-noise audit](../results/2026-11-09-breakout-denoising-low-noise.md)
finds scalar reward RMSE65.604 for the retained sigma_data.05 model on512
zero-reward ordinary predictions, despite small positive-reward probabilities.
Select one frozen head-only diagnostic to distinguish scalar-support tails from
input-distribution errors before changing training. Root
`/mnt/data/kindle-breakout-denoising-event-logits-20261109`.

Add one ignored native test collector; production code, weights, graphs,
inference and checkpoint formats stay unchanged. Restore only the retained
sigma_data.05 checkpoint at age2048. Feed its existing event CNN/MLPs:

- Actual history and actual next image, arrivals4..31: a privileged classifier
  task, including both positive source rewards. This is not a world forecast.
- Actual history and cached predicted next image, arrivals5..31/four draws:
  new causal one-step event evidence, including arrival5/lane1 reward1.
- Cached ordinary generated history/next image, arrivals16..31/four draws:
  require all12 saved reward/positive-probability/continuation arrays bit-exact
  before interpreting extracted logits.

Use all eight lanes and draws83001/83003/83009/83023. No new RGB generation or
old assessment rerun; cached current RGB is decoded solely for the declared new
head inputs. No other model query. Targets/rewards/endings never enter head
inference; only image history, incoming commands and the supplied current image.
Retain all200 batched head calls/1,600 rows,255 raw reward logits per row,
continuation logits, decoded outputs and exact scalar support. Verify135 frozen
tensors andage2048 unchanged.

CPU analysis must reconstruct the native softmax/decoded mean and attribute
probability/mean contributions across scalar magnitudes, with paired endpoints
and positive cases retained. Alternative modal/coordinate readouts are diagnostic
only: do not count them as new training, serve them as a new model or relax any
physical/event gate. Actual-image classification does not establish causal
forecast competence; the two positive source events and zero terminal cases
cannot establish natural-frequency calibration or reliability.

Build/package one release test executable, run22 focused CPU checks and strict
release test-target Clippy; five GPU tests remain ignored until selected.
Production bytes must matchfecc251b. Pin source/executable, data/cached inputs,
checkpoint/config/previous receipts and runner before one300s guarded AMD job.
Use Auto and the existing host guard, no reset/reload/NVML. Require terminal
success andnine sealed guard files; release only this disposable build cache.
No learner, extension, collection or automatic successor. All earlier jobs,
failures and repairs remain closed. Full physics/events/calibration, changed-
brick memory, late/tunnel/second-wall/above-top, untouched families and fresh
confirmation remain required; the full goal is unproven.

Completed: the [result](../results/2026-11-09-breakout-denoising-event-logits.md)
reconstructs all native means/probabilities. Mass outside±32 dominates scalar
error even with real-image inputs. Actual-image recognition still misses one
of two positive examples.12 cached arrays/135 frozen tensors exact;75 closure
pins andnine sealed files verified, service terminal, own cache released. This
is diagnostic attribution, not a learned replacement decoder or full-goal pass.
