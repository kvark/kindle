# Diagnose the failed Freeway disagreement screen

No new policy training or budget extension. Replay all six completed32,768-action
logs on CPU to measure player-height coverage and sustained requested movement,
checking every reward, terminal/truncation, reset and executed-frame count.
Original pixel hashes were not logged; do not claim original-image equality.
RAM14 supplies player height, independently checked against the yellow sprite
at x44 in raw frames. Road markings share its color and must be excluded by
connected row extent. RAM labels never enter the policy, replay or bonus.
Source: [OCAtari](https://github.com/k4ntz/OC_Atari/blob/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram/freeway.py).

For candidate seeds1009/2017/3019, restore the completed checkpoint and re-encode
its saved stream-zero trajectory with the existing GPU RGB actor. Force only
the already recorded actions for replay; this is a diagnostic, not a benchmark
action aid or new training experience. Sample current/next latent state, CNN
tokens and policy/value diagnostics every16 arrivals:256 states per seed.
All4,096 recorded actions per seed are retained; no learner update. A fresh
restored belief is not the original live online belief. Save after collection
and compare every model/optimizer tensor to the source checkpoint.

Evaluate the existing GPU disagreement graph for all18 actions at each sampled
state. Check finite nonnegative bonuses and unchanged saved tensors. Compare
within-state action variation with between-state variation, actual direction
and height, and the measured policy. Check an independent F64 MLP calculation
against GPU outputs before trusting derived ensemble statistics. Interpret
these as current-model diagnostics, not causal forecasts of actual game return.

Three300-second guarded collection jobs, then three120-second guarded bonus
readouts; serialize native processes, require RTX5080 and >=2GiB sampled Vulkan
headroom, retain warnings and stop/review failures. CPU replay/analysis/builds
use one CPU/2GiB/zero swap. Keep all failures and source artifacts.

Upstream recheck: Meganeura main6288f88 changes only paper/docs/artifact scripts
over the qualified592a2f5; Blade main remains e349cddf. Keep the numerically
unchanged qualified runtime for this diagnosis. No automatic learning launch:
use the evidence to declare one focused mechanism change and finite comparison.

The first CPU reference comparison fails its unchanged3e-6 absolute bound:
seed1009 differs by1.4632e-5. Retain that failure. A diagnostic-only precision
comparison evaluates the same saved states and weights with the default GPU
matrix policy and native-F32 GPU policy, recording selected pipeline keys.
Three additional120-second readouts; no actor/learner arithmetic is changed.
Require the F32 path to meet the original independent-reference bound before
interpreting contrast; report the default-path error rather than hiding it.

Retained setup failures: the first systemd build lacked Cargo on PATH (no native
job), and the first precision readout used SessionConfig's training default
instead of explicit inference. It stopped at graph differentiation with a shape
assertion before evaluating the bonus. Correct the diagnostic to inference;
this is not a driver fault or a changed learner. Review before the new queue.
