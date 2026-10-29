# Selected next development step: short open-loop world supervision

The three completed Qbert diagnostics show useful forecasts and values, with
weaker later-game action-block ranking. They do not establish that world error
causes the remaining score gap. Test a concrete learning change against that
hypothesis rather than treating an extra loss as a proven cure.

## Scope selected now

Implement optional three-step world supervision against recorded future targets
in native Rust/Meganeura. Keep the option disabled by default. Starting from an
observation-corrected replay state, use recorded actions and prior latent states
for successive predictions without reading intervening observations. Supervise
future detached CNN encodings, rewards and continuation where the recorded
sequence stays in the same episode. Do not use diagnostic oracle branches or
their outcomes as training experience.

Inspect existing graph/configuration/RNG conventions before choosing the exact
API and reductions. Prefer a small number of starts per replay sequence to bound
cost. Avoid simply multiplying the existing one-step CDP term; distinguish
existing supervision from new free-running future targets. Record the final
loss weights, sampled starts, horizons, masks and gradient boundaries before
GPU validation. Preserve the original teacher-forced objective and expert loss.

The selected work is implementation, meaningful CPU/gradient validation and a
bounded real execution/cost check. It is not a full training allocation or a
commitment to a particular loss weight without inspecting the implementation.
The greedy-posterior draft is parked separately and unused. Its exact seven-file
patch, source copies and hash receipt are retained under
`/x/Temp/kindle-open-loop-20261029/parked-posterior`; the reverse patch was checked
before applying and the saved forward patch applied cleanly to the baseline.
Its protocol remains an untracked draft. Build
from explicitly pinned sources; do not silently include its unqualified native
changes in the new experimental runtime.

## Selected implementation and numerical checks

`open_loop_supervision: {horizon: 3, scale: 1.0}` is optional and absent from
default serialization. The fresh CLI exposes `--open-loop-length` and
`--open-loop-scale`; restore rejects overrides. One posterior start at local
time0 of each replay chunk/row uses recorded incoming actions at times1..3.
Subsequent states use sampled prior categoricals with the existing straight-
through estimator, from a separate ChaCha12 stream; future observations never
enter these states. No new trainable parameters are added.

Detached centered CNN targets at steps2..3 receive cosine loss. Step1 has the
same deterministic prediction inputs as existing CDP and is not duplicated.
Prior-state reward two-hot cross-entropy and continuation logits BCE receive
targets at steps1..3. Each component averages all rows, including masked zeros,
then its supervised horizons (two CDP, three reward/continuation). Apply existing
component weights500/1/1 and global scale1. The detached centering mean remains
over the existing full real-observation batch. Gradients reach the initial
posterior/CNN and rollout dynamics, prior and prediction heads; targets and
centering mean remain detached. No auxiliary actor/critic loss is added.

Masks accumulate from the start: exclude any arrival with `is_first` or with
an `is_last` predecessor, retain a valid terminal arrival, and never re-enable
a row after a boundary. Horizon must be at least2 and shorter than world BPTT.
The original teacher-forced and expert losses remain intact. Metrics declare
the weighted total, three component means and valid fraction.

CPU checks cover configuration compatibility, RNG isolation, boundary masks,
action ancestry, detached-target gradients, unchanged parameters/inference and
ordinary/staged graphs. Separately guarded tiny GPU checks compare masked
continuation values/derivatives with F64, check a smooth reward-head derivative,
world gradient routing, future-target/state invariance, action sensitivity and
zero masks. These checks precede packaging and any bounded Atari allocation.

## Numerical checks complete; selected bounded runtime assessment

Source `ab80486` passes135 Rust CPU tests,611 Python tests and strict native/
binding Clippy. Both guarded numerical tests pass on the RX7900XT; all189 pins
remain unchanged. Service succeeds07:30:58 UTC October29, whole3.222s. Evidence
and isolated wheel/runtime are under `/x/Temp/kindle-open-loop-20261029`.
The original CPU-only missing-method compile error and wrong interpreter path
are retained in their logs; no GPU check failed or repeated.

Next restore the retained12M Qbert checkpoint with the new runtime, with this
option absent, for1024 greedy actions. Require zero updates,357 unchanged
tensors and an exact recorded-action/outcome prefix before new learning.

Then select one fresh8192-action Qbert execution/cost check:12M48/Auto/N8/B16/
T16/H15/R16,coefficient1,actor-unimix0,full18,original rates,seed4013,
20%episode continuation/cap1920/expert100/all18weights1/world gradients/all-state
corrective labels, plus open-loop H3/scale1. Compare all357 actual-initial tensors
with the retained12M4013 initial. Expected489 updates,first at376 actions,
residual scheduler credit.5. Restore its final model for1024 greedy actions at
evaluation seed4000004013,zero updates; require357 invariant tensors.

Require finite/nonzero auxiliary reports,declared weighted reductions and masks,
changed CNN/reward/actor/value parameters,complete guidance/pixel/action replay,
matching configs/source/counters and host guards. Fix budgets before launch:
180s training process,60s frozen process,180s CPU audits. Record aggregate
actions,emulator frames and measured training time. Project200k cost both
linearly and with10% allowance; a projection above35minutes calls for a cheaper
recipe rather than silently extending the budget. The short check cannot
establish task competence or select a full learner automatically. Reused seeds
and ordinary frozen logs without live pixel hashes retain their limitations.

## Required evidence before a full learner

- Disabled configuration preserves checkpoint compatibility, default RNG draws,
  model parameters and established inference trajectories.
- Future predictions depend on the proposed recorded actions and prior rollout,
  without receiving future observations as inputs. Targets remain detached.
- Loss masks exclude reset crossings and invalid tails, retain valid terminal
  arrivals and use declared normalization. Verify horizon/action/reward alignment.
- Small native derivative checks establish the intended world gradients and
  stop-gradient boundaries. The auxiliary loss reaches its intended dynamics,
  prior and heads; behavior parameters change only through existing losses.
- An isolated runtime passes relevant CPU tests/Clippy and bounded GPU checks,
  with exact source/native/config pins, finite reports, parameter-change checks,
  actual-initial controls, invariant frozen tensors and host guards.
- Measure early wall time and project the same-budget comparison. Choose a
  full-run recipe and wall cap only after that evidence, favoring useful learning
  per GPU-hour. Do not infer benefit from finite losses or an execution check.

Keep eight environments, Auto precision, native frames/one GPU resize, T16 and
the retained compact recipes as the starting point. Use new output directories
and fresh learners for later comparisons; do not extend completed lives. No
automatic full learner, seed campaign or backend optimization project follows.

Evidence motivating this test:
[forecast, critic and action-ranking results](../results/2026-10-29-cdp-learning-diagnosis.md).
