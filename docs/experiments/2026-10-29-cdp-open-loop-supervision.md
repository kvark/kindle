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
The existing uncommitted greedy-posterior draft is separate and unused. Build
from explicitly pinned sources; do not silently include its unqualified native
changes in the new experimental runtime.

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
