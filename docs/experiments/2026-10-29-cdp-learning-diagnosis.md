# Diagnose the remaining learning gap

The user requested an investigation of the architectural bottlenecks within the
existing five-game goal. Start with the retained Qbert12M48/B16/R16/expert100
learner, whose greedy frozen score11984.375 remains below15000. Its initial
checkpoint and recorded greedy evaluation already exist. Do not extend training
or treat the older unaided4M forecast results as a diagnosis of this learner.

## What the current loss actually supervises

- The CDP predictor uses the recurrent state before the current observation to
  predict its detached CNN encoding. Training then consumes the real observation
  at each successive step. T16 is not15 steps of free-running world supervision.
- Reward and continuation heads train on observation-corrected posterior states.
  Imagination evaluates those heads on prior states without new observations.
- The actor learns from imagined lambda returns, plus the optional expert loss.
  The critic also has a real-replay loss with scale.3. Its bootstrap is the
  imagined return beginning at each replay state, so this is not an independent
  Monte Carlo check of imagination. This wiring is intentional in the code;
  its contribution to the present gap is unmeasured.
- Intrinsic disagreement rewards also enter behavior targets. Comparing values
  directly with external game score would omit part of their training target.

Source: `world.rs::build_training_graph`, `world.rs::build_imagination_graph`,
`agent.rs::imagine_and_target`, `behavior.rs::build_training_graph`,
`distributions.rs::lambda_returns` and the active GPU path in
`targets_returns.wgsl`. No native training change is selected here.

## Selected frozen forecast measurement

Use the existing `probe_policy_forecasts.py` with native447fe50 and the matching
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Remove its historical
346-tensor count assumption; retain exact key, shape, dtype and byte comparison
of every tensor in all three checkpoint files. The selected models have357.
The separate optional greedy-posterior draft is not part of this measurement.

Source root:
`/x/Temp/kindle-qbert-expert100-capacity-20261028/full/qbert-12m-48-b16-r16-expert100-200k-seed4013`.
Replay `final-greedy.jsonl` and its existing full replay/manifest, using both
`final` and its own `initial` weights. Fixed48256 recorded actions per model;
the original first3 natural episodes per lane determine the analysis cohort.
First selected episode per lane supplies simple calibration baselines; the next
two supply16 evaluation episodes. These are reused development trajectories.

First run one128-action final-model execution check, capped at120s. It must
preserve all357 tensors and both learning counters, apart from the declared
diagnostic observation count; validate finite predictions, recorded transitions,
prefix consistency and host guard. After that check passes, the selected paired
captures each have a600s cap, zero updates and no guidance. Pin source, runtime,
checkpoint, replay and analysis inputs before GPU work. Stop on numerical or
host faults; no retry or driver recovery.

Measure one-step and15-step reward predictions,15-step reward sums, continuation
calibration and errors before/after the first pyramid. Also summarize prefixes
of length1/3/5/10/15 on common valid15-step origins, so different horizons share
the same starting states. Compare with zero, calibration mean, persistence and
unrelated-action forecasts. Report class support and sparse terminal limitations.
Require matching input trajectories and pixel hashes between models, exact
counters, all357 invariant tensors and successful guards.

These scalar re-observations use new posterior draws, not the original vector
beliefs. Unrelated-action predictions are evaluated against the recorded actual
outcomes as a sensitivity control; they are not counterfactual ground truth.
This measurement can locate forecast weaknesses but cannot establish their
causal contribution to policy performance. No independent learner, changed loss,
paired behavior-update intervention or automatic training successor is selected.

## Follow-up decisions

If corrected posterior predictions are good but free-running forecasts lose
useful information, consider short multi-step world supervision against recorded
actions/targets. If forecasts remain useful, investigate critic calibration and
action ranking using actual branched outcomes. A real-versus-imagined update
comparison must control policy distribution, intrinsic targets and update budget;
naive off-policy actor updates would not isolate model error. These are decision
branches, not claims that either mechanism is already established.
