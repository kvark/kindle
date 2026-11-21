# CPU audit of a possible reward-weighted replay actor target

The current actor learns from imagined returns and corrective teacher labels.
Real replay returns train the critic and, when enabled, the world representation,
but have no direct actor term. The completed400k run improves absolute scores
while lowering learning per minute. Before selecting a new loss, inspect the
already-saved two B16/T16 batches from the current200k actor-gradient diagnostic.
Do not repeat its capture, backward pass or readers.

This is a target audit, not a learner or an implementation selection. Positive
advantage imitation is motivated by [Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf)
and [Advantage-Weighted Regression](https://xbpeng.github.io/projects/AWR/AWR_2019.pdf).
Our existing bootstrapped lambda target is neither the episodic-return SIL
target nor AWR's exponential weighting. No claim of equivalent guarantees is made.

Use both saved native input archives, batch metadata, original frozen action
log, source checkpoint metadata, before/after normalizer state and the original
158 dependency pins under `/x/Temp/kindle-breakout-actor-gradients-20261103`.
Resolve the exact447fe50 distribution and target-alignment source. Hash all
new inputs and protocol before interpreting the target statistics. No native
module, emulator, model forward pass, new environment action or GPU work occurs.

Decode the saved two-hot targets with the native exponentially spaced F32
support. Reconstruct the two sequential percentile-normalizer updates from
saved imagined returns; require the final low/high state within1e-4 of the
saved native state. Recover each posterior value from the first imagination
step's identity `V = G_imagined - action_target_sum * return_scale / weight`.
Require finite positive weights, one selected action per imagined row, valid
two-hot distributions and no clipped support endpoints. This is approximate
recovery from rounded saved targets, not a new native value measurement.

At each of480 replay rows, pair feature at time t with the executed action and
teacher label stored at arrival t+1. Require all480 teacher targets to equal
the saved native targets and all512 incoming actions/rewards to match the
original frozen log. Require no terminal/reset crossing in any fixed window.

Report `A = G_replay - V` and the hypothetical normalized positive weight
`w = min(1, max(0, A / return_scale))`, with no coefficient selected. Report
counts, weight mass, effective sample size, saturation, teacher agreement and
all eight lane summaries. Preserve every row and both batches; no row selection
based on reward, advantage or label agreement. Treat abs(A)<=1e-3 as sign-ambiguous
and retain that count separately. Recompute summaries at A±1e-3 to expose sensitivity;
this fixed sensitivity range is not a proved native reconstruction error bound.

For a fixed no-external-reward counterfactual, subtract the linear contribution
of the observed external rewards inside each replay window:
`E_t = r[t+1] + gamma*lambda*E[t+1]`, with E=0 at the window endpoint.
Gamma and lambda use the original F32 configuration. Keep all imagined
bootstraps and intrinsic bonuses fixed. Report positive-row retention, weight
mass retention, sign changes and teacher agreement under `A-E`. This isolates
the arithmetic contribution of those short-window external rewards, not all
information learned from rewards or the value of complete successful episodes.
Model bootstraps and intrinsic bonuses remain mixed in the residual.

Qualification fixtures cover mirrored support, two-hot decoding, percentile
updates, inversion of weighted normalized advantages, outgoing action/label
alignment, reward recursion, empty-positive summaries, sensitivity and rejection
of malformed/nonfinite inputs. Run one CPU analysis capped at60s on one BLAS
thread. Keep original failures if any and never overwrite a completed result.
Retain source snapshots, tests, declarations, row arrays, analysis and execution
receipts under `/x/Temp/kindle-breakout-replay-actor-targets-20261103`.

The two correlated batches use reused sampled histories and fresh scalar
posterior/imagination draws, not original training replay or independent seeds.
Teacher agreement does not identify correct actions, and a positive replay
advantage does not establish that an action caused a good outcome. No native
gradient, optimizer movement, runtime cost or gameplay benefit is measured.
Select any implementation and bounded gradient/cost check separately. Keep
the200k70.750/143.667 control and the400k92.792/173.708 budget point unchanged;
all still have zero of24 two-wall successes. The five-game goal is unfinished.
