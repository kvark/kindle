# Optional actor gradients through imagined dynamics

Select native implementation and qualification of the architectural lead in the
[SuperMimic review](../results/2026-11-06-supermimic-architecture-review.md), within
the existing five-game goal. The coordinate comparison is closed and negative;
retain Breakout 143.667 greedy / 70.750 sampled with coordinates off. This is a
different actor estimator. DreamerV2's negative Atari dynamics-gradient evidence
remains relevant; numerical correctness cannot establish improved gameplay.

## Objective and ownership

Add optional positive finite `actor_dynamics_scale`; default absent preserves
legacy configuration, graphs and reports. The behavior optimizer remains the
single owner of actor parameters and moments. Add
`policy_scale * actor_dynamics_scale * dynamics_actor_loss` to its existing
score-function, entropy, critic and imitation losses. No coefficient is selected
for a full learner here; .5 is a qualification value, not a tuned optimum.

The auxiliary loss is the negative mean of continuation-weighted imagined
lambda returns, divided by the existing detached return-normalization scale.
Use the same B*T starts, H-step horizon, pre-update deterministic/categorical
states and action/latent uniform draws as ordinary imagination. Reward and
continuation at arrival t+1 belong to action t. Continuation already contains
the configured time discount, so do not multiply by gamma again. Both reward
and critic distributions decode to the mean of raw exponential support bins;
pair opposite probabilities before reduction to preserve exact uniform zero.

Use hard categorical draws in the forward pass and the derivative of mixed
softmax probabilities in the backward pass for both actions and latents. This
straight-through estimator is biased. Freeze world and critic parameters while
retaining derivatives through their inputs. Actor derivatives traverse future
actions, recurrent transitions, reward, continuation and critic bootstrap.
Initial posterior states, trajectory weights and normalization are detached.

The existing critic includes exploration value. Preserve sampled disagreement
bonuses as detached arrival rewards, with the configured intrinsic scale;
do not claim an extrinsic-only objective or a differentiated ensemble. This
keeps the forward return aligned while deliberately omitting the disagreement
action derivative. Existing score-function exploration gradients remain.

Take a private GPU copy of world dynamics/reward/continuation weights before
posterior sampling and world training. It must survive the subsequent world
update unchanged until behavior training. Share no world buffers into that
snapshot. The current critic remains unchanged until the single behavior
optimizer step. Record snapshot submission time separately. World snapshots are
graph inputs, with no optimizer slots or checkpoint entries; they are repopulated
before every update. Restore retains config and moments, with empty replay as
before. Resident device targets remain unsupported
for this option. Native CPU-target Auto is the selected AMD path.

## Qualification before Atari allocation

Require CPU configuration/serialization, graph gradient ownership, CLI native
capability/restore checks and report/config accounting. Run the appropriate
Rust/Python suites and strict Clippy. Preserve failed attempts and source-specific
limits. Package exact native, runner, test executable and source bytes outside Git.

Under the existing host guard, test an independent F64 recurrent surrogate:
forward return, six actor derivatives through actions and latents, unchanged
world/critic parameters, detached weights/normalization and zero-weight masks.
Finite differences must hold hard samples and detached probabilities at their
unperturbed values; finite differences through hard argmax are not an oracle.
Also test the real tiny learner for two synthetic updates: pre-update private
world snapshots, matching state/draw inputs, optimizer ownership and checkpoint
restore of all parameters/moments. Check the complete auxiliary loss against the
existing rollout/target formula, including disagreement and normalization, with
absolute/relative tolerance .005; separately qualify the scalar decoder and
surrogate derivatives against F64. No Atari actions occur in these fixtures.
Default-off legacy behavior must still pass
a bounded frozen restore before an Atari learner is declared.

Only after qualification, separately declare one bounded 8,192-action native
cost/numerics check with the retained 12M64/Auto/N8/B16/T16/H15/R16 recipe and
one fixed coefficient. Measure wall time and memory before choosing a full
learner. The current decoder and recurrent graph may add substantial backward
cost. Do not infer useful gradients from low prediction error alone; compare
the new objective against observed alternative-action outcomes in a targeted,
separately declared assessment if warranted. No old collector is repeated.

No full learner, sweep, extension or automatic successor is selected here.
Retain the original 864/90%-two-wall and five-game gates. Evidence root:
`/x/Temp/kindle-actor-dynamics-20261106`.
