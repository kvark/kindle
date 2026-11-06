# SuperMimic identifies a distinct policy-gradient experiment for Kindle

Source review at2d651cb0 confirms that Kindle learns a world model and trains
its actor on imagined trajectories, but does not differentiate the imagined
return through the dynamics into the actor. SuperMimic's corresponding path is
a meaningful architectural proposal within the five-game goal. The [optional implementation](2026-11-06-cdp-actor-dynamics.md) has passed
native qualification and the separately declared8,192-action cost check. The
new path adds22.39% bounded wall time; no full Atari learner or gameplay benefit
is established. The independent
[coordinate-channel comparison](2026-11-06-cdp-rgb-coordinates.md) is now closed
and regresses both frozen modes; it does not test this actor-gradient proposal.

[SuperMimic §3](https://arxiv.org/pdf/2610.28480) fits short autoregressive world
predictions and improves a policy by backpropagating critic-bootstrapped return
through a fixed model. Model parameters stay fixed during the actor update;
gradients through model inputs remain active. The authors report about64× fewer
imagined steps than PPO using their learned model on a locomotion subset.
The larger GPU-hour comparison is against a model-free robotics pipeline,
not Kindle or an Atari Dreamer control. Physical-state supervision,continuous
commands,dense tracking rewards and shared robot dynamics limit transfer.

[DreamerV2 §2.2 and Appendix C](https://arxiv.org/pdf/2010.02193) supply direct
counterevidence to assuming an Atari win: score-function gradients substantially
outperform dynamics backpropagation there; continuous control favors the latter.
An experiment must establish improved real gameplay per GPU-hour here.

## Verified current implementation

- `world::build_imagination_graph` unrolls prior dynamics and actor, followed by
  reward/continuation/value heads. `DreamerCore::with_gpu` builds it as an
  inference session. Its outputs become inputs to a separate behavior graph.
- `behavior::policy_loss` uses discounted normalized advantages multiplied by
  sampled-action log probabilities, plus entropy. No differentiable return
  connects this loss back through the imagined state transitions.
- `gumbel_sample` returns hard `argmax_one_hot`. A proposed backward path needs
  explicit estimators for both categorical actions and categorical latent draws.
  The existing world-training straight-through helper is available, but its
  biased derivative must not be described as the true discrete-action gradient.
- `actor_critic_gradient` routes initial posterior representation losses through
  frozen behavior heads. Later imagined states remain detached. Its negative
  experiment does not test return gradients through imagined dynamics.
- Learner targets use pre-update parameters, followed by world training,world
  synchronization,behavior training andbehavior synchronization. Reusing a newly
  synchronized model with stale rollout states would silently change semantics.
- Reward/value decoding is the probability-weighted mean of exponentially spaced
  raw-value bins. A connected objective needs this operation inside the graph;
  the external target/mean buffers alone do not preserve derivatives.
- Disagreement explicitly stops state/action gradients. Keeping its scalar
  forward bonus does not automatically provide its action derivative. An
  extrinsic-only auxiliary objective or a differentiated frozen ensemble must
  be declared explicitly; they are different interventions.

Relevant code: [imagination](../../kindle/src/dreamer/world.rs),
[actor loss](../../kindle/src/dreamer/behavior.rs),
[update ordering](../../kindle/src/dreamer/agent.rs),
[sampling](../../kindle/src/dreamer/networks.rs),
[scalar support](../../kindle/src/dreamer/distributions.rs),
[disagreement](../../kindle/src/dreamer/exploration.rs).

## Proposed development sequence

Keep CDP,representation,environment protocol andteacher fixed when testing the
new actor objective. Use one optimizer owner for actor parameters and moments;
model/critic copies used to propagate input gradients must not receive actor
updates. A new connected graph must preserve a coherent parameter snapshot and
declare how its dynamics-gradient term combines with existing score-function,
entropy andimitation losses. Do not infer the required coefficient from norms
alone: earlier actor-weight andimitation-decay experiments regressed.

First test independent tiny-model values,surrogate derivatives,gradient routing
andunchanged model/critic weights andmoments. Straight-through derivatives require
an independent reference for the declared surrogate; finite differences through
hard argmax are not a valid derivative oracle. Then measure a bounded native
update's cost andmemory before selecting a fresh fixed-budget learner.

Numerical correctness does not establish useful model derivatives. Reuse the
existing action-ranking evidence to define a targeted comparison against actual
alternative-action outcomes,including critic bootstrap andsparse-reward limits.
Do not repeat closed diagnostics unchanged. Enumerating discrete actions is an
alternative estimator,not equivalent to multi-step differentiable imagination;
it has separate compute andmodel-bias tradeoffs.

The retained Breakout70.750sampled/143.667greedy control andoriginal864/90% gate
remain unchanged. No claimed speedup,no completed experiment andno successor
allocation follow from this source review.
