# SuperMimic identifies a distinct policy-gradient experiment for Kindle

Source review at2d651cb0 confirms that Kindle learns a world model and trains
its actor on imagined trajectories, but does not differentiate the imagined
return through the dynamics into the actor. SuperMimic's corresponding path is
a meaningful architectural proposal within the five-game goal. The [optional implementation](2026-11-06-cdp-actor-dynamics.md) has passed
native qualification and the separately declared8,192-action cost check. The
new path adds22.39% bounded wall time. A separately declared
[fresh200k comparison](../experiments/2026-11-06-cdp-breakout-actor-dynamics-200k.md)
is now complete and negative:85.708 greedy/51.625 sampled in26m38.517s,
below retained143.667/70.750 in21m27.910s. Keep the option off; this coefficient
and seed do not establish useful extra learning or rule out all such recipes.
The [completed frozen action-path probe](2026-11-06-cdp-breakout-action-path-derivatives.md)
finds recorded-plan return gradients favor81/92 known short rescues for control
and80/92 for candidate. Survival directions favor12/18 and8/18 differing game-over
rescues, while actual-terminal plans still receive about.97 discounted survival.
This does not show generally reversed return gradients or isolate the full learner
regression; actor feedback, other losses and optimizer interactions are omitted.
The [completed actor-update diagnostic](2026-11-06-cdp-dynamics-actor-update.md)
adds a limited late-checkpoint result: the full estimated actor update changes
only.01136%/.01702% on two archived batches, with exact critic invariance.
Imitation and saved momentum still dominate. This neither explains early
training nor justifies increasing the coefficient. No successor allocation is
selected. The independent
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
remain unchanged. No speedup, completed five-game goal or successor allocation follows from the
paper alone; the implemented comparison and diagnostic results are linked above.

## Architectural implications after the measured comparison

Keep learned imagination and the optional connected actor path. The failed
coefficient.5 recipe is evidence against adopting that recipe, not a reproduction
of SuperMimic or a rejection of every differentiable-control method. A derivative
can be computed correctly for the declared surrogate while remaining poor
guidance for real action choices. The completed outcome and optimizer probes
provide narrower evidence than a full explanation of the regression.

For future Atari development, prioritize the model's usefulness for choosing
actions: accurate alternative-action rankings, survival and reward predictions
over the control horizon, followed by a measured effect on policy updates and
frozen gameplay per minute. Reuse the completed probes when defining a new
hypothesis. More multi-step supervision alone is not a fresh proposal: the
[three-step comparison](2026-11-03-cdp-breakout-open-loop.md) already regressed.

One distinct future estimator would score each discrete action with a fixed
world model and critic, then train the actor against the probability-weighted
action values. This avoids differentiating through a relaxed button choice,
but retains model/critic bias and adds branch computation. It is not equivalent
to multi-step dynamics backpropagation, and is now selected as the distinct [enumerated one-step actor experiment](2026-11-06-cdp-actor-expected-value.md), with CPU/native qualification and bounded cost gates passed. It adds31.35% bounded wall time; gameplay benefit remains unmeasured. A separately declared fresh200k comparison is now selected to test gain per training minute.
Continuous latent states alone would not remove the discrete-action issue.

For a future continuous-control task, direct actor gradients would be a more
natural candidate. The native capability now exists; its value would still need
measurement in that task. Sharing one robot's dynamics across goals also does
not establish that unrelated Atari games should share a single dynamics model.
These are architectural directions within the existing goal, not permission
to repeat closed experiments or a claim of a SuperMimic-sized efficiency gain.

The distinct [residual actor experiment](2026-11-06-cdp-residual-actor.md#full-200k-comparison)
also fails retention:117.625 greedy/49.333 sampled versus143.667/70.750, with
17.792% lower primary gain/min despite.751% less training time. The zero-initialized
46,098-parameter correction separates imagined policy/entropy updates from
imitation, preserving actual-initial scores and traces. Dynamics gradients stay
off. Its primary paired interval spans zero; this single recipe neither isolates
a gradient-conflict cause nor rejects every independent reward head. Keep it off.
This is not a SuperMimic reproduction or evidence that its ratios transfer to
Atari. No successor allocation follows automatically.

A subsequent [serving ablation](2026-11-06-cdp-residual-actor-serving-ablation.md)
raises both point scores when the correction is zeroed, but both paired intervals
span zero and neither mean reaches retained. Holding the learned world/base and
training history fixed makes this a narrower serving test; it does not establish
that a pure-imitation learner would be better. Useful action values and their
measured policy effect remain unresolved. The later enumerated one-step actor
experiment tests a different estimator; it does not repeat these closed learners.
