# Enumerated one-step actor objective

The direct dynamics-gradient learner regressed, and the completed late-checkpoint
probe found that its auxiliary term barely changed the full actor update. Frozen
posterior averaging also failed retention. Test a distinct policy estimator:
replace the sampled-action score-function term with an expectation over every
discrete action at each replay posterior start. Keep the existing entropy,
imitation and critic objectives, model, encoder, replay and collection recipe.
This changes the objective horizon and start-state distribution as well as the
gradient estimator; it cannot isolate categorical gradient bias alone.

For each start s, evaluate every full-space action with the pre-update world:
Q(s,a)=predicted arrival reward + current action-effects bonus + predicted
discounted continuation × current critic value at the arrival. Continuation
already contains the configured time discount; do not multiply gamma again.
Use one categorical prior draw per action, reusing the first ordinary imagination
step's latent uniforms across actions. This is an exact action expectation for
those sampled values, not an exact expectation over latent futures or a known
real-world action value. No emulator state or object labels enter the model.

The actor loss is minus the mean of the starting continuation weight times
sum_a policy(a|s)×stop_gradient(Q(s,a)), divided by the existing detached return
scale. Use the mixed actor probabilities, including unimix once. Center Q by
its per-state unweighted action mean to reduce numeric cancellation; the exact
actor gradient is unchanged. Keep original trajectory-weighted entropy over
imagined states. The ordinary imagination remains for critic targets and return
normalization. There is no new actor-loss coefficient or new optimizer owner.

Add `actor_expected_value`, default false and omitted from legacy serialization.
World parameters are private GPU snapshot inputs copied before world learning;
critic inputs/weights and all Q targets are detached from this actor objective.
Only actor parameters receive its derivative. Reject combinations with residual,
dynamics-gradient, completed-return actor or actor-to-world gradient experiments.
Use CPU target construction on AMD; reject the unavailable device-target path.
Preserve the option on checkpoint restore and reject fresh-training overrides.
Log the unscaled expected-value loss and snapshot time separately. No new learned
parameters, optimizer slots, replay fields or RGB transform.

CPU qualification must check default compatibility, configuration/CLI guards,
parameter ownership, branch layout and one-hot actions, shared draw placement,
input routing and native graph compilation. Prepare a guarded Tiny native fixture
that compares every branch with ordinary transition/head/value inference, checks
the independent probability-weighted objective and actor derivatives, proves zero
critic/world derivatives from this term, and verifies two synthetic updates,
pre-update snapshots and exact parameter/moment restore. The selected first-action
branch must match ordinary imagination under its shared draw. No finite-difference
oracle through hard action sampling is needed: all actions are enumerated and Q
is held fixed for the actor derivative.

Root `/x/Temp/kindle-actor-expected-value-20261106`. This declaration selects
implementation and CPU checks only. Packaging/native qualification and a bounded
8,192-action cost check require separate declarations after their prerequisites.
A fresh fixed200k learner may be considered only if finite updates, state/target
routing,2GiB projected replay headroom and a conservative30-minute cost cap pass.
No GPU allocation or learner is selected now; no completed stage, failed queue,
coefficient/count/seed sweep or automatic successor may repeat.

Retain Breakout143.667 greedy/70.750 sampled, ordinary inference and32×16.
Judge any learning comparison by frozen own-initial gameplay gain per training
minute, not loss size, derivative correctness or replay fit. The original864/90%
two-wall and five-game gates remain unfinished.
