# Separate reward-learning correction to the imitation actor

The completed actor-update diagnostic finds imitation and saved optimizer history
dominate two retained late-checkpoint updates. Adding dynamics gradients changes
the estimated update by only .01136%/.01702%; its full learner regresses. Actor
width512, policy100 and imitation-decay comparisons also fail their retention
rules. None establishes a causal explanation or justifies a coefficient sweep.

Test a distinct routing hypothesis: preserve the learned imitation actor as a
base, and give the imagined reward objective a separate linear logit correction.
Both heads see the same recurrent feature. During behavior training use
`stop_gradient(base_logits) + residual_logits` for the ordinary score-function
and entropy losses. Expert cross-entropy trains only the base. During acting and
imagination use their sum. The world representation retains its existing expert
loss through the frozen base actor; it receives no residual-policy gradient.
The correction's parameters have separate LaProp moments in the existing behavior
session. No additional optimizer, critic, learned dynamics or relaxed action
gradient is introduced. The existing clipping, schedules and rates still apply.

Add `actor_residual` default false, omitted from legacy serialization. A zero
initialized affine feature-to-action correction adds 46,098 weights/biases at
12M/18 actions. It must leave every existing initial parameter unchanged and
produce the same initial logits. Base supervision and residual RL remain coupled
through their shared feature distribution and combined actions; separate moments
do not imply independent learning. A small linear correction may be insufficient,
or may exploit inaccurate imagined values. No benefit follows from routing alone.

Scope the option to centered RGB CDP with positive constant expert imitation,
world expert gradients, and positive policy weight. Reject actor/world-gradient
coupling, dynamics-gradient and completed-return actor losses, imitation decay,
and actor-update cadence overrides. Preserve geometry and the option on restore;
reject fresh CLI overrides on restore/evaluation and enabled checkpoints on old
native packages. Keep the ordinary actor MLP and critic unchanged.

CPU qualification checks default compatibility, parameter shapes across acting,
imagination and training owners, and graph reachability: policy/entropy may reach
only residual actor parameters, expert loss only base actor parameters, and world
supervision must contain no residual parameters. The dedicated guarded Tiny
fixture must independently check affine logits and categorical-loss derivatives,
zero initialization, unchanged base/critic gradients for an isolated policy loss,
zero residual derivatives for an isolated expert loss, then two finite synthetic
updates, synchronization and exact parameter/moment restore. These checks allocate
no emulator and run once after the package/source receipt is fixed.

Root `/x/Temp/kindle-residual-actor-20261106`. CPU work and packaging first; no
GPU allocation or full learner is selected by this development declaration.
After CPU qualification, separately declare the native fixture (120s cap) and,
if it passes, a fixed 8,192-action cost check plus 1,024 frozen actions. Retain the
12M64/Auto/N8/B16/T16/H15/R16 baseline, encoder/base4e-5, dynamics4e-4,
behavior8e-5, critic1/1, expert100, all-state labels, 20% guided episodes/cap1920,
sampled collection/posteriors, unimix.01, disagreement1, full18/sticky.25/repeat4,
noops0, native RGB and one GPU resize. Coordinates and actor dynamics stay off.
Require finite updates, base/residual changes, full replay, frozen invariance,
2GiB projected replay headroom and conservative200k cost at most30min before
considering a separately declared fresh200k learning comparison.

Use the existing host guard on an idle healthy AMD device; stop on real faults
or numerical failures, never reset/reboot. Preserve original failures. No old
stage repeat, coefficient/width sweep, unchanged extension or automatic learner.
Judge a full comparison by frozen own-initial gameplay gain per training minute,
not imitation fit or derivative magnitude. Retain143.667 greedy/70.750 sampled;
the original864/90%-two-wall gate and five-game goal remain unfinished.
