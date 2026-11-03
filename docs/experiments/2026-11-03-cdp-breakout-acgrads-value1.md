# Initial imagined actor/critic gradients in the retained Breakout recipe

Select one fresh bounded native check after the completed subaction diagnostic.
The retained replay-value1 recipe scores70.750 sampled/143.667 greedy, with
zero two-wall successes. Its critic calibration remains weak. Changing replay
bootstrap, lambda, life-loss feedback and action timing has not supplied a
better recipe. Test the existing optional actor/critic representation-gradient
path in this supervised12M setting before considering a full learner.

Only `actor_critic_gradient` changes false→true. Native447fe50 already adds
initial imagined policy/value losses to world training through frozen behavior
heads and stopped targets, normalized by imagination length. Later imagined
states remain detached. Reward, continuation, replay-value and expert gradients
are already active; this is additional task supervision, not a missing-loss
repair or differentiation through the entire imagined rollout.

The older unaided4M/minimal-action comparison improved9.542→11.208, while Pong
regressed. Neither predicts the outcome here. The present12M/full18/expert100
recipe already stores the eleven frozen actor tensors, so the expected schema
remains357 tensors and16,334,353 trainable parameters. Verify actual initial
values and exports rather than assuming that expectation.

Use the byte-identical retained runnercf4ea238, teacher and native447fe50/
SHA35ad0db; no source change or build. Keep centeredCDP12M64/Auto/N8/B16/T16/
fullBPTT/microbatch16/H15/R16/lambda.95/discount333,encoder-base4e-5,
behavior8e-5,dynamics4e-4,replay-value1,expert100/unit weights/world gradients/
all-state corrective labels,sampled collection/posteriors,actor-unimix.01,
action-effects1,replay100k,warmup1000,20% episode guidance/cap1920,
learnerseed1009/evaluation4000001009,full18/sticky.25/repeat4/noops0. Preserve
native pooled RGB and one GPU64 resize. Raw rewards and real boundaries stay
unchanged; no life-loss shaping or prior experience transfer.

Require8,192 actions/489 updates, first update376 and residual.5, within180s.
Save actual initial/final checkpoints and perform1,024 frozen greedy actions
within60s, with guidance/labels off and zero updates. Require exact native
config equality except the declared flag, all357 initial tensors matching
control, paired pre-update semantics, finite reports, nonzero initial policy
loss in at least one report and positive initial value loss in all489. Check
changed CNN/reward/actor/value weights, invariant frozen tensors, ordinary
pixel/action/label replay and existing host guards. These checks establish
execution, not independent derivative correctness or gameplay benefit.

Qualify actual commands, fresh-only flag handling, exact config assertions,
cadence, existing357 schema and new report checks on CPU before launch. Keep
main480s, audit180s and original collection600s caps; no overlap, retry,
extension or cap increase. Preserve failures and use a separate CPU repair only
for missing audit work without repeating successful GPU/replay stages.

The conservative200k projection is check training seconds×200000/8192×1.10;
require at most1800s before separately selecting any full200k learner. That
later comparison must use fresh experience,100k recovery and fixed final/actual-
initial sampled/greedy24-natural-episode assessments with the original90%
two-wall gate. A passing bounded check does not select that full learner.

Root `/x/Temp/kindle-breakout-acgrads-value1-20261103`, runtime remains
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No architecture, teacher,
native implementation, global default, coefficient sweep or automatic successor
changes. Reused development seeds and coupled policy/value representation
gradients preclude isolated causal or seed-reliability claims.
