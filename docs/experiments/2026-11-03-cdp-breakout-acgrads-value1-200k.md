# Fresh Breakout actor/critic representation-gradient comparison

Separately select one fresh 200,000-action learner after the bounded check
passes execution, active-loss validation and a conservative 21m10s projection.
The question is whether the existing optional representation-gradient path
improves frozen gameplay within the compute budget. Retained control scores
70.750 sampled / 143.667 greedy, with zero of 24 two-wall successes in both.

Only `actor_critic_gradient` changes false→true. Initial imagined policy/value
losses train the representation through frozen behavior heads, stopped targets
and H15 normalization; later imagined states remain detached. Keep native
447fe50/SHA35ad0db, byte-identical runnercf4ea238 and teacher. Keep centered CDP
12M64/Auto/N8/B16/T16/full BPTT/micro16/H15/R16/lambda.95/discount333,
encoder/base4e-5, behavior8e-5, dynamics4e-4, replay-value1, expert100 with unit
action weights/world gradients/all-state corrective labels, sampled collection
and posteriors, unimix.01, action-effects1, replay100k, warmup1000, 20% episode
guidance with cap1920, learner seed1009 and evaluation seed4000001009.
Preserve full18/sticky.25/repeat4/noops0, native pooled RGB, one GPU64 resize,
raw rewards and true boundaries. Transfer no weights, replay, beliefs, game
state, RNG state or bounded-check experience.

Require 200k actions / 12,477 updates, first update376 and residual credit.5,
within 1,800 seconds. Retain the 100k / 6,227-update recovery checkpoint and
actual initial/final checkpoints. Require 357 initial tensors matching bounded
and control starts, 47 paired pre-update events, finite reports, nonzero initial
policy loss in at least one report, positive initial value loss in every report,
changed CNN/reward/actor/value, full training pixel/action/label replay and host
guards. Loss checks establish execution, not independent derivative correctness.

Assess fixed final and actual-initial checkpoints with sampled and greedy
actions: first three natural episodes in each of eight lanes, 24 per arm,
guidance and labels off, zero updates/truncations, 357 invariant tensors,
recorded-action replays/videos and unchanged 90% two-wall success criterion.
Use sampled as the primary comparison and report greedy separately; compare
absolute scores and initial-subtracted score per training minute. Preserve both
if they disagree; do not choose a checkpoint or evaluation setting afterward.
Ordinary frozen logs lack live pixel hashes, so their replay checks outcomes,
frames and boundaries rather than unavailable live-image equality.

Qualify all five actual commands, exact-config assertions, flag handling,
startup/recovery observers, source paths, schema and checkpoint retention on CPU
before declaration and launch. Training cap1800s; each frozen arm300s; audit1200s;
main service4500s. Launch once, with original progress/recovery/postprocessing/
collection observers. No overlap, retry, extension, cap increase, coefficient
sweep, confirmation learner or automatic successor. Preserve failures; any
needed CPU repair must retain successful GPU/replay stages unchanged.

Root `/x/Temp/kindle-breakout-acgrads-value1-20261103/full`; runtime remains
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Pin the qualified sources,
bounded closure, retained control, exact commands and host before launch.
Record actual guidance exposure, curves and CPU co-host timings without time
correction. Reused development seed/cohort and jointly changed policy/value
gradients preclude isolated mechanism or seed-reliability claims. No gameplay
gain is available at selection; Breakout and the five-game goal remain unfinished.
