# Breakout encoder8e-5 with the retained replay critic: bounded check

Select one fresh 8,192-action execution and cost check. Change only encoder
learning rate 4e-5 to 8e-5 from the retained single-image replay-value1 recipe.
The earlier [encoder8e-5 comparison](../results/2026-11-01-cdp-breakout-encoder8e5.md)
improved greedy 104.667 to 125.375 at similar cost, while sampled play stayed
flat. Its behavior rate was 4e-5 and replay-value scale .3. The newer
[retained control](../results/2026-11-02-cdp-breakout-replay-value1.md) uses behavior
8e-5/replay-value1 and scores greedy143.667/sampled70.750. Their combination has
not been assessed. This is a separate interaction experiment, not a repeat of
the earlier life or evidence that its gain transfers to the current critic.

The completed frame-pair comparison costs 13.13% more and fails its frozen
improvement rule. Return to the exact retained single-image native447fe50,
runnercf4ea238 and teacher. No build or architecture change is required.

## Fixed recipe and bounds

Use CDP12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16/context1,
base4e-5/encoder8e-5/dynamics4e-4/behavior8e-5, policy1/imagined-value1/
replay-value1, discount333/lambda.95, expert100/unit weights/world gradient,
all-state corrective labels, 20% guided episodes/cap1920, actor-unimix.01,
sampled collection/posteriors, replay100k/warmup1000. Keep full18/sticky.25/
repeat4/noops0 and one GPU resize of native pooled RGB. There are16,334,353
trainable parameters. Development seed1009/evaluation4000001009 are reused;
this cannot establish independent seed reliability.

Start fresh without transferred weights, replay, games, live belief or RNG.
Train8192 actions/489 updates within180s, first update376/residual credit.5.
Save actual initial and final checkpoints. Restore final for1024 greedy frozen
actions within60s, with no guidance, labels or updates. This short evaluation
is an execution check and cannot establish task competence.

Require357 finite tensors, exact initial values and all47 pre-update semantic
events matching the retained control, invariant frozen tensors, finite aligned
imitation reports, changed CNN/reward/actor/critic, complete training pixel/
action/label replay and both host guards. Preserve at least2GiB measured GPU
headroom. CPU replay cap120s/process150s; audit180s/main600s/collector650s.
No other Kindle GPU allocation may overlap. Stop on real host faults or
numerical failure; no repeated stages or automatic driver recovery.

The cost gate is training seconds times200000/8192 times1.10 at most1800s.
Passing permits a separate full-run decision after original terminal closure;
it does not select a full learner. If selected later, greedy frozen return
minus its actual initial per training minute is the primary comparison, with
sampled play and the original mean864/90%-two-wall gate also reported.
No sweep, cap increase, extension, automatic promotion or check-experience reuse.

## Package and CPU qualification

Root `/x/Temp/kindle-breakout-value1-encoder8e5-20261104`, outputs `check`.
Service `kindle-breakout-value1-encoder8e5-check-20261104` and its `-collect`
observer. Runtime `/x/Temp/kindle-breakout-corrective-20261028/runtime`;
exact retained runner copied without edits. Reuse completed bounded audit and
collector contracts, adapting only the selected rate/probability and paths.
Before launch exercise both actual CLI paths against blocked native/emulator
construction, reject undeclared rates/configuration/cadence/label bounds,
verify357 control shapes and all338 retained inputs, and test the original
collector with saved-output fixtures. Resolve the one historical replayer
from exact retained bytes without rewriting old pins. Pin protocol, helpers,
commands, native/runner/teacher and the completed frame-pair predecessor.
Check live service identity and GPU descriptors immediately before launch.
