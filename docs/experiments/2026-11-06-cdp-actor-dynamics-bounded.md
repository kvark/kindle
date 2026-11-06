# Bounded dynamics-gradient actor cost and numerics check

The [native qualification](../results/2026-11-06-cdp-actor-dynamics.md) passes,
including default-off frozen legacy trajectories. Separately select one fresh
8,192-action Breakout check with **actor_dynamics_scale=.5**. This coefficient
is a development choice, not a tuned optimum. No full learner is selected.

Keep the retained 12M64/Auto/N8/B16/T16/H15/R16 recipe, seed1009, replay100k,
centered CDP, latent/actor mixing.01, encoder/base4e-5, dynamics4e-4,
behavior8e-5, imagined/replay critic scales1/1 and expert100 with world gradients.
All states receive native-RGB corrective labels;20% episode selection enables
overrides up to1,920 actions. Preserve full18 actions, sticky.25, repeat4,
zero reset no-ops, native pooled RGB and one GPU resize. Coordinates remain off.
The optional actor loss is the sole configuration change. Native and runner
versions also differ from the retained control and are separately qualified.

Use native source744dd793, SHA256
`876ed0c55a9dae95b414aea650d31bfa35544f866c71236da82f937dd50c1a23`,
with exact packaged runtime, copied runner, teacher, helpers and source pins.
Require the same357 initial checkpoint tensors and47 pre-update transitions as
the retained control. Expected first update is action376; total489 updates.
Follow training with1,024 greedy frozen actions at seed4000001009 and zero
updates. Require357 invariant frozen tensors, retained optional configuration,
finite reports and changed encoder, reward, continuation, actor and critic
parameters. Independently recount both weighted losses, including.5 times the
new auxiliary loss, and audit the snapshot timing on every update. Replay all
training actions and pixel labels on CPU under the original environment pins.

Use the existing host guard, one learner and eight environments. Check live
service/device identity before submission. Stop on numerical or real GPU faults;
no reset, recovery, automatic stage repeat or cap increase. Stage caps:180s
training process,60s frozen process,300s queue,180s audit. Retain original
failures; a separate CPU-only repair may complete evidence without rerunning GPU
stages. The parent audit must use this packaged runtime too.

Record construction, training and update-component time, native memory snapshots
and exact actions/frames. Cost gate: training wall time times200000/8192 times1.1
must be at most1,800s. Memory gate: subtract the entire5,939,200,000-byte full
replay footprint from minimum observed headroom; at least2GiB must remain.
This conservatively counts allocated replay pages twice. Report failed cost or
memory gates as results; do not extend the budget or select a full learner
automatically. A successful short check establishes execution and projected
cost only, not gameplay, model-derivative usefulness or efficiency gains.

Root `/x/Temp/kindle-actor-dynamics-20261106/bounded`. Keep weights, logs and
receipts outside Git. Never reuse this check's experience or weights for a
claimed fresh full learner. Original864/90%-two-wall and five-game gates remain
unfinished.
