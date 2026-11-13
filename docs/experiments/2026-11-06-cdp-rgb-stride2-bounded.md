# Bounded learning cost of the stride-two RGB encoder

The [development](2026-11-06-cdp-rgb-stride2.md),
[native](2026-11-06-cdp-rgb-stride2-native.md) and
[default-off compatibility](2026-11-06-cdp-rgb-stride2-legacy.md) checks pass.
Native values/derivatives and two synthetic updates/restore pass once; both
default-off1,024-action traces match all128 saved transition rows and all357
checkpoint tensors exactly. No earlier stage is selected for repetition.

Select one fresh8,192-action/489-update cost check, followed by1,024 frozen
greedy actions, eight independent Breakout environments. Only `rgb_stride2=true`
changes the retained learning configuration. Use12M/RGB64,Auto,B16/T16/H15/R16,
world microbatch16,replay100k, encoder4e-5/dynamics4e-4/behavior8e-5, centered
CDP500/reconstruction0, disagreement1, latent/actor mixing.01, critic1/1,
expert imitation100 with world gradients and all-action corrective labels.
Retain20% episode guidance capped at1,920 actions, native RGB teacher, full18
actions, sticky.25, repeat4, no noops, one native max-pool and one GPU resize.
Training seed1009; frozen seed4000001009. This reused development seed is not
an independent replication. All guidance is off during frozen execution.

Production source22c262ea, native SHA256
`cde123e1aa0d68f61cd6111fe392d605d2ae6864de39b4349a139c82301f6b3e`.
Pin the matching current runner, package, teacher, protocol, queue, controllers,
guards and checkpoint schema. Retain all source-specific failures and repairs.
Use existing host guard on AMD RX7900XT/RADV; verify live host identity, no
competing render holder/service and all pins before launch. Cap training180s,
frozen60s, queue300s and CPU audit180s. Stop on real faults/nonfinite values;
no automatic retry/reset/reload/reboot or successor.

Require finite learner reports, correct loss sums and cadence, changed encoder,
reward, continuation, actor and value parameters. All357 initial tensors must
equal the retained actual initial checkpoint; parameter shapes/count stay
16,334,353. Encoder computation changes initial outputs, so pre-update action
equality is not required. Require exact final/frozen tensors, optimizer counts
and config. Independently replay every action/reward/frame/terminal and all
teacher labels/overrides; verify exact guidance policy source definitions.
CPU command checks must exercise both actual queue commands with all model and
emulator construction blocked before launch.

Report aggregate actions, actual emulator frames, elapsed collection/update time,
construction and stage times separately. Compare bounded wall time to the saved
retained8,192-action/489-update control47.689684s; this is a historical comparison
of the same learning recipe, not a paired timing benchmark. Convolution output
positions are quartered, but total training speed must be measured.

Cost gate: training elapsed seconds ×200000/8192 ×1.10 must be at most1,500s
(25 minutes). Memory gate: subtract the entire5,939,200,000-byte replay footprint
from minimum observed native budget headroom, conservatively double-counting
already allocated replay, and require at least2GiB remaining. An exceeded cost
or memory gate is a measured failure, not permission to change caps or extend.

Root `/x/Temp/kindle-rgb-stride2-20261106/bounded`. Check experience cannot feed
a full learner. Even passing gates establishes only numerical execution,
compatibility and cost. A full fixed-budget learning comparison needs a separate
declaration and own actual-initial controls. Retain143.667 greedy/70.750 sampled;
original864/90% two-wall and the five-game goal remain unfinished.
