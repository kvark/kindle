# Bounded completed replay-value activity and cost check

The [native qualification](2026-11-06-cdp-completed-value-native.md) passes:
one Tiny fixture in 2.767s, 277 unchanged pins, two direct behavior steps and
two full synthetic updates; default-off final/initial frozen prefixes match
128 transition rows each with 357 invariant tensors. Original packaged Python
fixture failures and their separate repair remain retained.

Select one fresh 8,192-action / 489-update Breakout check, followed by a
1,024-action greedy frozen restore. No old weights, replay, games or experience
transfer. Keep 12M64/Auto/N8/B16/T16/full BPTT/micro16/H15/R16/context1,
base/encoder4e-5, dynamics4e-4, behavior8e-5, policy1/value1/replay-value1,
CDP500, expert100/unit weights/world gradients/all-state corrective labels,
20% guided episodes/cap1920, sampled collection/posteriors and actor/latent
unimix .01. Full18/sticky.25/repeat4/noops0, native pooled RGB/one GPU resize,
training seed1009 and frozen cohort4000001009. Disable model disagreement and
enable completed replay values; leave the completed-return actor off.

Compare cost against the retained bonus-off control's 36.676835s bounded run,
without repeating it. Its full 200k point is 65.208 greedy / 46.458 sampled in
17m05.486s. The primary bonus-on control remains 143.667 / 70.750 in21m27.910s.
Native and runner versions differ from that historical bonus-off control.
Pin current and control identities. Later guidance modules add unused schedules
and action mappings; verify the tracking, predictive and feedback policy class
ASTs are unchanged, and require identical actual-initial tensors and 47
pre-update transitions. Do not claim isolated algorithmic timing effects.

Before GPU launch, check both actual command paths with native/emulator
construction mocked, exact effective configs, 261-tensor schema, target option,
teacher rule and ALE/ROM/adapter identities. Pin scripts, queue, runtime and
control artifacts. Use the existing AMD host guard, current device/boot/driver
identity, Auto and native budget checks; stop on real faults/nonfinite results.
Caps: 180s training, 60s frozen, 300s queue and 180s audit. No driver recovery,
automatic retry, completed-stage repeat or concurrent learner.

Require 8,192 aggregate actions, 489 finite updates, first update at376, matching
initial weights, config/counters, changed CNN/reward/continuation/actor/critic
parameters and correct weighted world/behavior totals. Each report must contain
a valid completed-value row count, with positive activity after a true terminal
has been observed. Count repeated target exposures and first active update;
do not call them unique states. Verify frozen zero updates and all261 tensors
unchanged. Replay all training/frozen actions, rewards, frames and boundaries,
including complete corrective labels and guidance pixel hashes.

Report actual actions and emulator frames, construction/training/frozen time,
stage costs, overrides and minimum observed native memory headroom. The
conservative 200k projection is `training_seconds * 200000/8192 * 1.1`; require
under30min and at least2GiB headroom after subtracting the entire5,939,200,000-byte
full replay footprint. These are projections, not measured full-run cost.
Independently recount target activity and check sealed terminal processes and
artifact hashes after completion. Preserve all failures without relabeling.

Root `/x/Temp/kindle-completed-value-20261106/bounded`. This check establishes
execution, target activity and cost only. It selects no full learner, coefficient
sweep, extension or automatic successor. Keep original864/90% two-wall and the
unfinished five-game goal.
