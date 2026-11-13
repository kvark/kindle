# Breakout: clean waiting-to-launch fitting branches

**Closed.** [Collection and verification](../results/2026-11-08-breakout-world-clean-launch.md).

Select one new fitting-data collection, motivated by the [measured conditional
launch gap](../results/2026-11-08-breakout-world-transition-support.md). The
small64 model has no first-imagined FIRE-alias examples after a stable wait.
This collection covers a different physical condition and all18 commands;
do not repeat the completed paired collector or its source selection.
Root `/mnt/data/kindle-breakout-world-clean-launch-20261108`.

Use eight fresh fitting seeds951001+1009/lane. Each prefix has32 arrivals:
reset, then31 non-FIRE actions. Lanes0–3 start with2/3/4/5 LEFT commands;
lanes4–7 with2/3/4/5 RIGHT commands; the remainder are NOOP. Require positive
constant lives, absent ball reference, no reward/terminal/reset and16 final
NOOP transitions. Save all native pooled RGB, one GPU-resized RGB64 version,
incoming actions/reward/game-over, snapshots and full source transcripts.
References are metadata only. Do not use object/RAM labels as model targets.

Exclude exact native16/32-history matches with any retained development oracle
prefix in all seven categories. Stop on a match or failed prefix condition;
no automatic reseeding, parent replacement or outcome-based selection. Related
initial game configurations remain; this does not create an untouched test.

From every parent run each constant command0–17 for at most32 arrivals, with
two sticky-action repetitions. Use the unchanged qualified reconstruction of
ALE's omitted previous-action state before every branch; preserve chosen RNG
states as in the existing collector. Keep terminal arrivals and short branches,
never pad training truth. Process eight environments together through the
unchanged native camera. No world prediction, derivative or optimizer call.

Bounds:288 branch files,at most9,216 new branch actions/36,864 frames;248 prefix
actions/992 frames;8,928 private-state replay actions/35,712 frames. Total at
most18,392 executed actions/73,568 frames and296 explicit resets,including
reconstruction. Report actual source/replay/new actions and frames separately.
Use the retained small64 final16,384 only as a frozen camera: exact v18
native/source/runner7f797a2b,SHA38680c73,Auto,240 checkpoint tensors. Save before/
after state and require all values/optimizer ages unchanged. No build or new
backend qualification. Existing host guard applies; stop on genuine faults or
nonfinite values; no reset/reload/reboot. Other host work is not ours to stop.

Before allocation pin exact code/helper/runtime/dependencies/ROM, camera state,
all development prefix arrays, prior qualification/closure and protocol. Check
live process/service identity. After collection independently check each
prefix/branch shape, incoming action/outcome alignment, byte hashes, counts,
age/tensor preservation, all18/parent/repetition combinations and source seeds.
Retain every failed run and artifact; no repeat on successful completed stages.
Package the data, publish the result and update PR31. A corrective learner needs
a separately declared fixed budget and sampler after data review; none follows
automatically. Full-world gates and precise physics/event/memory/coverage/fresh
confirmation requirements remain unchanged.
