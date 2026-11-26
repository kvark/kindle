# Fresh Breakout200k conservative-actor assessment

The [bounded comparison](../results/2026-11-03-cdp-conservative-actor.md) passes
target activity, derivatives, compatibility, full replay and cost checks.
Select one fresh200k-action learner with actor100 and explicit observed
lower-bound mode, keeping the retained action-effects1 recipe otherwise.
This is a new development learner within the existing five-game goal.

Use qualified native3f6c6648 and the matching runner/runtime at
`/x/Temp/kindle-conservative-actor-20261103`. Preserve the bounded candidate's
exact settings:12M64/Auto/N8/B16/T16/full BPTT/micro16/H15/R16/context1,
encoder4e-5,dynamics4e-4,behavior8e-5,constant expert100 with world gradient,
policy1/critic1/1,disagreement1,actor100/lower-bound mode. Keep full18,
sticky.25/repeat4/noops0/native pooledRGB/one GPU64 resize, sampled collection
and posteriors,teacher/guidance20%/cap1920 and all retained normalizer settings.
Reuse development seed1009 and evaluation seed4000001009. Transfer no weights,
replay, live state or experience from any previous learner.

Cap training at200k actions/12,477 updates or30 minutes, whichever comes first.
Save the actual zero-experience initial and a read-only100k/6,227 checkpoint
with weights/moments and matching metadata. These are recovery artifacts, not
exact resume state: replay, games, live beliefs and all RNG state are absent.
Require first update376,357 finite/matched initial tensors,47 matching bounded
pre-update events, finite reports and actor coefficient/target accounting.

At completion, assess final and own actual-initial checkpoints with sampled
and greedy actions. Use the retained published protocol and first three natural
episodes in each of eight lanes:24 per arm, no time-limit substitution, zero
updates and357 invariant tensors per frozen arm. Retain full recorded-action
replays/videos, full training pixels/actions/corrective labels, source/runtime/
ROM/host guards, learning curves, guidance exposure and score-band coverage.
Keep guided online scores separate from frozen competence.
Each frozen arm has the retained600k-action bound and a300-second wall cap;
the retained episode limit is100k emulator frames. Cap training replay at900s
and each frozen replay at500s. Preserve any failure instead of increasing caps.

Primary comparison is final greedy against retained200k action-effects1
143.666667 in1287.910495s, own initial2.208333; sampled70.750 with initial1.416667
is secondary. Reuse those already audited arms without another execution.
Report absolute return and initial-subtracted return per actual training minute.
Use10,000 fixed paired lane-bootstrap resamples,seed0, and report95% intervals
for return and gain/min differences. Favor this candidate only if primary mean
and gain/min both increase and the primary gain/min interval excludes zero.
The original mean864 and90% two-wall gate remains; check its task audit directly.
The bonus-off actor95.375/56.542 is contextual evidence, not a matched control.

The new bounded control exactly matches357 old final arrays and1,065 semantic
events; nevertheless native/runner revisions differ from the retained full
learner. Disclose that limitation and the reused seed/cohort. This experiment
cannot establish an isolated loss effect or training-seed reliability. A fresh
confirmation would require later selection after a useful result.

Before launch, exercise the actual CPU postprocessing on compatible retained
fixtures. Use the repaired200k exposure schema and strict semantic comparison
that excludes only recorded wall clocks. Do not copy the known400k-count or
episode-elapsed-time assumptions from the failed original actor-pair helpers.
Require all queued commands, expected configs, schemas, caps, hashes and
observers to be concrete and pinned. Root is
`/x/Temp/kindle-conservative-actor-20261103/full`; it is not launched by this
document alone. Check live state before GPU work. Preserve failures, never
restart successful stages, and schedule no overlapping allocation, cap
extension, coefficient sweep, second learner or automatic successor.
