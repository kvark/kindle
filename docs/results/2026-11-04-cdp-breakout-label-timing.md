# Saved Breakout actions align best with current corrective labels

The retained learner's late sampled collection agrees most with the teacher's
**current command:52.73%**, versus43.02% for the preceding command. Every tested
shift lowers agreement in each of eight lanes. This does not support a simple
fixed-delay explanation for the observed imitation gap. It does not establish
the cause of that gap or rule out delays on particular states.

The exact retained source review also finds no off-by-one mismatch in the
inspected label path. No labels, policy, loss, runtime or training settings change.
Frozen greedy143.667/sampled70.750 and zero-of24 two-wall successes remain the
retained result. The original864/90% gate and five-game goal remain unfinished.

## Matched timing comparison

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-label-timing.md)
uses the existing200k-action training log. Its primary late subset contains
**24,998 anchors** from the final50k aggregate actions. Every nine-action window
is consecutive, entirely unaided, within one lane/episode/quarter, and contains
only four-frame transitions with NOOP/RIGHT/LEFT teacher labels. Identical
anchors are used for every offset; actual policy actions retain the full18-action
vocabulary. Teacher-executed actions during guidance are excluded.

Offset k compares recorded action a[t] with teacher label e[t+k]. Negative
offsets use earlier labels, positive offsets later labels. Teacher self-agreement
compares e[t] with e[t+k], describing label persistence on those same windows.

| Teacher offset, in actions | Policy agreement | Teacher self-agreement |
|---:|---:|---:|
|−4|37.54%|39.81%|
|−3|38.80%|41.27%|
|−2|40.76%|44.28%|
|−1|43.02%|47.61%|
|0|**52.73%**|100.00%|
|+1|36.11%|47.36%|
|+2|35.81%|43.93%|
|+3|35.11%|41.04%|
|+4|35.13%|39.62%|

Current-command matches are13,182, versus10,755 for the preceding command:
a9.71-percentage-point difference on identical anchors. The independent-marginal
match reference is33.02% at offset0; it is a descriptive reference, not a
randomized control or significance test. All shifted comparisons are lower in
all eight lanes.

The13,096 anchors where the teacher changes command also favor the current
label:47.59% agreement, versus29.05% with the preceding label and31.43% with the
next. Current agreement is again higher than every shift in all lanes. Teacher
commands change frequently here; these observations do not identify whether
changes are necessary, noisy or difficult for the network to reproduce.

The secondary late subset permits FIRE and has34,060 anchors:current agreement
55.14%, preceding42.95%, next36.49%. The early directional subset has10,079
anchors and current32.42% versus preceding31.28%; these are different states and
evolving sampled policies, so their difference is not a matched learning effect.
All six predefined groups, action/label marginals and per-lane counts are saved.

## Source alignment and limits

The saved runnercf4ea238 computes corrective labels from current frames before
calling `act`, then feeds back the executed command. In native447fe50, pending
labels accompany the next arriving observation. Replay's `expert_targets` pairs
state t with `previous_expert_action[t+1]`, excluding terminal source states and
reset crossings. The final sequence state receives no expert target. World
supervision uses the same aligned rows with T/(T−1) normalization. The live
observation/posterior update precedes the next policy call.

Four exact native source snapshots and the inspected locations are retained.
This is a source review, not another native execution or qualification. Earlier
native alignment tests and captured target checks remain separate evidence;
they are not rerun or counted as new tests here.

The data describe sampled training behavior, not the final frozen greedy policy.
Labels use executed-action feedback; future labels also respond to current
actions' effects. These dependencies, persistence and state differences prevent
reading a lag curve as causal reaction time. Natural episode boundaries are
checked, but the log has no life boundaries. Directional filtering does not
guarantee removal of every life boundary. Agreement with an imperfect teacher is
not optimality or gameplay success. Representation precision, recurrent state,
optimization and teacher quality remain unresolved; no label shift or successor
learner is selected.

## Verification and closure

All200,000 actions/799,595 frames,263 completed episodes,86,117 overrides and
113,883 native actions reconcile with the original run_end and exposure audit.
The old complete replay is reused as evidence; no emulator replay occurs.
The new analysis preserves52,477 anchor rows, with overlapping subgroup tags.
Eight hand-built CPU tests pass for lag sign, matched windows, boundaries,
overrides, shortened transitions, FIRE/switch filtering, malformed sequences
and marginals. All18 declared input hashes remain unchanged.

Analysis completes **20:02:57.209 UTC November4**, source`185e3180`, in1.067s
(1.124s process wrapper). A separate1.198s recount completes **20:03:15.242**,
verifying all54 group/offset counts, marginals and per-lane results directly from
saved anchors. Qualification takes.091s. No failure, prior-stage repeat, native
or emulator import, GPU work or learning update occurs. No Kindle service is
active before or after. Training time is unchanged.

Root `/x/Temp/kindle-breakout-label-timing-20261104` is closed. It retains the
protocol, exact sources, scripts, tests, all anchor rows and execution receipts.
Do not repeat completed stages. Check live state before separately selected work.

| Artifact | SHA256 |
|---|---|
| Declaration |`e0b3b0cfebe1ae96f8774813f9a67683df3a2e7e319ad4a5b792925586e6a0d3`|
| Analysis |`0a83da86ecf3b52cd9edff031bc6588a98eb8c4b51cf448454d7b50b5918ea4e`|
| Completion/recount |`8f2defd3cf8df296cfd577ff9109447dec8ddf88c6322a29c577d622a3c836a0`|
