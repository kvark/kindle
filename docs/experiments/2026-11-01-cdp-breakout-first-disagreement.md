# One early Breakout command followed by the same teacher controller

The [completed teacher-takeover diagnostic](../results/2026-11-01-cdp-breakout-recoverability.md)
avoids108/120 recorded losses from an earlier origin, versus26/120 from the last
near-descending observation. It changes an entire action sequence. Select a
CPU-only intervention to test the local effect of the first command disagreement
in each earlier window. This is architecture diagnosis within the existing
five-game goal; the original greedy-training Qbert learner continues.

Use all120 earlier windows from the retained encoder4e-5 final-greedy Breakout
recording: all lost lives in the fixed24 development episodes. Select the first
exact policy/teacher command disagreement within each window, without using
rescue outcome, phase, confidence or reward. All120 windows contain one;
none is excluded. The original teacher and recorded trajectories must match
through the common prefix before that disagreement.

At the selected state, compare teacher-first and recorded-policy-first, then
use the same feedback-aware teacher for both branches. This changes one command
choice and its consequences for subsequent pixels and teacher memory. Neither
branch recomputes the native policy on changed states. Preserve sticky-action
memory, game, wrapper and teacher state by the qualified process-copy helper.
It already honors the first command, including NOOP; no shared helper changes.

The primary endpoint is survival through the original recorded loss action,
with the same remaining action count for both branches. Report matched reward
and all four paired survival outcomes. Separately measure survival through64
actions from the new disagreement origin, stopping at first lost life or natural
terminal. This endpoint differs from the older window-origin64-action endpoint.
Retain all individual trajectories, per-lane results, action pairs and source
phases. Do not filter results after seeing branch outcomes.

One complete CPU replay must reproduce all37,016 source actions/148,038 frames,
pixels, teacher labels, rewards, lives, boundaries, task outcomes and partial
tails. Source phase rows and aggregate metrics must match exactly. At every
origin, validate the recorded suffix and reproduce the old teacher suffix from
the corresponding earlier takeover. Parent ALE serialization and teacher state
must remain unchanged after each branch. Reverse teacher/policy branch order
at the first origin in each of eight lanes and require identical results.

Teacher failure does not establish optimal irrecoverability. A local command
effect under this controller does not identify representation, forecasting,
actor fitting, timing or coverage as the architectural cause. This failure-
conditioned reused cohort is not a new game score or seed-reliability result.
The source greedy score remains104.667 with0/24 two-wall successes.

Run on one low-priority CPU core,180s analysis cap/195s controller timeout.
No native model, policy inference, GPU context, training, objective change,
retry or automatic successor is selected. Preserve any failure and completed
stages. Check all diagnostic inputs and active QbertPID755504/invocation
`5fb0d30ffb5d47aca552009e586bb929`/317 pins before and after. Keep CPU co-host
timing without correcting Qbert training wall time; never disturb its service.

Nineteen CPU tests cover selection across episodes, earliest exact disagreement,
NOOP, outcome-independent selection, exclusions, duplicate/incomplete windows,
common-prefix equality, matched versus later endpoints, reward accounting and
the actual feedback-aware continuation. Pure fixtures and preflight forbid
native/emulator imports. The retained source replay and process-copy code are
reused unchanged; all original evidence stays immutable.

Root `/x/Temp/kindle-breakout-first-disagreement-20261101`. No Breakout learner,
new loss, policy replacement or successor GPU allocation follows this protocol.

## Preserved ordering failure and separate CPU repair

The source23fe57f controller fails before replay or branching: a list-equality
assertion assumes the old saved origins use selection order. They actually use
execution order. All240 complete origin records are identical when indexed by
stream/episode/step; no missing or changed origin exists. Original command
SHA256`517f9b794bff154b4aa12072b4848b8b6f437e7a719106ee3c3f499b97a75cb1`
and the original309 pins remain preserved. No original origin, replay, review
or completion output exists; no emulator replay or branch has run.

Select a separate CPU repair under `order-repair`. Compare complete origin
records by their unique identities, rejecting duplicates and any changed field.
Preflight now exercises this check on the real saved inputs. Twenty-two CPU
tests pass, including order permutations, duplicate identities and changed
records. Keep the same origins, source replay, branches, endpoints,180s/195s
caps and active Qbert protection. No GPU stage or completed replay repeats.
