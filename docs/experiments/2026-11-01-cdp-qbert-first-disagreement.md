# Qbert first-disagreement action intervention

The [exact actor readout](../results/2026-11-01-cdp-qbert-policy-readback.md)
finds weak movement agreement in the retained later-play failure windows.
The teacher avoids 25 of 32 recorded losses, but it changes a sequence of
commands. Select one CPU experiment within the five-game goal to measure the
effect of the first differing command under a fixed continuation policy.

Use all 32 windows from the completed teacher-takeover diagnostic: every lost
life in the first natural episode of each of eight retained lanes. Select the
first recorded-policy/teacher disagreement in each window, using labels computed
with the complete original episode history and actual-action feedback. There
are 25 such windows; retain the other seven explicitly as having no disagreement.
Selection does not use rescue outcome, score, confidence or direction. In this
existing corpus the selected windows coincide with the teacher-rescued windows.

Reconstruct all 16,613 original actions and 66,440 frames. At each selected origin,
fork the CPU process to preserve emulator sticky-action memory, wrapper state
and teacher history. One child submits the teacher's first command. A second
submits the original policy's first command. Both then use the unchanged
feedback-aware pixel teacher, stopping at the first raw-frame life loss, episode
boundary or 64 actions. Feed back the actual first command: it can alter both
the physical trajectory and subsequent teacher memory and decisions. This is
a one-command intervention followed by the same closed-loop controller, not
a comparison with identical subsequent commands or a native-policy rollout.

The primary paired endpoint is survival through the original recorded loss
action. Report both survive, teacher-only, policy-first-only and neither.
Separately report 64-action survival from the new disagreement origin, reward
through the matched endpoint and reward through the branch's stop. A later
loss is not necessarily prevented eventual loss. Keep first-loss positions,
per-lane outcomes and exact saved actor probabilities for these origins.
Join probabilities from the completed native readout; no new inference occurs.

Require the complete original parent pixels, rewards, frames, boundaries,
score, life gains/losses and first-pyramid event to reproduce. Validate a third
child's recorded commands through the original loss. Before the first
disagreement, teacher and recorded commands coincide; require the teacher
branch's available prefix to equal the suffix of the earlier retained teacher
takeover. The new 64-action endpoint can extend beyond that older branch.
Reverse the order of the two intervention branches at the first selected origin
in each lane. Require exact branch equality and unchanged serialized parent
emulator and teacher state after every child. Count raw-frame gains and losses
separately, so a bonus cannot mask a loss within one repeated action.

Eleven CPU tests pass, including forced-first-NOOP execution, feedback rejection,
subsequent teacher control, raw bonus/loss counting, first-disagreement selection
independent of rescue outcome and paired endpoint accounting. The 1.415-second
preflight checks 550 input hashes, exact teacher identity and all 25 origins
without importing an emulator or native model. No Kindle service is running.

Use one low-priority CPU core, a 180-second analysis cap, a 195-second controller
timeout and the existing five-second branch readiness bound. Preserve all
trajectories and immutable inputs under
`/x/Temp/kindle-qbert-first-disagreement-20261101`.

This reused, failure-selected subset measures an action's effect under this
teacher, not optimal actions, average-policy accuracy, independent reliability,
representation sufficiency or an architectural cause. No training, GPU work,
loss change, teacher change, retry, cap increase or automatic successor is
selected. The five-game goal remains unfinished.
