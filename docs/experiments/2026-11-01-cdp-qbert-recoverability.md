# Qbert teacher takeovers on retained later-play failures

The [life replay](../results/2026-11-01-cdp-qbert-life-replay.md) locates all
96 losses after the first pyramid for the retained encoder4e-5 greedy policy.
Its pixel teacher scores better on its own trajectories, but that does not
establish useful guidance on the learner's states. Select one small CPU
intervention to test whether the unchanged feedback-aware teacher can avoid
recorded losses from those states.

Use the first natural episode in each of eight saved lanes, selected by episode
index rather than score or outcome. These episodes contain 16,613 actions and
32 lost lives. Select every loss in those episodes. For each, start immediately
before the action at `max(previous_loss_action + 1, loss_action - 31)`, using
action one as the initial floor. This gives up to 32 actions through the loss
and never crosses an earlier loss. No origin sweep or later cohort is selected.

Reconstruct the teacher's history from the start of each saved episode by
proposing from native max-pooled pixels and feeding back the actual recorded
command at every step. Fork the process at each origin to preserve the emulator,
sticky-action memory, wrapper and teacher history. One child executes the saved
commands through the original loss. The other follows the same teacher for up
to 64 actions, stopping at its first observed life loss or episode boundary.
The main replay continues only the saved commands. No trained model is loaded.

The primary result is whether teacher control avoids a life loss through the
same original loss-action endpoint. Also retain reward through that endpoint
and separate 64-action survival/reward. A delayed loss is not necessarily a
prevented eventual loss. Count life increases and decreases independently at
each raw frame, so a bonus followed by a loss cannot disappear in a net count.
An imperfect teacher's failure does not prove irrecoverability. This changes an
entire action sequence and cannot attribute the outcome to representation,
forecasting, actor fitting or any particular command.

Require every reconstructed parent input/output pixel hash, action, reward,
frame count, boundary, life event, score and first-pyramid milestone to match
the completed life replay. Validate each recorded child against its saved
trajectory. Reverse branch order at the first origin in each lane and require
identical results; require the parent serialized emulator and teacher state
to remain unchanged after every branch. Preserve full child trajectories and
per-lane outcomes. These eight development episodes are a diagnostic subset,
not a new 24-episode game evaluation or independent reliability evidence.

Eight pure CPU tests pass, including an executed continuation with a bonus and
loss inside one action. The 3.331s preflight verifies 307 input hashes, all 32
origins, exact teacher identity and the original Breakout service/input identity
without constructing an emulator or native model. Its original service may
finish normally during this diagnostic; no restart or replacement is allowed.

Use one low-priority CPU core, a 180-second total analysis cap, 195-second
controller timeout and the existing five-second per-branch transport bound.
The CPU core is the highest allowed affinity index, separate from the lowest
index used by the original Breakout CPU audit. Retain co-host timings without
correcting learner time. No GPU work, learning, teacher change, retry, cap
increase or automatic successor is selected.

Root: `/x/Temp/kindle-qbert-recoverability-20261101`. Source trajectories:
`/x/Temp/kindle-qbert-life-replay-20261101`, from the completed Qbert4e-5 learner.
The existing Breakout8e-5 comparison retains PID742893 / invocation
`eccf5a76eed6458babd75b872c1704e4`; its frozen/audit pipeline remains authoritative.
