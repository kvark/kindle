# Breakout life-loss reward visibility in saved trajectories

Select one CPU-only analysis of existing life-annotated trajectories before
adding a training signal. The latest replay16k comparison is closed and failed
to improve gameplay. No learner or GPU job is selected by this analysis.

Use the completed encoder4e5 own-policy capture under
`/x/Temp/kindle-encoder4e5-policy-phase-20261101`, whose candidate rows describe
37,016 recorded actions/148,038 frames and the fixed first three natural episodes
per eight lanes. Its saved life-window report contains120 selected lost lives,
24 at game-over. Reuse these rows and windows; do not replay the emulator,
recapture features, refit a readout or rerun the earlier analysis.

Verify the original completion artifact hashes and join every saved row to the
original action/reward/boundary log, including unselected extra episodes and
partial tails. Require unique ordered lane/episode/step rows, continuous life
counts within episodes, correct selection and exact action/reward/boundary
agreement. Reproduce the saved selected score and life-event identities.

Report raw reward at each lost-life event, separating game-over from earlier
losses; retain bonus-life gains and all events. Calculate the hypothetical
training reward `raw_reward - max(0, lives_before - lives_after)` for each row,
without changing any logged reward or running a model. A one-point penalty is
an explicit sparse training aid, not the environment's original reward. Do not
claim an improved policy or evaluate these hypothetical totals as game scores.

Inspect the exact retained runner's observe call to confirm raw environment
rewards are passed directly to the native learner. Pin the current audit rule
that requires stored extrinsic reward to equal raw reward; a future feature
must extend this contract explicitly, preserve raw score accounting, default
off, and disable shaping during frozen evaluation. No implementation is selected
until this analysis is complete.

Use one low-priority CPU core,60-second process cap, pure-CPU fixtures for life
loss/gain/terminal cases and corrupt data rejection. Forbid native and emulator
imports, verify pinned inputs before and after, and retain failures. The data
come from an older encoder4e5/behavior4e5/replay-value.3 greedy policy, not the
latest replay-value1 policy. Net per-decision life changes may not expose a
gain and loss within the same action repeat. No causal claim or seed reliability
follows from these observed events.

Root `/x/Temp/kindle-breakout-life-feedback-20261103`. The original five-game
goal and two-wall acceptance gate remain unchanged.
