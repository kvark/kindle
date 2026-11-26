# CPU-only feasibility check of a pooled-pixel paddle-catch signal

The retained Breakout rescue forecast has zero external reward in all240
nine-action branches despite92 teacher rescues. A positive catch event could
provide a nearer outcome signal. This is distinct from the completed negative
life-loss penalty comparison; neither result establishes that reward shaping
will improve gameplay. Select only a saved-data feasibility check within the
five-game goal. No reward coefficient, training change or GPU job is selected.

Use the existing52,072-action/208,254-frame teacher trajectory under
`/x/Temp/kindle-breakout-edge-reset-20261028/standard24/edge-reset-standard24`.
Select its first three completed episodes in each of eight lanes,24 total.
Compare causal pooled observations against the776 already fitted raw-frame
contact candidates in
`/x/Temp/kindle-breakout-teacher-contact-errors-20261101/contact-events.json`,
also reporting its674 accepted-error-fit subset. Verify the prior action,
replay, raw-centroid and contact artifact hash links before reading outcomes.
This is reused teacher-policy data, not the current learned-policy distribution.
The reference is filtered and incomplete: unmatched signals are unresolved,
not proven false positives, and no population precision claim is allowed.

Fix one detector before applying it. At each decision, use only the recorded
pooled `ball_seen`, `paddle_seen` and `ball_y`; never use teacher targets,
commands, future frames, lives or emulator state. Require finite image y in
[0,210). Arm on a rise in y of at least1 pixel with current y at least170.
Emit once on the next fall of at least1 pixel while y remains at least150,
then disarm. Falling below150 clears the arm. Reset on missing paddle,
episode boundary, invalid y or a gap of more than two decisions between valid
ball observations. One missing ball observation may bridge a contact. New
descent is required before a second emission. These geometric thresholds are
fixed without tuning against this diagnostic's outcomes.

The observation before actionj arrives after actionj−1. Attribute an emitted
signal and the corresponding raw game reward to actionj−1, never outgoingj.
Clear that predecessor on reset. Keep cumulative lane-frame coordinates to
match the existing raw contact fits. Match within observed-frame minus fitted
contact-frame in [−2,+12] raw frames, allowing fitted-time uncertainty and up
to three decisions of detection delay. Accept a match only when both signal
and candidate have exactly one eligible neighbor. Retain all ambiguous and
unmatched rows, without nearest-neighbor selection or widening the window.

Report reference coverage, accepted-fit coverage, timing distribution, signal
counts by lane/episode, immediate raw reward, unresolved signals and ambiguous
matches. No inference about task benefit follows from coverage alone. A usable
training aid would still need validation on current-policy data and a separately
declared bounded integration and learner comparison. No automatic successor.

Implement private standard-library analysis and meaningful fixtures for
causality, reset, upward serves, duplicates, missing observations, invalid
coordinates, alignment and ambiguous matching. Pin scripts, protocol and all
consumed evidence in an immutable declaration before the diagnostic. Run once,
with a90-second cap; preserve any failure and repair only unfinished work.
Verify input hashes afterward and independently recount saved output records.
No ALE, native library, GPU, new trajectory, training update, model mutation or
repeated prior replay/analysis. Check live services and record actual CPU time.
Root `/x/Temp/kindle-breakout-catch-signal-20261104`.
