# Fixed teacher takeovers on current Breakout policy failures

Investigate whether the saved command-dependent paddle predictor improves
recovery from the retained replay-value1 policy's imminent life losses. The
older predictor improves CPU reference means225.083→295.333 on development
and250.875→273.125 on a fresh cohort, but the fresh8.87% gain fails its fixed10%
criterion. That comparison remains failed; this selects no promotion or learner.
The new question concerns paired interventions on the current student's states,
not another full-game retry or a revised teacher-score threshold.

Use the completed retained final-greedy log scoring143.667, first three natural
episodes per eight lanes, with full18/sticky.25/repeat4/noops0/native pooled RGB.
One new CPU annotation replay identifies every observed life decrement in those
24 episodes and records exact input/output pixels, actions, rewards, frames,
resets and life counts. Reconcile all original episodes, partial tails and
frames with the accepted replay. No lives/RAM values enter either policy.

For every selected life loss at zero-based decisionL, select origin
max(first decision after the preceding life loss or reset, L−8). Selection
uses only the recorded trajectory, before either intervention is evaluated.
Retain short available windows and all selected losses; do not substitute
origins or tune the lookback. If duplicate origins or a life increase makes
accounting ambiguous, stop and preserve evidence rather than silently dropping
cases. Expected120 losses is a checkable expectation, not an outcome filter.

A second replay supplies three process-copy branches at each fixed origin:
exact recorded actions through the original loss, existing feedback-aware
pixel teacher, and saved command-dependent teacher. Both teachers receive the
same historical pixels and actually executed actions before takeover; the new
adapter replaces each proposed previous-command history with the executed
command, preserving the candidate's older command and observed-velocity rules.
The candidate predictor, coefficients, detector, clipping and planner are
byte-identical to its saved implementation. Validate this feedback boundary
before collection. No teacher sees life counts or future observations.

Each teacher follows its own closed-loop choices for32 four-frame actions,
stopping at the first life decrement, natural terminal or original time limit.
The primary outcome is completing all32 actions without life loss or truncation.
Also retain avoidance at the original loss endpoint, rewards and first-loss
latency; do not use them to replace the primary decision. Report paired both/
existing-only/candidate-only/neither outcomes and counts for every lane.

The fixed development screen requires at least10 percentage points higher
32-action survival over all selected origins and strictly higher survival in
at least five of eight lanes. Passing only supports separately considering
training utility; it proves neither full-game success nor independent reliability.
Reject incomplete, truncated or divergent branches. No window/coefficient
sweep, extra cohort, teacher integration or automatic GPU successor follows.

Use process copies to preserve ALE sticky-action memory and RNG. Every recorded
branch must reproduce pixels/actions/rewards/lives/boundaries/counters through
its known loss. Check parent ALE state, pooled buffers, counters and both policy
histories before/after each branch group. Reverse branch order at the first
origin of each lane and require exact results. Full second replay must reproduce
every annotation row and original task outcome, including extra episodes/tails.
Original frozen logs have no live pixel hashes; equality of the two CPU replays
is not a claim of original live-image equality.

Before selection, verify accepted source/teacher closures, exact inputs,
feedback/reset handling, within-life origin selection, primary screen arithmetic,
32-action accounting and branch-copy parity. Block native imports; one affined
low-priority CPU core, zero GPU work or learner updates. Annotation cap120s,
branch replay300s, per-branch5s and controller480s; never retry a completed stage.
Require clean committed source and check live Kindle state. Preserve failures
and all partial evidence; any CPU repair must be separate.

Root `/x/Temp/kindle-breakout-command-takeovers-20261104`. Retain declaration,
original source identities, full annotation, fixed origins, every branch,
receipts and terminal recount. Update PR31 and linked reports. The retained
Kindle recipe, original864/90%-two-wall gate and unfinished five-game goal stay.
