# Breakout: one-step response-error penalty at interior imminent contacts

The [fixed command-conditioned second-moment estimate](../results/2026-11-01-cdp-breakout-response-uncertainty.md)
predicts the original teacher's next pooled paddle error better overall and near
descending balls, but remains poorly calibrated at stationary edges. Select one
bounded CPU gameplay test using it only at its measured one-step horizon.
The earlier command-dependent mean controller remains rejected and unchanged.

Keep the original pixel detector, ball/landing tracking, response coefficients,
five-step 243-plan enumeration, clipping, fallback, edge reset and all base costs.
For a predicted contact within one action (`.1 <= collision_steps <= 1`), add
`collision_steps² * command_second_moment` to each plan's original cost. The
group uses its first horizontal command and the previous actual command, with
the existing stationary-edge reset. Fix the penalty coefficient at one.

Apply the penalty only when the current paddle center and **every** candidate
first-step predicted center lie strictly inside20–143, the original training
region. Otherwise use the original plan selection, without a penalty for any
plan. Never allow a plan to avoid the penalty merely by predicting an edge.
Do not propagate uncertainty beyond one step, alter the predicted mean, refit
moments, change detectors or search a penalty coefficient. Retain all new debug
fields: eligibility/reason, selected penalty and base versus penalized choice.

This is a heuristic risk cost. The second moment includes bias; the residual
cross term with intended target error is not modeled. Pooled interpolation,
ball timing and conditional distribution changes limit its interpretation.
No Gaussian, causal-error or calibrated collision-probability claim follows.

Collect the first three natural episodes per each of eight lanes (24 selected)
with retained evaluation seed4000001009, full18/sticky.25/repeat4/native pooled
RGB, no no-op starts and100,000-frame episode limit. Keep extra episodes/tails.
Use the retained225.083333 mean reference without rerunning it. No learner,
environment, teacher state or experience is reused.

Bound collection at600,000 actions/300s, full recorded-action replay at300s and
controller at660s. Require exact candidate actions/debug, live input/output pixel
hashes, rewards, frames, resets, outcomes and video. A cutoff or failed protocol
does not authorize a repeat or cap increase. Report penalty-eligible/changed
decisions and all fallback reasons over both selected episodes and full records.

Require24 natural selected episodes/no truncations, at least10% higher mean than
225.083333, improved mean in at least five of eight lanes, and no fewer two-wall
successes before considering a separate fresh-cohort comparison. The original
two-wall task gate remains. This reused development cohort also supplied error
data; a passing score is not independent reliability or Kindle competence.

Qualify scalar/vector rollout and risk arithmetic, zero-penalty equivalence,
all37,448 original one-step predictions, boundary fallback across every plan,
actual-command history, episode reset and missing-object fallback. Pin the
unchanged coefficients/moments, source receipts, candidate, runner, ROM,
emulator, wrapper and protocol. Use one low-priority CPU core. Preserve active
Qbert PID836919/invocation425aae9cee2541fe9623b753225af57f and all521 inputs
before and after; retain co-host timing without correction. No native import,
RAM use, GPU allocation or learner update. No native teacher integration,
penalty sweep, repeated control, automatic confirmation or learner is selected.

Root `/x/Temp/kindle-breakout-one-step-risk-20261101`. Qbert RGB64 continues
under its original observers; the five-game goal remains unfinished.

CPU qualification passes six fixture groups and 34 pinned inputs in 2.723s.
The original policy/response bytes match their retained source; all 37,448
one-step predictions and zero-penalty baseline decisions reproduce. Emulator
construction is blocked in preflight. All 521 active Qbert inputs and original
process identity remain unchanged. No gameplay has run at selection.
