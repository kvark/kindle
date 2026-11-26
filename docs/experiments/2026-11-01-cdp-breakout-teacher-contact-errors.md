# Fixed Breakout teacher contact-error diagnostic

Select one CPU-only replay of the retained edge-reset pixel teacher trajectory:
52,072 actions/208,254 raw frames, with the fixed first24 natural episodes
retaining225.083 mean and zero two-wall successes. This is diagnostic work
within the five-game goal, not a new policy or a native learner. The completed
H5/R24 Breakout comparison lowers frozen scores despite improved imitation
fit; inspect the imperfect supervision source before choosing another change.

Keep every recorded action, the detector, predictor, teacher parameters, seeds,
sticky actions, resets and episode selection unchanged. Reuse the already
recorded raw ball/paddle centroids and accepted incoming/outgoing contact fits.
The saved reference lacks life-loss timestamps and paddle widths. One new
annotation replay records exact emulator life-count changes and visible paddle
components at each raw frame. Life counts are diagnostic only; they never enter
the teacher or learner. Require the complete recorded policy/debug/input/output
pixel/reward/frame/reset/outcome replay and the exact earlier aggregate raw RGB
hash. No new trajectory, GPU work, agent construction or learner update occurs.

The replay uses one low-priority CPU core, a300-second internal cap and a360s
controller budget. The active Qbert10007 learner must retainPID781585/invocation
`539c22282c4e42c7a22bc150961ef963` and all442 selected inputs before/after work.
No observer is restarted and no GPU job is scheduled. Preserve CPU co-host
timing without correcting the learner's wall time.

## Fixed measurements

- Count all life losses and bonus-life events in the declared24 episodes.
- Extract the unique longest red component of at least4 pixels in the paddle
  region (y185:198,x8:152). Retain visible-width exposure counts. Exclude cropped,
  ambiguous or changing-width interpolation from contact comparisons. Components
  can still merge with the ball; detected width is not physical ground truth.
- Compare each recorded paddle model's one-step prediction with the next pooled
  paddle observation in the same lane/episode, both overall and for projected
  descents at ball-y≥150. Never join across episode resets.
- For each life loss, fit the last four visible raw ball points within128 frames.
  Require gaps≤2 frames, vertical span≥3, last y175–189, positive vertical speed,
  coordinate fit RMSE≤1, and a y187 crossing no more than2 frames before or4
  frames after the last point and no later than the life loss. Record every
  exclusion; do not refit with relaxed thresholds.
- At that inferred crossing, interpolate stable visible paddle position/width.
  Use the latest prior projected near-descending plan with a0–5-decision impact
  horizon, from the preceding128 frames. Report landing-target error, predicted
  paddle-position error, collision-time error, observed/model offsets and offset
  beyond half the detected width. These are descriptive errors, not proof of
  which action caused the loss or exact collision geometry.
- Reuse the previously accepted successful-contact fits, projecting their incoming
  line to the same y187. Their filtering differs from the loss subset; do not
  interpret these counts as unbiased failure probabilities. Width at the life
  decrement is auxiliary and may differ from width at the inferred crossing.

Keep all25 completed episodes and partial tails in replay evidence, while the
analysis selects only the original first three episodes per lane. Report
eligible and rejected counts, width groups and error distributions. Sticky
actions, pooled-observation timing, tracking noise and collision geometry remain
confounded. A stronger teacher, altered labels, loss, architecture or GPU learner
requires a separate evidence-based decision; none is selected by this protocol.

## Qualification and inputs

Sixteen CPU tests pass for known line intersections, exclusion boundaries,
past-only plan selection, fractional impact interpolation, width changes/crop
edges, separate ball components and invalid numerical inputs. The2.429s CPU
preflight passes32 pinned inputs, all11 original raw-replay inputs, prior
completion hashes, replay identity and all442 active Qbert pins. It constructs
no emulator and imports no native model. No historical source substitution or
preflight failure occurs. Do not repeat passed qualification.

Source reference:
`/x/Temp/kindle-breakout-edge-reset-20261028/standard24/edge-reset-standard24`.
Existing raw tracks/contact fits:
`/x/Temp/kindle-breakout-contact-20261028/raw`.
New evidence/code root:
`/x/Temp/kindle-breakout-teacher-contact-errors-20261101`.
The new annotation replay supplies missing evidence; it does not rerun any
completed original controller, collector or analysis stage.

## Completed diagnostic

The44.731s annotation replay and1.420s analysis complete10:25:39 UTC November1,
sourcef658f2a,with48.646s total controller time. Full52,072-action/208,254-frame
policy/debug/pixel/action/outcome replay and the original raw RGB hash pass.
All34 diagnostic/442 active inputs and the original Qbert identity stay unchanged.
No failure,retry,native import,GPU work or repeated original stage occurs.

All120 selected life losses are retained;65 pass the fixed contact filters.
Their mean absolute landing/paddle errors are1.852/4.943 pixels, versus.370/2.317
on674 filtered successful contacts. In38 fitted losses, predicted offset is
inside half the detected width while reconstructed offset is outside. Retain
55 rejected losses and102 rejected prior-success contacts explicitly; this is
not unbiased failure probability or causal diagnosis. The.005s independent
JSON recount verifies24 episode life balances andten scalar means.
See the [result and limits](../results/2026-11-01-cdp-breakout-teacher-contact-errors.md).
No teacher/loss change or successor is selected. Do not repeat completed stages.
