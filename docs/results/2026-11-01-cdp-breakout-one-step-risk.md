# A one-step response-error penalty lowers Breakout teacher return

The candidate scores **184.417 versus 225.083 (−18.07%)**, with only three of
eight lane means improved. Both policies have **zero of24 two-wall successes**.
It fails the fixed requirement of at least10% mean gain and five improved lanes.
Retain the original teacher; no fresh cohort, penalty retry, integration or
learner is selected.

| Lane | Original mean | One-step risk mean |
| --- | ---: | ---: |
| 0 | 190.333 | 158.667 |
| 1 | 96.667 | 123.667 |
| 2 | 184.667 | 263.333 |
| 3 | 221.000 | 132.333 |
| 4 | 345.000 | 212.667 |
| 5 | 363.667 | 158.333 |
| 6 | 99.000 | 225.333 |
| 7 | 300.333 | 201.000 |

The [fixed protocol](../experiments/2026-11-01-cdp-breakout-one-step-risk.md)
keeps the original mean-response model, detector, landing estimate and five-step
planner. Only when contact is within the next action, and current plus all
candidate next positions are strictly inside20–143, it adds the saved command
error second moment times squared contact fraction. Coefficient1 is fixed.
Elsewhere it uses the original cost for every plan. No mean refit, long-horizon
uncertainty propagation or edge-specific estimate is introduced.

Within the24 selected natural episodes,1,355 of38,223 decisions are eligible;
39 choose a different first command than the unpenalized planner at that same
candidate state. All records contain1,478 eligible decisions and40 changed
commands out of43,056. These are comparisons on the candidate's own trajectory,
not a count of all differences from the original policy's trajectory. The
selected fallback counts are32,973 outside one step,270 current positions
outside the training region,291 candidate-position exclusions and3,334 missing
object fallbacks. All counts and exact debug values remain in the evidence.

The earlier [error-prediction improvement](2026-11-01-cdp-breakout-response-uncertainty.md)
remains valid on its fixed data. This gameplay result does not show that its
errors are irreducible or that uncertainty can never help. The penalty ignores
the residual-bias cross term with target error, uses pooled-position errors as
a contact proxy and changes subsequent state distributions. The reused cohort
also supplied fitting data. Neither independent reliability nor a particular
causal explanation for the decrease is established. This remains a CPU reference
controller comparison, not a trained Kindle result.

Collection completes **43,056 actions /172,188 frames in26.617s**; full replay
takes32.202s, with all24 selected episodes natural and no truncations. Every
candidate action/debug record, live input/output pixel hash, reward, frame,
boundary and outcome reproduces; the replay video is retained. All36 diagnostic
pins and521 active Qbert inputs pass. Six CPU fixture groups and2.723s preflight
pass, including all37,448 original one-step predictions, scalar risk arithmetic,
boundary fallback, zero-penalty equivalence and history/reset checks. The test
log retains a nonfatal unclosed-file ResourceWarning; no test fails.

The controller completes **19:52:07 UTC November1** in61.624s, source`316a857`,
original CPU PIDs838400/838403. Qbert remains originalPID836919/invocation
425aae9cee2541fe9623b753225af57f before and after. Retain CPU co-host time without
correcting training wall time. No failed/repeated stage, control rerun, native
model, RAM input, GPU work or learner update occurs. The original Qbert RGB64
learner continues; the five-game goal remains unfinished.

Root `/x/Temp/kindle-breakout-one-step-risk-20261101` is complete. Do not repeat
qualification, collection or replay.

- Declaration: `6973b4a57130fedbe8c102bb643cd15e52672a7b205a1e1966817f630f90277a`.
- Comparison: `45ca354a6875061cf216f8be1bc5862f4d78c59025bca29cbc59b77126115474`.
- Completion: `9381b052fafe6e7683c4b7bbe560e5b00e0665cbed9c59f6407513139d7e0920`.
