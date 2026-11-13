# Breakout: carry response-error cost to the contact horizon

Select one bounded CPU reference-controller comparison within the five-game
goal. The earlier one-step risk teacher is closed and negative:184.417 versus
225.083, with39 changed first commands in its selected38,223 decisions. It only
penalized uncertainty when contact was within one action. This distinct test
carries the same saved command-error estimates through each plan to contact,
allowing earlier decisions to account for future noisy reversals. The failed
outer-region aiming screen does not select an aiming rule.

Keep the retained RGB detector, landing estimate, edge reset, five-step mean
response, clipping, cached descent, FIRE fallback, action effort and all response
coefficients fixed. Use the nine second moments already fitted on19,146 interior
rows in the completed response-uncertainty experiment. No refit, coefficient
search, new observation or native learner is selected. The penalty coefficient
is1 and the ordinary mean-response planner remains the fallback.

## Fixed contact penalty

Let a be the retained velocity coefficient and s_i the saved second moment for
plan command i and its preceding command. Initialize Pxx=Pxv=Pvv=0. For each
whole action before contact, propagate:

```
next_Pvv = a*a*Pvv + s_i
next_Pxv = a*Pxv + a*a*Pvv + s_i
next_Pxx = Pxx + 2*a*Pxv + a*a*Pvv + s_i
```

For contact h=k+f, with k=floor(h), after k whole steps the added position cost is
`Pxx + 2*f*a*Pxv + (f*a)^2*Pvv + f*f*s_(k+1)`.
At integer h use Pxx alone. This equals the old one-step penalty for h<=1.
Only apply it for .1<=h<=5, with the current paddle center and **every plan's**
predicted centers through ceil(h) strictly inside20–143. If any candidate leaves
this calibrated region, use the unpenalized cost for every plan. This preserves
comparability within a decision and avoids inventing edge-specific uncertainty.
After contact no additional second moments enter the penalty. Original planner
behavior for longer horizons is unchanged.

This covariance formula is exact for independent zero-mean additive velocity
shocks under the unclipped linear recurrence. The saved second moments include
bias and correlated errors, so the result is a planning penalty, not calibrated
contact variance or an established expected-loss estimate. Keeping all eligible
mean paths inside the region does not prove perturbed paths remain inside.

## Qualification and fixed gameplay allocation

Use synthetic independent signed-shock enumeration to verify the penalty for
fractional/integer horizons, command groups and zero noise. Check one-step
equivalence, all-plan region fallback, reset/history handling and zero-penalty
equivalence to the retained controller. These are new fixtures; do not rerun
completed historical tests, fits, collectors or analyses. Verify byte identity
of the reused model data, detector and unchanged planner. Preflight imports may
load ALE but must construct zero emulators and import no Kindle native module.

One new candidate arm uses eight independent environments, full18 actions,
sticky.25, repeat4, noops0, native max-pooled RGB and no RAM policy inputs.
Reuse development seed4000001009 and its original eight lane seeds. Select the
first three natural episodes per lane, retaining excess episodes and tails.
Caps:600k aggregate actions,300s collection,300s full replay,660s controller,
one low-priority CPU core. Retain every action/debug observation and input/output
pixel hash. Replay every candidate decision, reward, frame, reset and outcome
exactly; keep a complete lane0 video. No control rerun or budget extension.

Compare with the retained225.083 mean/0-of24 two-wall teacher. Require at least
10% higher mean, at least five improved lane means, no fewer two-wall successes,
24 natural selected episodes and zero truncations. Report all per-lane scores,
actions, emulator frames, collection/replay time and cap/failure outcomes.
Count penalty eligibility, fallback reasons and changed first commands on the
candidate trajectory, including h<=1, 1<h<=3 and3<h<=5 separately. These counts
compare penalized and ordinary costs at the same candidate states, not paired
trajectory decisions or causal contributions.

Pin exact scripts, native ALE/ROM/adapter, retained comparison, previous negative
result and error-model receipts. Check unchanged hashes before/after execution,
terminal process identities and absence of native/GPU work. Preserve any failure
without overwriting or repeating completed stages. Passing permits a separately
declared fresh-cohort comparison; it does not select integration or training.
Failure retains the existing teacher; no automatic coefficient retry, extension
or successor. Original864/90% two-wall and five-game gates remain unfinished.

Root: `/x/Temp/kindle-breakout-contact-risk-20261106`.
