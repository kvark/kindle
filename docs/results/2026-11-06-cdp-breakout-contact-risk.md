# Contact-horizon response-error cost does not improve the Breakout teacher

The fixed CPU comparison fails retention: **206.250 versus225.083 (−8.367%)**,
with only **three of eight lane means** improved. Both teachers remain **0/24
two-wall successes**. Retain the original teacher and native learning recipe.
No fresh cohort, coefficient retry, integration, learner or successor is selected.

| Lane | Retained teacher | Contact-horizon penalty |
|---|---:|---:|
|0|190.333|161.667|
|1|96.667|211.000|
|2|184.667|183.667|
|3|221.000|298.000|
|4|345.000|228.333|
|5|363.667|251.000|
|6|99.000|130.333|
|7|300.333|186.000|

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-contact-risk.md)
required at least10% higher mean, five improved lanes, no fewer two-wall successes
and24 natural selected episodes with zero truncations. All episode-completion
requirements pass, but mean/lane requirements fail. One reused development cohort
does not establish training-seed reliability or a causal explanation of the
decrease. These are hand-written pixel controllers, not learned Kindle policies.

## Intervention and measured coverage

The candidate keeps the original detector, mean-response coefficients, five-step
planner, landing target, clipping, edge reset and missing-object fallback. It
adds a coefficient1 penalty from the nine previously fitted command-error second
moments, propagated through the retained linear velocity recurrence to contact.
Only the contact prefix contributes. If contact is beyond five actions, current
position is outside20–143, or any candidate's prefix leaves that region, every
plan uses its original cost. No response model is refitted.

The earlier [one-step risk policy](2026-11-01-cdp-breakout-one-step-risk.md)
remains closed and negative. The new penalty equals it within one action, but
also affects earlier planning decisions. Six new fixtures verify independent
signed-shock arithmetic, fractional/integer horizons, one-step equivalence,
all-plan fallback, zero-penalty legacy behavior and reset/history handling.

| Contact horizon, selected episodes | Eligible decisions | Changed first command |
|---|---:|---:|
| At most one action |1,326|49|
| Above one, at most three |879|110|
| Above three, at most five |0|0|
| Total |2,205|159|

The earlier-planning intervention therefore executes, while the all-plan region
gate leaves no eligible3–5-action cases. It should not be described as an
empirical five-action uncertainty test. The selected episodes contain40,195
decisions:28,137 have contact beyond five actions,4,833 fail the candidate-prefix
region gate,927 fail the current-position gate and4,093 lack objects. Full
collection including tails has46,200 decisions,2,514 eligible and182 changed.
Changes compare penalized versus unpenalized costs at the same candidate states;
they are not counts of differences from the retained teacher trajectory.

The propagated cost is exact only for a declared independent zero-mean additive
shock surrogate without clipping. The saved second moments include bias and
correlated errors. Mean paths inside the calibration region do not guarantee
perturbed paths remain there. The penalty is not calibrated contact variance;
this result neither proves response errors irreducible nor rejects every method
for planning under uncertainty.

## Execution and validation

One candidate collection completes **46,200 actions /184,763 emulator frames in
32.161830s**. All24 completed episodes are the selected first three per lane;
eight partial tails are retained. Maximum complete-episode score is425 and no
tail reaches432. There are zero truncations. The complete **35.692589s replay**
reproduces every decision/debug record, input/output pixel hash, reward, frame,
boundary, reset and task outcome, retaining a full lane0 video. The historical
control is reused without another rollout.

Preflight verifies six fixture groups,44 inputs and11 historical sealed artifacts,
constructing zero emulators. The ROM, native ALE, preprocessing and task observer
match the retained control bytes. Reused detector/model files and the mean-planner
class remain exact. Native Kindle is never imported; there are zero learner
updates and no GPU work.

Source `2ce19be9e35116bf200885c3573c3bcf98d697f3`; CPU controllerPID1516160 and
childPID1516164. Controller completes in **68.250249s at16:53:43.976 UTC November6**.
All45 declared pins and terminal child checks pass, with no active Kindle service
before/after and no failed or repeated stage. A separate.222782s reader finishes
16:55:32.367, verifying all23 completion artifacts, every ledger reward/frame,
24 episode totals, intervention counts and actual selected commands independently
of the policy implementation. No collector, fit or qualification is repeated.

Evidence root: `/x/Temp/kindle-breakout-contact-risk-20261106`. Exact scripts,
model copies, source identities, declarations, logs, complete action ledger,
replay/video, comparison, decision counts and closure/recount receipts remain
outside Git. Do not restart this experiment. Original teacher/guidance and
Breakout143.667 greedy/70.750 sampled remain retained; original864/90%-two-wall
and the five-game goal remain unfinished.
