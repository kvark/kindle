# Breakout: can the posterior correct a drifted forecast state?

The frozen-visual phase is closed and failed retention. Before selecting another
loss, test whether the existing posterior supplies a useful image-informed target
at the forecast's own recurrent state. Earlier state substitutions paired a
teacher-history posterior sample with a forecast-history recurrent state; they
did not recompute the posterior conditioned on that forecast state.

Root `/mnt/data/kindle-breakout-world-posterior-correction-20261107`.
This is a test-only native readout diagnostic, not learning or a new evaluation
of ordinary forecasts. No production graph, API, package or checkpoint changes.

## Fixed query

Use only the retained coverage H0 checkpoint and its cached state-substitution
traces: all four natural batches (32 origins), two ending-at17 batches (10 real
origins plus excluded padding), four original seeds and all32 future steps.
Keep the old masks and exact source identities. Do not reconstruct teacher or
prior transitions, rerun old forecasts, collect data or execute an optimizer.

At every cached h_prior, encode the true current future image and compute
q(z | h_prior, image). Decode h_prior with a sample from this corrected q, using
the original per-step random numbers. Compare with cached prior, teacher-z swap
and full-teacher images at1/4/8/16/32. No corrected state feeds another transition.
Future images are privileged diagnostic inputs, never causal forecast evidence.

Native encoder/posterior/decoder run on Meganeura + Blade. Verify all36 logical
weights against the original checkpoint before/after, zero optimizer age/gradients
in the inference-only sessions, exact regenerated cached prior samples, finite
outputs and the step1 posterior/logit/readout identity with the cached teacher.
The original204-tensor checkpoint bytes must remain unchanged. CPU checks cover
state-block layout and graph parameter scope. No unchanged primitive campaign.

After focused CPU checks and strict Kindle Clippy, declare a single guarded
native diagnostic (900s) with exact test executable/source and input pins. Use
AMD/Auto and the host guard; stop faults/nonfinite values, no recovery/reset/NVML.
There are24 new readout traces,768 posterior-correction steps in batches of8;
no ordinary prior/teacher forecast is executed.

## Interpretation

Report paired visible-ball rescues/harms, detection/location, paddle and brick
errors and categorical KL to the cached prior at every supported horizon.
Preserve all failures. Respect unavailable/reset/terminal masks and repeated
origin/draw clustering; these are reused development data.

For triage only, require corrected natural ball accuracy >=80% at8 and >=70%
at32 before treating this posterior as a promising future-latent training target.
Failing this triage rejects that justification, not all possible posterior losses.
Passing it establishes an available privileged readout correction, not a useful
gradient, learned improvement, calibrated dynamics or world competence. Any
training loss/learner requires a separate justification and declaration; no
automatic successor. Full physics/events/brick-memory/late/tunnel/second-wall,
untouched-family and fresh-confirmation requirements remain unchanged.
