# Breakout: distinguish value offset and scale from state discrimination

Select a CPU-only analysis of the four completed replay-value critic captures.
The latest sampled critics trail a constant despite improved gameplay. Determine
whether a correction fitted on earlier episodes generalizes to later episodes.
This is a diagnostic of the scalar value output, not an inference or loss change.

Use the unchanged logs under
`/x/Temp/kindle-breakout-replay-value-critic-20261102`: replay-value 0.3 and 1,
each sampled and greedy. Keep the original first three natural episodes per
eight lanes. The first episode in each lane is calibration; the next two are
test episodes. Reconstruct complete discounted external and intrinsic returns
with the recorded F32 discount and zero terminal bootstrap. Require agreement
with every original complete-return metric and episode-start record to 1e-12.
Exclude extra episodes and incomplete tails. No model, emulator or GPU execution.

Fit exactly two corrections on calibration rows only:

- Shift: add the calibration mean of realized return minus predicted value.
- Affine: ordinary least squares of realized return on predicted value, with an
  intercept. Retain the unconstrained fitted slope; reject degenerate variance.

Compare raw, shifted and affine predictions against the original calibration
constant on the sixteen test episodes. Report bias, MAE, RMSE, correlation and
R², the fixed coefficients, and each lane's squared-error sums and row count.
Compute row-weighted MSE differences from the constant with 5,000 resamples of
the eight whole lanes, RNG seed 20261102. Use common resamples across all four
arms. These are conditional descriptive intervals, not learner-seed reliability
or a matched-state effect of replay-value weight.

Use calibration-value quartiles as fixed bin boundaries. Report test count,
predicted mean and realized mean for each bin; preserve empty bins. This gives
a view of state discrimination without choosing bins after seeing test returns.
No additional predictor, tuning, subgroup search or refit follows the result.

Before analysis, qualify synthetic exact-affine and shift cases, test-set
independence, discounted terminal alignment, boundary bins, whole-lane weighting
and degenerate input rejection. Verify all 74 original input pins, capture log
hashes and terminal completion evidence; pin this protocol, analysis source,
wrapper, summaries, reviews and logs. Preserve failures separately if any.
Use a single CPU and a 60-second wrapper cap, with exclusive output files.

Root `/x/Temp/kindle-breakout-value-calibration-20261102`. Complete one analysis
and retain its arrays, lane statistics and execution receipt. Check source pins
before and after. Original gameplay scores, training times, tensors and gates
stay unchanged. A correction that helps is evidence of recoverable scalar
calibration, not proof it would improve actor learning or gameplay; failure
does not prove the recurrent representation lacks useful information. No new
learner or GPU successor is selected. The five-game goal remains unfinished.

CPU qualification passes all seven fixture groups in 0.078s on November 2.
It imports neither the native learner nor ALE. The unchanged qualified script
is pinned before the single saved-log analysis; no previous capture or reader
is repeated.
