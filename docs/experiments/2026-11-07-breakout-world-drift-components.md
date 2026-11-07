# Breakout: separate shared recurrent drift from sample dispersion

The [recurrent-energy learner](../results/2026-11-07-breakout-world-recurrent-energy.md)
fails physical retention. Before selecting another learner, inspect the retained
[state-substitution traces](../results/2026-11-07-breakout-world-state-swap.md).
This is a new CPU-only statistic of completed measurements, not another model
query, collection, training run or repeat of the earlier readout scoring.

Root `/mnt/data/kindle-breakout-world-drift-components-20261107`. Use exactly the
two retained coverage/latent-overshooting models, all 32 natural and 10
terminal-aligned origins, four fixed draws and all 32 horizons. Respect original
validity/real-lane masks; include terminal arrivals and exclude reset/unavailable
states and padding. Preserve the original state-query failure and corrected
completion separately. Do not infer these older traces describe the newest model.

For each physical origin/time, let delta_d=h_prior,d-h_teacher,d. With four
draws, decompose mean squared error exactly into squared mean delta plus
mean squared centered delta. Also retain prior/teacher dispersion and paired
covariance, verifying their equivalent identity. Report origin-averaged and
per-origin results, not just a pooled favorable example. Show the finite-draw
unbiased estimate of squared population mean delta using sample variance/n;
negative estimates remain negative, not silently clipped.

Compute mixed prior/teacher categorical entropy, categorical disagreement across
draws and recurrent RMS norms from the already saved states/logits. Verify shapes,
finite values, one-hot layout and probability normalization. These are internal
descriptive statistics, not a common cross-model physical score. Analyze all
horizons; summarize 1/4/8/16/32 plus ending17 where supported.

Before looking at results, use the natural-origin empirical mean-delta fraction
at8/16 in both models as a limited triage rule: all >=2/3 favors examining shared
transition/representation drift before suppressing samples; all <=1/3 favors a
separate frozen sampling intervention; otherwise the evidence is mixed. Four
draws give noisy mean/variance estimates; teacher states are stochastic and use
future images, and prior/teacher draws share random numbers. No causal attribution,
unbiased physical-error claim or automatic learner follows from this rule.

Pin the original data and analysis receipts, corrected settings/query ledger,
selected raw traces and the exact new script/protocol before execution. Rehash
after analysis. A tiny independent algebra fixture and full per-origin identities
must pass. No GPU/native inference/emulator construction, optimizer, package or
production graph change. Keep actor/critic and CDP efficiency parked. The full
world-model goal remains active and unproven; this diagnosis selects at most the
next question, not success or an unchanged extension.

Preserve the initial CPU declaration failure on the old receipt's live repository
runner reference, now changed by the recurrent-energy implementation. Verify the
original source member against its old expected hash in the previously pinned
archive, retain it locally and record that provenance substitution explicitly.
Never replace the historical hash with today's runner or mutate the old receipt.
No data, state query or analysis result was produced by the failed declaration.
