# CPU attribution through the saved Breakout optimizer

The qualified raw-gradient diagnostic finds imitation larger than CDP in the
encoder, posterior and recurrent dynamics on both fixed batches. Before choosing
another learner, analyze how the existing per-parameter AGC and LaProp state
transform those exact captured gradients. This is a CPU calculation at saved
weights, not a native update, new learner, coefficient sweep or gameplay test.

Use both complete batches and every trainable parameter from
`/x/Temp/kindle-gradient-attribution-20261029/auto-corrected`. Verify the prior
139 pins, completion/review hashes, raw gradients and complete saved checkpoint.
Require the gradient layout to match all 86 trainable parameters and their two
optimizer moments. Read the optimizer step from safetensors metadata; verify
12,477 before estimating its next step. Both estimates start independently from
the same saved state; never feed the first estimate into the second.

Pin the Meganeura423d373 AGC/LaProp shader and runtime source, Kindle optimizer
configuration and analysis source. Use F64 calculations on saved F32 tensors,
with the configured F32 coefficients, per-name learning-rate multipliers, warmup
and bias corrections. AGC scales the total gradient independently for each
parameter tensor. LaProp updates its variance from that clipped total, normalizes
the clipped gradient, then combines it with saved momentum. The configured
weight decay is zero and clipping occurs every update. Require all arrays finite
and saved variances nonnegative. Keep zero components and all four parameter
groups visible.

For attribution, hold each total-gradient clipping factor and new variance
denominator fixed while transforming every weighted component. Separate the
current-batch contribution from inherited momentum. Their sum must reconstruct
the full F64 estimate; the transformed component sum must match the current
contribution within relative L2 1e-3 and maximum absolute error
`1e-3 * (1 + maximum absolute current contribution)`, globally and per parameter
group. These conditional contributions are not independent optimizer steps:
removing a loss would change the clipping factor and denominator. Do not report
them as counterfactual gains or additive percentages of update norms.

Report both batches' per-parameter clipping factors, raw/clipped norms,
conditional component norms/directions, current/history/full estimate alignment
and update-to-weight ratios. Preserve all per-parameter values and aggregate
matrices. Validate the equations on independent scalar fixtures, including
nonzero history, first-step bias correction, cancellation before clipping,
zero gradients and unequal learning rates. Block native imports throughout.

F64 arithmetic does not emulate GPU reduction order, F32 rounding, shader
contraction or parameter-write rounding. No claim of measured native parameter
movement follows without a separate qualification. No checkpoint or optimizer
state may be written or changed. Pin inputs before analysis and recheck afterward;
save timing, source, tests and terminal CPU receipts outside Git. No GPU allocation
or architecture/optimizer change is selected by this protocol.

Evidence root: `/x/Temp/kindle-optimizer-attribution-20261029`.

## Completed CPU estimate

The [result](../results/2026-10-29-cdp-breakout-optimizer-attribution.md) completes
in4.150s, source46534b7, with191 input pins, both86-parameter batches and12 CPU
tests passing. Every encoder tensor is clipped on both batches. The estimated
encoder update is much smaller relative to its weights than recurrent dynamics;
inherited momentum changes alignment with the current imitation contribution.
Native optimizer movement and gameplay effects remain unmeasured. Terminal
review23:53:36 UTC October29; zero GPU work or learner updates. No optimizer
setting or new learner is selected.
