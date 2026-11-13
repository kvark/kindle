# CPU attribution through the current actor optimizer

The current [raw-gradient diagnostic](../results/2026-11-03-cdp-breakout-actor-gradients.md)
finds weighted imitation norms6,721×/10,925× imagined score-function on two fixed
batches. Analyze how saved clipping, preconditioning and momentum transform those
same derivatives before selecting another training change. No native update,
emulator replay, learner, coefficient sweep or GPU allocation is selected.

Use both complete batches from `/x/Temp/kindle-breakout-actor-gradients-20261103`,
all22 behavior parameter arrays and their saved first/second moments. Verify the
158 original pins, native result, independent audit and terminal summary. Require
the complete66-tensor behavior checkpoint and saved optimizer step12,477. Both
estimates start independently from that state; neither changes a checkpoint or
feeds into the other. Keep score,entropy,expert,imagined-value and replay-value
components, including cross-head zeros and every parameter.

Reuse the qualified F64 AGC/LaProp equations, with original saved F32 coefficients,
and pin native447fe50 optimizer setup plus Meganeura423d373 runtime/shaders.
Use behavior rate8e-5 for actor and critic at this checkpoint; the base4e-5 and
world-prefix overrides do not set the behavior rate. Apply the real warmup and
actor-enable multiplier. AGC uses each tensor's complete total gradient; LaProp
updates variance from that clipped total, normalizes it and combines it with
saved momentum. Keep per-tensor clipping and the new total-gradient denominator
fixed while transforming each component. These are conditional contributions,
not independent optimizer steps or predictions for a changed loss coefficient.

Require finite parameters,moments,derivatives and estimates, nonnegative saved
variances, exact complete layout and unchanged inputs. F64 raw dot products must
reproduce every saved native parameter/group matrix with rtol1e-9/atol1e-8.
Current/history must reconstruct the complete estimate exactly. Conditional
component sums must reconstruct current contribution with relative L2<=1e-3
and maximum error<=1e-3*(1+maximum current magnitude), globally and per group.
Report all per-parameter/group clipping factors, raw/clipped/component norms,
history/current/full-update directions and update-to-weight ratios for both
batches. Norms are not additive percentages.

CPU fixtures must cover bias correction, saved momentum, cancellation before
clipping, zero gradients, finite-state rejection, input invariance, behavior-rate
fallback/warmup, exact parameter prefixes and actor freezing independently of
critic updates. Block native, JAX and PyTorch imports throughout execution; inspect
open descriptors for GPU devices. Pin protocol,helper,fixtures and source state
before the single analysis, then recheck them and collect terminal CPU evidence.

F64 calculations do not emulate GPU reduction order,F32 rounding,contraction or
parameter-write rounding. The earlier world-optimizer native comparison remains
protocol-failed; it is not validation for measured actor movement. Saved moments
contain mixed training history and cannot be assigned to one objective here.
These two reused sampled-history batches do not establish early-training behavior,
independent-seed reliability or a causal gameplay bottleneck. No coefficient or
architecture change follows automatically.

Evidence root `/x/Temp/kindle-actor-optimizer-20261103`. The five-game goal remains
unfinished; capture and backward stages in the source package are closed and
must not be repeated.
