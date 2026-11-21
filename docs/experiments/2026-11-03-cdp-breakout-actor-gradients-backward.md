# Backward-only actor-gradient measurement

The [fixed context capture](2026-11-03-cdp-breakout-actor-gradients.md) completes
35,552 actions/142,169 frames in34.280s with zero updates and357 invariant tensors.
Its two B16/T16 batches,32 current scalar contexts and512 raw frames are fixed.
Do not repeat the capture. Source checkpoint and gameplay outcomes stay unchanged.

Use native447fe50 source, with only a cfg(test) module registration and diagnostic
module added. The packaged Python runtime is unchanged. Restore the complete
policy1/replay-value1/expert100 checkpoint, including all optimizer moments.
Compute Auto posterior samples and imagined targets through production methods.
Use3840 imagined rows (B*T*H) and240 replay rows (B*(T−1)), matching the
production constructor. Expert targets have240 unit-weight labels and sum240;
world-graph T/(T−1) normalization does not apply to the behavior graph. Use the
same parameters, inputs and draws for every derivative comparison in each batch. Target-normalizer workspaces and diagnostic RNG may advance in the
discarded runtime; no unchanged transient-state claim is made.

The production behavior graph contains independent actor and critic heads.
Append gates to the original graph prefix for: original total; score-function;
entropy; expert imitation; imagined value; replay value. Recover score-function
from original policy output minus its entropy term, using the original weighted
entropy output and configured coefficient. Apply original policy1,expert100 and
critic1/1 weights. Scalar gates select one term at a time. Disable optimizer and
gradient clipping in both production and diagnostic behavior sessions; do not
call train_behavior or the learning entry point. Keep environment/learner counters
at200,000/12,477 and require all357 before/after/source tensors unchanged.

Before interpretation, require finite inputs, scalars and every derivative;
native resize/reference absolute error<=1e-6; production versus gated-total
gradient relative L2<=1e-3 and maximum error<=1e-3*(1+reference maximum), globally
and separately for actor/critic. Component-sum L2 error must be<=1e-3 times the
sum of component norms, with the same maximum-error bound in each group.
Scalar production/total and component-sum/total errors must be<=1e-3*(1+abs(total)).
F64 CPU dot-product matrices must reproduce saved native matrices within
rtol1e-9/atol1e-8. Require zero critic derivatives for score,entropy,expert and
zero actor derivatives for both value terms. Retain every parameter and both
batches, including zero norms and undefined zero-vector cosines.

CPU graph-prefix/zero-gradient tests, all ordinary Rust CPU tests, strict Clippy,
exact production-source comparison and finite/error-bound fixtures precede
selection. Pin executable, all sources, batches, checkpoint, prior capture
closure and host guard. One original guarded backward service has a180s native
cap and at least2GiB sampled memory headroom. Preserve any failure; no retry,
successful-stage repeat, learner or coefficient sweep follows automatically.

Raw weighted gradients describe these two reused state batches. They do not
measure optimizer movement, original training batches, independent seeds or
causal gameplay effects. In particular, the policy100 regression is not explained
by a large imitation norm alone. Separate score/entropy contributions avoid
attributing the coupled policy-weight change to one component without evidence.

Root `/x/Temp/kindle-breakout-actor-gradients-20261103`; the five-game goal remains
unfinished. No new learner or architecture change is selected by this diagnostic.
