# Frozen Breakout world-gradient attribution

The completed FIRE-weight4 comparison has no score gain. Earlier position
supervision improves fixed state readouts, while more fitting updates lower
imitation loss without improving control. Scalar loss values cannot establish
which objective dominates or opposes useful representation gradients. Select a
fixed-weight diagnostic of the accepted 12M64/B16/T16/H15/R16/expert100 control
before choosing another training change. No new learner or policy evaluation
is selected by this protocol.

Use the accepted checkpoint at
`/x/Temp/kindle-breakout-expert100-capacity-20261029/full-repaired/breakout-12m-64-b16-r16-expert100-200k-seed1009/final`.
Reuse the completed 20,960-action control corpus, its scalar posterior captures
and the already verified corrective labels. For each lane, eligible origins
must have a saved posterior and sixteen subsequent frames without an episode
boundary. Select ranks 20%, 40%, 60% and 80% of that eligible list, rounded down.
Require distinct, nonoverlapping windows. The first two ranks per lane form
one B16/T16 batch; the last two form another. Selection never uses reward,
teacher action, loss or a gradient result. These are two reused diagnostic
batches, not original training replay samples or independent learners.

Replay all recorded actions on CPU and verify every input hash, reward, reset,
frame count and teacher label. Retain the 512 selected native RGB frames,
previous actions/rewards/labels, and cached posterior immediately preceding each
window. Preserve deterministic/stochastic feature order. No native module may
be imported during CPU preparation. A Pillow RGB64 reference is validation data
only: the native diagnostic must resize each raw frame once on the GPU and
match every reference value within 1e-6. Do not feed the CPU resize to the model.

Implement instrumentation only in a private Rust test module. Production graph
construction, learner methods, configuration and packaged runtime remain
unchanged. Restore the complete checkpoint, including optimizer state. Use the
production staged posterior and imagination/target computation at the saved
weights. Generate fresh fixed training-posterior/imagination draws in the
discarded diagnostic runtime, shared by every objective comparison. Supply
corrective targets with the existing T/(T−1) normalization and masked tail.
Then disable the optimizer and clipping for a raw backward pass. Do not invoke
the learning entry point or increment learner/environment counters.

In a separate diagnostic graph, retain the original total loss and add scalar
input gates for its weighted components: CDP prediction, dynamics KL,
representation KL, reward, continuation, replay value, disagreement and expert
imitation. The selected control has no centroid or open-loop term. Measure each
component and their sum with identical inputs, hard draws, weights and one
compiled diagnostic graph. Also measure the original total through that graph.
Compare the latter with the production backward pass both globally and for each reported parameter
group; compare component sums
with the original total. Require finite scalars/gradients, relative gradient
L2 error at most 1e-3 and maximum absolute error at most
`1e-3 * (1 + reference maximum absolute gradient)`. For sum cancellation, bound
L2 discrepancy by 1e-3 times the sum of component norms. Check scalar agreement
with the same relative-plus-absolute tolerance. A failed qualification stays
failed and prevents interpretation of gradient magnitudes.

Report F64-accumulated norms, pairwise dot products/cosines and alignment with
the total, separated into CNN encoder, posterior, recurrent dynamics and other
trainable world parameters. Retain per-parameter results and both batches,
including zeros; do not select a favorable layer or batch afterward. These are
local raw derivatives, not measured optimizer updates, counterfactual gameplay
or proof that a coefficient change helps. Adaptive clipping, optimizer history,
the fixed state distribution and two-batch coverage limit interpretation.

Save before/after checkpoints and require all 357 tensors, including optimizer
moments and frozen actor copies, to be identical, with unchanged learner and
environment counters. Source checkpoints must remain byte-identical. Target
normalization workspaces and diagnostic RNG may advance in the discarded copy;
no claim of unchanged transient runtime state is made.

Before GPU execution, build the test executable in an isolated directory, run
CPU validation and pin sources, executable, fixed inputs and host guard. Use
the RX7900XT under the existing guard, with a 180-second diagnostic cap and
at least 2GiB sampled memory headroom. Stop on real device/numerical failure;
no host recovery, automatic retry, learner extension or coefficient sweep.
Preserve failures and complete any CPU-only evidence repair without repeating
completed GPU work. Check live state before launch and collect terminal evidence
afterward. The five-game goal remains unfinished.

Evidence root: `/x/Temp/kindle-gradient-attribution-20261029`.

## CPU preparation

Full20,960-action CPU replay takes10.520s and verifies every input pixel and
corrective label. The fixed selection retains512 native frames and32 cached
preceding states, with six selection cases passing and zero native import
attempts. Both batches include all eight lanes. The prepared raw frames,
reference-only Pillow resize, metadata and input hashes are retained.

The test executable builds in an isolated directory. All141 Rust CPU tests and
strict Clippy pass. The first fixture incorrectly leaves intrinsic reward zero
while requesting disagreement; that CPU assertion and Clippy's indexed-loop
failure remain recorded with the original source. Explicit fixture coefficient1
and an enumerated loop fix those test-only issues. No GPU diagnostic has run at
this preparation milestone. Production learner methods and packaged runtime
are unchanged; only a `cfg(test)` module is added.
