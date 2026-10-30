# Breakout raw gradients: imitation is already a major representation objective

The [native follow-up](2026-10-30-cdp-breakout-native-optimizer.md) measures similar
group movement, but its full comparison remains protocol-failed on six scalar
F64 rounding allowances. The source corpus is unaided greedy evaluation with
offline teacher labels; earlier prose calling it teacher-guided was incorrect.

Two fixed B16/T16 batches from the accepted Breakout control show larger weighted
expert-imitation gradient norms than CDP in the CNN encoder, posterior and recurrent
dynamics. Total-gradient alignment with imitation is 0.925–0.997 across those six
group/batch measurements. CDP and imitation point partly in the same direction;
the KL terms oppose imitation in several groups. This does not support a simple
claim that CDP overwhelms imitation on these saved states. It does not establish
which training change would improve play.

The corrected guarded diagnostic succeeds 23:40:08 UTC October 29, source `d3283ba`.
The original startup assertion and failed guard remain preserved. There are zero
learner updates or gameplay actions, and all 357 tensors remain exactly equal to
the source checkpoint. The five-game goal remains unfinished. No new learner,
coefficient change or automatic GPU successor is selected.

## Fixed measurement and qualification

The [protocol](../experiments/2026-10-29-cdp-breakout-gradient-attribution.md)
uses the accepted 12M64/Auto/B16/T16/H15/R16/expert100 control at 200k actions /
12,477 updates. Two batches reuse 512 native frames and 32 preceding cached
posterior states from the completed 20,960-action control corpus. Eligible-window
ranks 20/40/60/80% are fixed per lane, without selecting on rewards, labels,
losses or gradients. Full CPU replay verifies pixels/actions/rewards/reset/frame
counts and teacher labels in 10.520s, with zero native imports.

The test restores the checkpoint and uses production Auto posterior sampling
and imagination/target computation. The ordinary world-training graph receives
those hard samples and the same late targets. Optimizer updates and clipping
are disabled. A second graph adds input gates around the already weighted loss
components, retaining the original graph prefix and outputs. All component
measurements use identical inputs, draws and parameters within each batch.
Saved actual graph inputs, all parameter gradients, pairwise F64 dot products,
cosines and before/after checkpoints remain in the evidence package.

| Qualification | Batch0 | Batch1 |
| --- | ---: | ---: |
| Production total loss |231.641357|262.514832|
| Gated-total versus production gradient relative L2 error |0|0|
| Component-sum versus total gradient relative L2 error |1.346e-6|4.742e-7|
| Component-sum maximum absolute error |5.798e-4|5.188e-4|
| GPU resize versus CPU reference maximum error |2.980e-8|2.980e-8|

Production agreement is exact both globally and in every declared parameter
group. All scalar, component-additivity, finite-value and declared error bounds
pass. All 139 inputs remain pinned; terminal collection rechecks 37 native
artifact hashes. All 357 source/before/after tensors match exactly, including
optimizer moments and frozen actor copies. Counters remain 200,000/12,477. The
host-only guard passes, with no recovery or unfinished child; it does not claim
hardware qualification or continuous GPU telemetry.

The measured two-batch loop takes 2.868s; the complete native test takes 7.31s and
the service wall time is 9.805s. These are diagnostic costs, not training speed.
Minimum sampled device-memory headroom is 16.60GiB. All 141 Rust CPU tests and
strict Clippy pass. Production learner code and packaged learning runtime stay
unchanged; instrumentation is confined to a `cfg(test)` module.

## Raw weighted gradients

These are L2 norms after applying the configured loss coefficients, before AGC
and LaProp. Norms do not add and must not be read as percentages of the final
update. `Other` includes the prediction/reward/continuation/disagreement heads;
the world imitation path uses frozen actor weights.

| Batch | Parameter group | Total | CDP | Dynamics KL | Representation KL | Reward | Continuation | Replay value | Disagreement | Expert |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | encoder | 4431 | 2082 | 344.3 | 95.29 | 3.981 | 0.6157 | 12.72 | 0 | 3271 |
| 0 | posterior | 75.77 | 29.31 | 4.608 | 2.321 | 0.08935 | 0.0191 | 0.3139 | 0 | 69.49 |
| 0 | dynamics | 130.2 | 42.05 | 10.28 | 0.9451 | 0.1108 | 0.02167 | 0.5974 | 0 | 120.3 |
| 0 | other | 141.8 | 141.8 | 0 | 0 | 0.1729 | 0.05383 | 0 | 0.01259 | 0 |
| 1 | encoder | 1.733e+04 | 1654 | 482 | 42.58 | 7.007 | 0.1461 | 93.88 | 0 | 1.665e+04 |
| 1 | posterior | 181.6 | 34.12 | 5.028 | 1.889 | 0.1165 | 0.001418 | 1.476 | 0 | 161.9 |
| 1 | dynamics | 151.5 | 40.38 | 8.008 | 0.8294 | 0.1293 | 0.001103 | 1.403 | 0 | 142.5 |
| 1 | other | 129.1 | 129.1 | 0 | 0 | 0.2121 | 0.00348 | 0 | 0.01984 | 0 |

Expert/CDP norm ratios are 1.57/10.07 in the encoder, 2.37/4.74 in the posterior,
and 2.86/3.53 in recurrent dynamics for batches 0/1. The much larger imitation
norm on batch 1 also shows substantial variation between these two fixed state
sets. Reward and replay-value raw gradients are much smaller here, even when
their direction opposes the total. Disagreement has zero raw derivative in the
three representation groups and a nonzero derivative in its own head.

| Batch | Parameter group | Total–expert cosine | Expert–CDP cosine | Expert–dynamics KL cosine | Expert–representation KL cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | encoder | 0.9468 | 0.5605 | -0.5665 | -0.5405 |
| 0 | posterior | 0.9248 | 0.0248 | 0.0877 | -0.2284 |
| 0 | dynamics | 0.9383 | 0.0210 | 0.0935 | -0.0393 |
| 1 | encoder | 0.9974 | 0.7220 | -0.9220 | -0.4914 |
| 1 | posterior | 0.9870 | 0.5641 | -0.3259 | -0.0784 |
| 1 | dynamics | 0.9596 | 0.0731 | 0.0118 | -0.0359 |

These local directions leave several distinctions unresolved. AGC depends on
parameter scales, and LaProp uses saved optimizer history; raw gradient size
is not parameter movement. The diagnostic uses the final control checkpoint,
fresh fixed sampling draws and a reused unaided final-greedy evaluation corpus,
with corrective teacher labels added offline. It does not measure the original
training minibatches, earlier learning, the centroid/FIRE candidates, independent
seeds, or broader policy-distribution effects. Opposing KL
components are expected to trade off representation fitting and regularization;
their signs alone do not justify reducing them.

Together with [more updates lowering imitation CE without improving play](2026-10-29-cdp-breakout-h5-r32.md)
and [position supervision improving readouts without an accepted gameplay gain](2026-10-29-cdp-centroid-representation.md),
this shifts attention toward how the optimizer transforms those signals and
how learned state/action fitting transfers to autonomous trajectories. Saved
gradients and optimizer moments permit a CPU follow-up before allocating another
learner. No coefficient or architecture replacement is established by this test.

## Preserved failure and corrected execution

The original service starts 23:33:25 and fails 23:33:29 UTC, source `523e67e`. Its test
incorrectly assumes the staged posterior path, which production selects only
for BF16. The actual Auto runtime has a separate posterior session. The
assertion fails after restore and before either batch or backward pass; the
native output directory stays empty. All 119 original pins are checked before
repair. The original service/guard remain failed, and its failure snapshot's
journal-record-limit warning is preserved. No passing original result exists.

Corrected execution explicitly uses Auto's ordinary training graph and hard
posterior inputs. The corrected CPU graph test checks that distinction. It
reuses the exact prepared corpus and retains the old source under a verified
historical hash; no completed batch, training or evaluation stage repeats.
Earlier CPU fixture and indexed-loop Clippy failures also remain retained.

Evidence root: `/x/Temp/kindle-gradient-attribution-20261029`.
Corrected root: `/x/Temp/kindle-gradient-attribution-20261029/auto-corrected`.
The corrected service is `kindle-breakout-gradient-attribution-auto-20261029`,
invocation `fcb53e5eeb6844c1be220dab6033f787`. Terminal collection finishes
23:41:12 UTC. Both invocations are terminal; no GPU allocation is active at
collection. Check live state before later work.

Key evidence:

- `failure-closure.json` and `gradient_contributions-staged-failure.rs` preserve
  the original assertion, source and 119-pin closure.
- `auto-corrected/declaration.json` pins 139 inputs; its historical-resolution
  receipt resolves the changed test source without rewriting old evidence.
- `auto-corrected/native/result.json` SHA256
  `a0f4a9642de05f1fb8cd95b24e598c044ee5b54abf79b52c544ffc13c29cafe1`.
- `auto-corrected/review.json` SHA256
  `a6856609c70fd1d581a878745ca5b2ba47d3fbd74c7ad0997b0f0c7999857852`.
- `auto-corrected/completion.json` records terminal service identity, 139 input
  pins, 37 rechecked native artifacts and 357 invariant tensors.
