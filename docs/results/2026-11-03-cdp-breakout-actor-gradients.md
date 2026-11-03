# Imitation dominates the current actor's raw weighted gradients

The [backward-only diagnostic](../experiments/2026-11-03-cdp-breakout-actor-gradients-backward.md)
finds imitation gradient norms **6,721× /10,925×** the imagined score-function
norm on the two fixed batches. The actor's total raw gradient is almost exactly
aligned with imitation (cosines .999999989/.999999996). This measures the current
actor at retained policy1/expert100, unlike the older world-gradient diagnostic.
It does not measure the optimizer update or explain the policy100 gameplay
regression. No coefficient or new learner is selected.

| Raw weighted derivative norm | Batch0 | Batch1 |
| --- | ---: | ---: |
| Actor total |305.301907|251.756398|
| Imagined score-function |.0454243|.0230442|
| Entropy |.000297644|.000292618|
| Expert imitation |305.301189|251.756973|
| Critic total |10.026791|9.798460|
| Imagined value |3.566229|3.671317|
| Replay value |7.046689|7.067445|

These norms include policy1,expert100 and critic1/1 coefficients and are measured
before clipping and LaProp. Norms are not additive percentages. Score/imitation
cosines are .015193/−.028276: the small imagined component is nearly orthogonal
to imitation on these batches. Entropy norm is .655%/1.270% of score-function
norm. Imagined/replay-value cosines are .759303/.627870. Actor and critic are
separate heads: both value terms have exactly zero actor derivative, and
score/entropy/imitation have exactly zero critic derivative.

This closes a measurement gap, not the causal diagnosis. The batches use reused
sampled-policy histories, fresh scalar posterior and imagination draws, and
corrective labels added offline. They are not original replay batches or earlier
learning checkpoints. AGC, per-element LaProp preconditioning and inherited
momentum can change both sizes and directions. Saved gradients and moments
permit a CPU optimizer follow-up before another coefficient is considered.
The retained policy1 gameplay scores70.750/143.667 and0/24 two-wall stay unchanged.

## Qualification and completion

The native loop takes **.601s**; the entire native test takes3.55s and the service
5.541s. These are diagnostic costs, not learner throughput. The isolated source
tree matches native447fe50 except its cfg(test) registration and diagnostic
module; packaged learning/inference runtime and teacher remain unchanged.
Production Auto posterior/target methods supply3840 imagined rows and240 replay
rows. Each batch uses identical inputs, hard draws and parameters across terms.
The optimizer and clipping are disabled; no learner entry point is called.

Production/gated-total gradients match **exactly**, globally and separately for
actor/critic. Native component-sum relative L2 errors are2.440e-9/2.571e-9.
Independent F64 sums give actor relative errors2.506e-8/2.532e-8 and critic
2.796e-8/2.757e-8; both use the fixed cancellation-aware bounds. All per-parameter
and group F64 dot products reproduce the native matrices. Scalar accounting,
finite inputs/derivatives and cross-head zeros pass. All512 GPU-resized frames
match CPU references with maximum error2.980e-8. Both batches contain240
unit-weight expert labels; behavior targets sum240 without world normalization.

All **158 pins**,22 behavior parameter arrays per term,357 identical source/
before/after tensors (including optimizer moments),unchanged200k/12,477 counters
and host guards pass. Minimum sampled memory headroom is17.128GiB. Target
normalizers and diagnostic RNG may advance only in the discarded runtime;
no unchanged transient-state or continuous GPU telemetry claim is made.

Original mainPID1008732/invocation`5abfe2a661004b8eaa2e3aa02fb96b97` succeeds
**10:24:44.106 UTC November3**, source`7dbbc29bd029f87c8c56a7f4e0826c0932805023`.
CPU audit completes10:24:43.829 in.527s. Independent terminal collection completes
once **10:26:06.728**,.420s CPU, checking the original invocation and all saved
artifacts without repeating gradients or analysis. Both original GPU services
(context capture and backward diagnostic) succeed; no GPU stage repeats.

All132 ordinary Rust CPU tests,strict Clippy,nine finite/error fixtures and the
75-file source-tree comparison pass before launch. The capture's mode assertion,
initial build's missing fixtures/resize signature, prelaunch replay-row correction
and exclusive-output-path failure remain preserved with original sources and
receipts. No original GPU stage fails. The earlier incomplete preflight remains
historical; only the final qualified executable is used for backward execution.

Both stages and terminal readers under
`/x/Temp/kindle-breakout-actor-gradients-20261103` are closed. Do not repeat them.
No Kindle GPU allocation is active at closure; check live state before work.
The five-game goal remains unfinished. This result selects no learner,coefficient
sweep or automatic successor GPU allocation.

| Artifact | SHA256 |
| --- | --- |
| Capture terminal summary | `9493ecf904e3414ef79f40b3928038665644607c44b3c8e34d5a41cb1fb236e6` |
| Native gradient result | `fa7cc085f07548eca3668a5f4b4033ab165b728ee9f9d0dee6539012f1134802` |
| Independent gradient audit | `96d8518918c81c9f369b446f95aa08888a3f0997e3bc713c0e1709bd036f8fc4` |
| Backward terminal summary | `f5cb3cc7624b65efeb38d89b4cc84cf5116f311b6e2e4ea141e7efe31a28529b` |

## Context capture and preparation history

The [fixed capture](../experiments/2026-11-03-cdp-breakout-actor-gradients.md)
completes **35,552 actions /142,169 frames in34.280s**, with zero updates and
all357 checkpoint tensors unchanged. It retains32 preceding current-model states
and512 raw RGB frames for two B16/T16 batches. All32 repeated state readbacks
preserve probabilities/counters, and all recorded rewards,boundaries,resets and
frame counts reproduce. The capture alone establishes inputs; the later gradient result is reported above.

Inputs come from the retained policy1/replay-value1 model's sampled evaluation.
Each lane supplies four windows chosen by fixed20/40/60/80% eligible-index ranks,
without looking at reward,labels or gradients. Native447fe50 and original teacher
remain unchanged. Corrective labels are added offline with executed-action
feedback; scalar posterior draws differ from the original vector draws. Original
live pixel hashes are unavailable. CPU resize arrays are validation references;
the native backward stage must resize raw frames once on GPU.

CPU preparation passes eight selection cases,32 input pins and338 retained
learner pins in3.338s at10:13:23.393 UTC November3. The original source-mode
assertion failure and pre-execution memory-helper signature correction are
preserved. Neither constructs a native agent. The original capture service
launches10:14:12.632,sourceef390a4c,PID1004049/invocation
`7499ea127a3d4038ace5ba7c041b6042`, and writes its passing review10:14:48.099.
The service is terminal success. Host guards pass; no capture stage repeats.

Independent terminal collection passes **10:17:29.526** in.465s CPU, checking
all36 declaration pins,all512 raw-frame hashes and exact CPU resize references,
32 finite contexts with categorical one-hot state,source action/reward alignment,
original invocation and terminal service state. No emulator or GPU work occurs
in that reader. Capture manifestSHA246b519b/reviewSHAce8d4b66. The capture and
its reader are closed; do not repeat them.

The [backward-only stage](../experiments/2026-11-03-cdp-breakout-actor-gradients-backward.md)
was CPU-qualified against an isolated exact447fe50 source archive. Production
files differ only by a cfg(test) module registration. The first build fails on
four omitted include fixtures and the newer four-argument resize signature;
the exact historical fixtures and three-argument call fix preparation only.
The corrected build,132 CPU tests and strict Clippy pass. Subsequent source
review catches a diagnostic shape error: behavior uses240 replay rows,while
imagination uses3840 rows. Its labels sum240; world-graph target normalization
does not apply. The corrected dimensions are now covered by the graph fixture
and pass the final build,132 CPU tests,strict Clippy and nine finite/error-bound
fixtures. Exact source verification checks75 files; final preflight pins155
inputs in.229s. An exclusive-output-path failure preserves the earlier queue
under its own name before final selection; no native work occurs in that failure.
All earlier sources,receipts and the initially passing but incomplete preflight
remain preserved. The separately selected180s guarded backward job subsequently completes as
reported above; no preparation or capture stage is repeated.

Root `/x/Temp/kindle-breakout-actor-gradients-20261103`. No GPU allocation is
active at capture closure; check live state. No learner,coefficient change or
automatic successor learner is selected. Breakout and the five-game goal remain unfinished.
