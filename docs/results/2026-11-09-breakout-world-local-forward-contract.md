# Local world training and streaming comparison

The strict contract **fails** on small observed-image differences and four
categorical choices. With training categorical samples held fixed, every
recurrent state and every actual generated suffix pixel agrees exactly between
the measured training and streaming paths. All24 independent loss checks pass.
The saved image differences change no conservative ball-detection or within1px
decision. This narrows the implementation question; it does not prove learned
physics, harmlessness of unconstrained sampling differences or the full goal.

[Protocol](../experiments/2026-11-09-breakout-world-local-forward-contract.md).
Artifacts: `/x/Temp/kindle-world-local-forward-contract-20261109`.

## Fixed scope and values

Retained early physical-featurev25 and late-commandv26 final16,384-update worlds,
both1M/RGB64/B8/T32/H16/Auto,80 weights/603,915 scalars/240 saved arrays. Each is
measured on the existing sealed mixed/contact batches and fixed noise pairs
95001/96001 and95003/96003: eight cases. These are reused diagnostic controls,
not untouched confirmation. The prior plain12M cases are not rerun.

Test instrumentation exposes existing nodes of each actual MotionEventRollout
training graph without adding an operation/parameter or changing original
outputs. Training mode clears its optimizer; the same full graph also runs in
Inference mode. Streaming uses the production observe/transition/decoder graphs,
physical command features, independent reset masks and forced training samples.
Observed decoding sees causal actual history. The suffix starts after arrival15
and uses only submitted commands, its forced samples and its own generated
two-image history; no true future image or outcome corrects it.

|Comparison|Arrays|Values|Maximum absolute difference|Failed values|
|---|---:|---:|---:|---:|
|Same full graph, Training versus Inference|880|52,297,824|0.00000190735|0|
|Streaming recurrent h|384|1,572,864|0|0|
|Streaming observed RGB versus F64 sigmoid of training logits|8|25,165,824|0.000646627|295|
|Streaming suffix RGB versus actual native training means|128|12,582,912|0|0|
|Streaming suffix RGB versus F64 sigmoid of training logits|128|12,582,912|0.000000095017|0|
|CPU matched-uniform Gumbel samples versus training samples|136|393,216|1|8|

Continuous tolerance remains atol2e-4+rtol2e-4; categorical equality is exact.
The full graph passes every comparison.879/880 arrays are bitwise equal; the
only differing value is early/contact/draw1 representation loss, differing by
1.9073486e-6. Do not describe the entire full-graph comparison as bitwise exact.

All streaming h and actual native suffix means are bitwise equal. Observed
decoding is time-batched at256 rows in training versus8 in streaming. All eight
observed RGB arrays fail, at172 early/123 late values. The largest difference
equals0.16489 of one8-bit intensity step. This comparison does not isolate a
convolution, normalization or compiler operation as its cause.

Eight unequal one-hot values represent **four of98,304 categorical blocks**:
early mixed/draw1, early contact/draw1, late mixed/draw0 and late mixed/draw1,
all in observed sampling. All suffix samples match. Every categorical vector
remains exactly one-hot. This is a matched-uniform CPU Gumbel control, not
ordinary inverse-CDF inference RNG parity. Streaming logits are not exported,
so probability arithmetic, batching and near-tied choices are not separated.
Forced samples make the recurrent/image result conditional; unconstrained
trajectories may diverge after a different categorical choice.

All24 independent F64 observed motion-BCE, suffix motion-BCE and weighted-total
checks pass at the unchanged loss tolerance. Maximum relative error is
2.15911e-7. The total includes reconstruction100, KL terms and both observed
and imagined reward/continuation losses. Reset-crossing suffix states are masked
from loss using the complete B×16 denominator; each mixed case has116/128 valid
targets and each contact case128/128.

## Saved-image consequence

After terminal analysis, a separately selected CPU-only check reuses the
unchanged conservative detector and all eight cases. Observed pairs compare the
F64 sigmoid reference with streaming RGB; suffix pairs compare actual native
training means with streaming RGB. No model/emulator query or original gate
is repeated or changed.

|Stage|Images|Truth-supported|Training /streaming within1px|Detection changes|Within1px decision changes|
|---|---:|---:|---:|---:|---:|
|Observed|2,048|1,636|1,407 /1,407|0|0|
|Generated suffix|1,024|744|327 /327|0|0|

All3,072 image pairs keep the same detection availability and within1px decision.
Maximum observed coordinate difference is0.00329039 pixel; suffix coordinates
are identical.976 suffix images have reset-valid supervised targets; all744
truth-supported suffix frames fall in that subset. Missing/diffuse detections
remain failures. Related frames and draws are not independent episodes.

The unchanged detector does not find a physical consequence of the checked RGB
differences on this data. It cannot establish harmlessness of categorical
differences, other visual attributes, unseen histories or future training.
These counts are conditional path comparisons, not a new free-running physics
assessment. The strict contract remains failed.

## Execution and retained limits

Test-only source5bbeaccfc3b588b8f1211b42a8459f83b7ecc97e; stripped diagnostic
executable SHA256e6995c54e5bfae2c9588d944992739a3a0da75692098e3ee8585daa8cad6e9ca.
All158 existing source files match qualified81a5e4e3 except cfg(test) module
registration. Installed production code, runtime and arithmetic are unchanged.
Both focused CPU checks, strict core/test Clippy and the dynamic categorical
analysis fixture pass. No broader backend qualification is claimed.

One host-only guarded native stage completes in268.266793s: eight full backwards,
eight full inference forwards and768 streaming component forwards, **zero
optimizer steps**. All80 training weights and every streaming-session parameter
remain exact after each case; both source checkpoint hashes remain unchanged.
This unoptimized diagnostic executable's time is not a learning benchmark.
The packaged2.14GB compiler cache is released with source/executable/checks kept.

All107 execution pins,70 analysis pins and73 original closure pins verify after
the observer exits. The guard passes with nine sealed files and none unsealed.
Native and observer services are inactive/MainPID0/Resultsuccess; their recorded
invocations and terminal completion journal agree. There is no native failure,
retry, CPU repair, numerical nonfinite value or driver recovery. The CPU strict
analysis takes1.454s; the separate56-input consequence check takes1.495s.
Raw arrays, logits, metrics, categorical failures, derived centers and all
declarations/closures remain outside Git.

This measurement finds no forced-sample recurrent or generated-history pixel
mismatch on these local paths. It does not establish a cause for the poor free
rollouts. Keep both action options default-off and the completed late learner's
failed dense screen. No new learner, collector, physical assessment, extension
or automatic native successor is selected. Accurate action-conditioned physics,
natural event calibration, changed-brick memory, late/above-top coverage,
untouched families and fresh confirmation remain required. The world-model
goal remains active and unproven; actor/critic and five-game work stay parked.
