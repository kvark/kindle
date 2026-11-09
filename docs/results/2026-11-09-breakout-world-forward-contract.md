# Frozen world training and forecasting forward contract

The strict forward contract **fails**. With categorical samples held fixed, recurrent states agree exactly across the measured training and streaming paths, and the identical full graph agrees bitwise between Training and Inference modes. Small observed-image differences and two categorical sampling disagreements remain. This does not establish learned dynamics or discharge the Breakout world-model goal.

[Protocol](../experiments/2026-11-09-breakout-world-forward-contract.md).
Artifacts: `/x/Temp/kindle-world-forward-contract-20261109`; corrected execution and CPU evidence: `/x/Temp/kindle-world-forward-contract-20261109/prelaunch-v2`.

## Fixed measurements

Original plain H16/RGB100 and plain H16/RGB1 final8,192 checkpoints, each on two retained T32/B8 mixed/contact batches and two new diagnostic noise pairs (95001/96001,95003/96003): eight cases. RGB64/12M/Auto, full BPTT,68 logical weights and204 saved checkpoint tensors. These are conditional development controls, not fresh learners or generalization evidence.

Training mode runs the complete actual world graph with its optimizer cleared. Inference mode runs the same graph and feeds. Test instrumentation exports existing nodes without adding an operation or parameter and preserves the original outputs. Streaming uses the actual observe/transition/decoder graphs, recursively advances h and feeds the training categorical state at each step. The observed stream reads all32 actual observations; the prior suffix starts at15 and receives only the16 future commands and its forced samples.

|Comparison|Arrays|Elements|Maximum absolute difference|Failed elements|
|---|---:|---:|---:|---:|
|Identical unrolled graph|736|45,613,136|0|0|
|Streaming recurrent h|384|6,291,456|0|0|
|Streaming observed RGB|8|25,165,824|0.0009346547786|3,579|
|Streaming suffix RGB|128|12,582,912|9.60912816e-08|0|
|Streaming CPU Gumbel samples|136|1,572,864|1|4|

The predeclared continuous tolerance is atol2e-4 +rtol2e-4; categorical arrays require exact equality. All736 full-graph arrays are bitwise equal across modes, including raw RGB logits, states, samples and losses. All384 streaming h arrays (6,291,456 values) are also exact. All128 suffix-image arrays pass, with maximum difference9.61e-8.

All eight observed-image arrays exceed the continuous tolerance at3,579 of25,165,824 values. Maximum RGB difference0.000934655 equals0.23834 of one8-bit intensity step. Observed decoding is time-batched at256 rows in training versus8 rows in streaming; this measurement does not isolate a particular convolution, normalization or compiler operation.

Two of98,304 categorical blocks differ in the streaming CPU Gumbel control: original-H16 mixed draw1 and RGB1 contact draw0, both in observed samples; all suffix samples match. Four unequal one-hot elements represent two different categorical choices. These are matched-uniform Gumbel comparisons, not ordinary inverse-CDF inference RNG parity. Streaming logits were not exported, so this study cannot separate encoder batching, probability arithmetic and near-tied sampling margins. All h comparisons condition on forced training samples; agreement does not prove unconstrained sampled trajectories stay identical.

All24 independent F64 observed/suffix motion-CE and weighted-total checks pass. Maximum relative error is 9.974690176e-08. Same-time RGB targets, reset masks, first-reset-onward suffix masks andfull B×T/B×16 denominators are checked. These are values, not a new derivative or optimizer test.

## Physical consequence of the image differences

A separately declared CPU follow-up applies the unchanged conservative RGB detector to the saved paired images and actual targets. No new model query, tolerance change or re-evaluation of a closed challenge occurs. The original contract remains failed.

|Stage|Images|Target-supported|Ball presence changes|Within1px decisions changed|Training /streaming hits|
|---|---:|---:|---:|---:|---:|
|observed|2048|1636|0|5|1318 /1323|
|suffix|1024|744|0|1|188 /189|

Across3,072 paired images, ball detection availability never changes, but six within1px decisions change (five observed, one suffix), all toward a streaming success. Observed ball-center differences reach0.46813 pixel because the detector uses hard per-pixel thresholds; suffix coordinate differences are at most8.19e-7 pixel, yet one case crosses the exact1px decision boundary. These are instrument discontinuities at fixed predictions, not improved learned physics. Keep every original gate, tolerance and negative result.

The suffix table includes1,024 generated states, of which976 have supervised non-reset-crossing targets; all744 truth-supported suffix states lie in that valid subset. Related states/draws are not independent episodes. Full per-case numbers and exact derived centers/masks remain in `consequence-centers-v1.npz`.

## Execution, failures and limits

Test-only sourcebff576f74ba9f721d2b162dcacb4d735ec7f640e, executable SHA2565d84ef342ad88f9b3c1a9d4fc46c4f44e5cc565e0e503dbeb31e18371e37f2ba. One native stage completes in219.305s: eight full backwards, eight unrolled inference forwards and768 streaming component forwards, zero optimizer steps. All68 full-graph weights andall parameters in the streaming sessions remain exact; both source checkpoint hashes remain unchanged. The executable is an unoptimized diagnostic build; this time is not a learning/backend benchmark.

The focused CPU graph-extraction check and strict core/test Clippy pass. The initial global format check reported existing unrelated differences and the new unformatted file; only the new file is formatted and its focused check passes. No whole-workspace formatting pass is claimed. Own2,244,447,285-byte disposable build cache was released after retaining the executable. Installed learner runtimes are unchanged.

The original prelaunch declaration pins its own stdout log while empty; its final print changes that file. The audit catches this before GPU work. Preserve the original declaration, log and failed audit; a separate30-pin declaration under `prelaunch-v2` launches the single native stage. An immediate post-launch command-line assertion races process startup; the same live PID2169475/invocationb52d663b161d443a9e6d191c691e5943 is reinspected and the observer attached without restarting. Native execution succeeds without retry or repair.

CPU analysisv1 counts states but asserts categorical blocks, stopping after its first case. Its script/declaration/failure remain. Separatev2 changes that accounting to states×32 and preserves all tolerances and metrics; it performs no native rerun. CPUv2 verifies72 analysis pins; the separate physical-consequence stage verifies54. The successful queue and observer are inactive, with one host-only guard/nine sealed files, no unsealed files, real fault, driver recovery or NVML polling.

The experiment finds no deterministic recurrent mismatch in these matched-sample cases. It does not establish harmlessness of the smaller numerical/sampling differences during unconstrained learning or forecasts, and does not diagnose the cause of failed physics. The strict contract remains failed; no runtime arithmetic change, learner, extension, collector or automatic successor is selected.

Keep the H0 broad reference and the closed plain RGB1 and mean-latent results. Accurate free flight/collisions/action responses, natural reward/life/game-over calibration, persistent removed-brick memory, late/tunnel/second-wall/above-top coverage, untouched families and fresh learner confirmation remain required. Actor/critic, CDP comparison and five-game work stay parked. The full goal remains active and unproven.

Closure verifies124 pins,one terminal successful native stage/observer andnine sealed guard files. The measured contract remains failed.
