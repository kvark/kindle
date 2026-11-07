# The existing position target is much more precise than its learned head

The CPU reference of the existing native color-position target has ball x/y
mean absolute error of **0.058/0.225 source pixels** on the fixed 555 shared test
rows. The saved learned head is **5.461/5.206 pixels** away from that target.
The crop difference identified in the saved-head study therefore does not
explain most of its position error on this corpus. No crop or loss change is
selected from this diagnostic.

| Comparison on the same 555 rows | Ball x MAE | Ball y MAE |
| --- | ---: | ---: |
| Existing target formula versus teacher | 0.058 | 0.225 |
| Earlier pixel reference versus teacher | 0.059 | 0.140 |
| Saved learned head versus teacher | 5.460 | 5.275 |
| Saved learned head versus existing target | 5.461 | 5.206 |

All 601 held-out rows are retained for detection accounting. Of 557 rows with
teacher coordinates, the target and earlier pixel reference each detect the
same 555. Both miss lane 7 indices 1032 and 1312, whose teacher coordinates are
(78.5,93.0) and (92.5,93.0). The target produces no extra detections where the
teacher sees no ball; the earlier pixel reference produces four, all on lane 6
at indices 552,728,831,1399 and y184.891, its lower boundary row. These are
disagreements with the teacher, not independently verified physical detections.

On 136 near-paddle descending rows, target x/y MAE is 0.054/0.374 versus
head-to-target 4.171/4.785. Target MAE is 0.067/0.223 on lane 6 and 0.047/0.226
on lane 7. Maximum target errors on the common test rows are 2.250/2.964 pixels;
small average error does not imply exact targets on every frame.

## Scope and interpretation

This uses native2974160's existing target formula and the old encoder6e-6
centroid100 checkpoint settings: red excess with background scale 1.5, rows
27–55 and columns 3–60, normalized pixel centers and the original minimum mass.
The earlier pixel reference uses rows 28–56. Settings are fixed before replay;
no thresholds, crops or heads are fitted or tuned. All 2,064 saved pixel-reference
predictions reproduce exactly, with the original 954/509/601 split unchanged.

The existing head predictions are reused from the completed CPU reconstruction;
no head inference or fitting is repeated. Its coordinates are always present,
so generic metric fields counting predictions without a reference are not a
head detection-quality score. Head errors are reported only where a target is
available. These posterior states come from the old scalar capture on reused
control trajectories, not the learner's replay states or the newer encoder4e-5
model. The old full gameplay comparison remains protocol-failed.

This is a **CPU F64 reference**, not a direct GPU readback of targets or head
outputs on these frames. The documented native RGB64 byte fixture passes;
maximum CPU F32/F64 target drift is 0.00001507 pixels. Teacher coordinates and
the target share color priors and are not independent physical truth.

The result weakens target imprecision as an explanation for this old head's
multi-pixel error. It does not distinguish representation, optimization,
training distribution or state-history effects, nor establish a causal gameplay
limitation. The newer centroid gameplay comparison already failed to give an
overall benefit; this diagnostic does not select another loss or learner.

## Execution and evidence

Selection completes16:59:46 UTC November1/source79701cc in2.625s (2.728s wrapper).
Seven independent scalar cases, mask boundaries, minimum-mass behavior,
coordinate conversion and the RGB64 byte fixture pass before replay.
The first CPU replay completes17:00:05 in11.769s (11.876s wrapper). All20,960
actions/83,799 frames reproduce every recorded input pixel hash, outcome and
boundary. All27 declared inputs and521 active Qbert25M inputs stay unchanged,
as does originalPID821599/invocation208ec1ae7b96453f927694fae11fa0ff.

One low-priority CPU core and one BLAS thread are used. No native import, GPU
work, learner updates, failure or repeated stage occurs. Retain CPU co-host time
without correcting Qbert's wall time. Root:
`/x/Temp/kindle-centroid-target-precision-20261101`. Complete; do not rerun.

| Receipt | SHA256 |
| --- | --- |
| Declaration | `ee0e8de4c4db12d2bb40e27fc191ddcb02e71552376ebe679b05e7affd9c2b54` |
| Review | `569020de6103d532837d1ce936ce91235eba9b284fee7166e0a1a7912477ee91` |
| Predictions | `6a61f13aba4b9ba1f5443bd1ca6ae1c69240dea94db344effb20ff1179a65895` |

See the [fixed protocol](../experiments/2026-11-01-cdp-centroid-target-precision.md)
and [saved-head study](2026-11-01-cdp-centroid-head-readout.md).
