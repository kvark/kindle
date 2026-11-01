# The saved position head outperforms the earlier fitted probes

A CPU reconstruction of the original trained centroid head gives ball x/y
mean absolute error of **5.460/5.275 source pixels** on the same 555 test rows
where the linear recurrent probe gives 9.267/8.064. The existing head therefore
recovers more precise position than those probes suggested. Substantial error
remains; this does not show that the actor uses that information effectively.

This is the **old encoder6e-6 centroid100 checkpoint**, not the newer encoder4e-5
model or the November 1 B8/T32 learner. Its original full gameplay comparison
remains protocol-failed. The analysis uses saved posterior states and performs
no fitting, game replay, native inference, learning or checkpoint writes.

| Fixed method | Ball x MAE | Ball y MAE | Ball x RMSE | Ball y RMSE |
| --- | ---: | ---: | ---: | ---: |
| Existing centroid head, CPU F64 | 5.460 | 5.275 | 7.673 | 7.040 |
| Earlier linear recurrent probe | 9.267 | 8.064 | 12.545 | 10.684 |
| Earlier combined-feature kernel probe | 10.012 | 8.388 | 13.218 | 11.227 |
| Fixed RGB64 pixel reference | .059 | .140 | .143 | .394 |

On the 136 near-paddle descending rows, head MAE is 4.162/4.997, compared with
7.625/8.039 for the linear probe. Both held-out lanes improve: head x/y MAE is
5.728/5.292 on lane 6 and 5.139/5.254 on lane 7. Across the common rows its
bias is −2.400/.454 pixels and 95th-percentile absolute error is 15.395/13.635.
All 557 teacher-visible test rows give 5.460/5.275, so the two pixel-reference
misses are retained separately rather than silently dropped from every metric.

## Method and limits

The fixed native2974160 architecture is 2560→256 linear with bias, RMSNorm
(epsilon1e-4, scale without offset), SiLU, and a two-output linear layer with
bias. Its five arrays contain 656,386 parameters. Predictions convert to source
pixels as `(prediction + .5) * [160, 210] - .5`, without clipping or calibration.
All 2,064 saved rows retain the original 954/509/601 train/validation/test split;
the previously fitted probe predictions are reused without refitting.

An independent scalar implementation, coordinate conversion and seven rejection
cases pass. Maximum CPU F32/F64 difference is .00002813 source pixels. **No
separate agreement check against the native GPU head was performed.** This is
a CPU reference result, not proof of exact native forward agreement.

The existing loss already uses a red-excess centroid, `max(R−1.5 max(G,B),0)`,
inside a crop. Its exact resized-grid mask differs from the saved pixel decoder:
both use columns 3–60, but the native target includes rows 27–55 and the pixel
reference includes rows 28–56. The latter applies the source-center −.5 offset
when testing its crop. Actual native-target errors and the effect of that
boundary difference were not measured. The pixel result is consequently not
an exact measurement of auxiliary-target precision.

Teacher coordinates use color/crop priors, not independent physical truth.
Saved scalar posterior states differ from the original vector policy beliefs;
the corpus and development model are reused. Better readout accuracy does not
establish missing information, causal gameplay improvement, or a result for
the current architecture. No replacement head, new loss or learner is selected.

## Execution and evidence

Selection takes 3.463s wrapper time at16:21:19 UTC November1. The first analysis
completes16:21:52 in2.640s (2.696s wrapper), with one CPU core/BLAS thread. All29
diagnostic inputs and292 active Breakout inputs remain unchanged before/after,
as does originalPID812861/invocation258e9d9deb0b463a91d4206a198085e4. Retain CPU
co-host time without correcting the learner wall time. No failure or retry occurs.

Root: `/x/Temp/kindle-centroid-head-readout-20261101`. Native sources, declaration,
predictions, review and execution receipts remain there. The checkpoint is
`/x/Temp/kindle-color-centroid-20261029/full/breakout-12m-64-b16-r16-centroid100-expert100-200k-seed1009/final`;
features come from `/x/Temp/kindle-centroid-representation-20261029/candidate`.

| Receipt | SHA256 |
| --- | --- |
| Declaration | `01cbf0c17bf25ffc598887ee5cf599d0d0bf5e61051251d4b27488172f26f319` |
| Review | `4c946ea24a908ecafa8768b3031aae715881b4fa7179f8e4c4156b76b43a625f` |
| Predictions | `e612bf8dbfd6d2d9ca993777c231180bece4b0eba5c695171e859d0706490394` |

See the [fixed protocol](../experiments/2026-11-01-cdp-centroid-head-readout.md)
and [original representation study](2026-10-29-cdp-centroid-representation.md).
Complete; do not repeat this analysis.
