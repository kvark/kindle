# Breakout world model at 96x96: give the ball a pixel

In the 64x64 frames every learner has used, the ball is sub-pixel. Across
20,000 corpus frames its brightest red value averages 77/255 (10th percentile
0, 90th 198) because the antialiased bilinear resize spreads a 2x4-pixel
object over a fraction of one cell, and the resize yields 2,322 distinct
colours. A pixel model therefore predicts a dim blob whose intensity depends on
sub-pixel phase; its soft forecasts fed back as decoder history fade step by
step, and sharpening them at forecast time destroys the forecasts (four-step
development tracking 59/109 → 3/109 at temperature .5, 0 below). Corpus scale
did not change this ([result](../results/2026-11-10-breakout-world-scale.md)).

This experiment changes only the frame side. `ObservationKind::Rgb96` already
exists in the encoder, decoder, preprocessing and config. On 1,026 re-rendered
development frames the ball is detected in 895 (the same 87% as at 64), its
brightest red pixel averages 181/255 (10th percentile 162, 90th 196) and about
four pixels exceed 100/255: a consistent multi-pixel object instead of a
phase-dependent blob. The 1M learner costs 347 ms per update at 96 against
129 ms at 64 (both measured with light concurrent GPU use), so 16,384 updates
take about 1h50m.

## Data

Fitting: a fresh 524,288-action collection with the same teacher-plus-random-
block policy, sticky .25, seed base 251001, stored at 96x96
(`corpus/fit96`). Development: the November 7 `standard-development` corpus
re-rendered from its native frames at 96 (`corpus/dev96`), so the natural,
contact, ending and 128 dense origins are the same episodes and arrivals the
64x64 runs scored. Collection and re-rendering run outside the host guard
because its 3 s kernel-log probe cannot survive the lane-file write burst.

## Learner and scoring

Unchanged early physical-feature recipe (1M/B8/T32/H16/Auto/41001,
`motion_event_rollout`, RGB100, KL1/.1, rates 4e-4, spatial decoder/fusion/
action, LEFT/RIGHT/FIRE table), 16,384 updates, evaluation every 2,048. The
ball, paddle and brick detectors scale their 64x64 calibration windows by 1.5;
"within one pixel" means within 1.5 pixels at 96, the same physical distance.
The generalized detectors reproduce every stored 64x64 metric exactly.

Compare against the 64x64 corpus-scale run and its control on the same origins.
A sustained gain at eight and sixteen steps supports the sub-pixel diagnosis;
parity means the loss of the ball is not a resolution effect. Cost per update is
recorded; this is a development screen on reused seed 41001.
