# A multi-pixel ball at 96x96 does not stop the forecast fading

[Protocol](../experiments/2026-11-10-breakout-world-rgb96.md). Root
`/x/Temp/kindle-world-scale-20261110`; output `train-96`, guard directory
`guards-96`, unit `kindle-world-scale-20261110.service` (invocation
`fc82ed7a573e42da982636ea212486d2`, PID 2335623). Source `aabeb505` on
Meganeura main `a0eff600`.

## Data

Fitting corpus `corpus/fit96`: 525,019 arrivals from 524,288 aggregate actions
(2,096,090 emulator frames, 723 game-overs) collected in 256 s by the same
eight lanes, seeds and policy mix as the 64x64 corpus; emulation is
deterministic for a seed, so the two corpora hold the same trajectories at
different frame sides. Development `corpus/dev96`: the November 7
`standard-development` lanes re-rendered from their native frames at 96, so
the natural (32), contact (8), ending (10) and 128 dense origins are the same
arrivals every 64x64 learner scored.

In the re-rendered development frames the ball is detected in 895 of 1,026
sampled frames (87%, as at 64), its brightest red pixel averages 181/255
(10th percentile 162, 90th 196) and about four pixels exceed 100/255; at 64x64
the same statistic averaged 77/255 with a 10th percentile of 0.

## Learning

Unchanged early physical-feature recipe (1M/B8/T32/H16/Auto/41001,
`motion_event_rollout`, RGB100, KL1/.1, rates 4e-4, spatial decoder/fusion/
action, LEFT/RIGHT/FIRE table) with `observation_kind: rgb96`; 16,384 updates.
Ball within one 64-scale pixel (1.5 pixels at 96) over visible targets, first
latent draw. The 64x64 rows are the corpus-scale run on the same trajectories
and the closed early-feature control.

| Learner, cohort, updates | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 96x96, dense development, 2048 | 75/113 | 69/112 | 10/112 | 3/112 | 0/115 | 0/102 |
| 96x96, dense development, 4096 | 97/113 | 97/112 | 26/112 | 9/112 | 0/115 | 0/102 |
| 96x96, dense development, 6144 | 101/113 | 97/112 | 21/112 | 11/112 | 1/115 | 0/102 |
| 96x96, dense development, 8192 | 101/113 | 100/112 | 30/112 | 10/112 | 1/115 | 1/102 |
| 96x96, dense development, 10240 | 102/113 | 103/112 | 47/112 | 30/112 | 18/115 | 9/102 |
| 96x96, dense development, 12288 | 101/113 | 107/112 | 39/112 | 29/112 | 19/115 | 3/102 |
| 96x96, dense development, 14336 | 102/113 | 103/112 | 44/112 | 17/112 | 9/115 | 6/102 |
| 96x96, dense development, 16384 | 103/113 | 105/112 | 54/112 | 36/112 | 19/115 | 4/102 |
| **64x64 corpus-scale, dense development, 16,384** | 101/112 | 104/109 | 59/109 | 38/107 | 18/114 | 0/95 |
| 64x64 control, dense development, 16,384 | 106/112 | 101/109 | 66/109 | 53/107 | 25/114 | 11/95 |
| 96x96, dense fitting, 16,384 | 99/114 | 107/123 | 47/125 | 24/118 | 13/109 | 2/116 |
| 64x64 corpus-scale, dense fitting, 16,384 | 98/111 | 95/118 | 47/117 | 28/116 | 6/105 | 2/116 |
| 96x96, natural development (32 origins), 16,384 | 27/28 | 25/27 | 10/26 | 7/27 | 8/27 | 1/25 |
| 64x64 corpus-scale, natural development, 16,384 | 25/26 | 26/27 | 15/26 | 8/27 | 3/27 | 1/25 |
| 64x64 control, natural development, 16,384 | 25/26 | 23/27 | 17/26 | 11/27 | 5/27 | 0/25 |

## Reading

With the ball rendered as a bright multi-pixel object, the same recipe and
budget reach the same place as at 64x64 on the shared development origins:
observed 103/113 against 101/112, one-step 105/112 against 104/109, four-step
54/112 against 59/109, eight-step 36/112 against 38/107, sixteen-step 19/115
against 18/114, thirty-two-step 4/102 against 0/95. The natural group tells the
same story (16-step 8/27 against 3/27, 4-step 10/26 against 15/26). Both runs
stay below the control's eight- and sixteen-step 53/107 and 25/114, and neither
memorizes: fitting equals development at every horizon. The 96 curve is also
less stable: eight-step tracking reaches 30/112 at 10,240, drops to 17/112 at
14,336 and ends at 36/112.

The sub-pixel representation therefore was not the limiting factor; it removed
one plausible cause (phase-dependent ball intensity) without changing the
outcome. What remains is the mechanism the state-history diagnostics pointed
at: the decoder carries the ball through its own fed-back soft frames, and a
hedged prediction re-enters as a dimmer, wider blob. The next experiment keeps
everything from this run and only commits each fed-back frame to the nearest of
32 corpus colours with a straight-through gradient
([protocol](../experiments/2026-11-10-breakout-world-palette-feedback.md)).

Native calls took 4,170.1 s for 16,384 updates (254.5 ms each, 1.97x the 64x64
run's 129.4 ms) and the loop 4,542.8 s. Evaluation files for every point are
under `train-96/evaluations/*/*.json`; the forecast-time sharpness sweep on the
64x64 checkpoint is under `sharpness-16384/`. This is a development screen on
reused seed 41001; nothing here qualifies the world model.
