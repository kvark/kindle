# Actual-history denoiser fitting check

The goal **prove the world model training on Breakout** remains unproven.
The declared actual-history fitting check fails.
Ordinary matches on four draws are [0, 0, 0, 0]/110; privileged actual-history
one-step matches are [0, 0, 0, 0]/198. Observed sigma.5 denoising ball accuracy is
[0.0, 0.0, 0.0, 0.0], with eight supported balls per draw. This is an eight-clip development
fitting task, not generalization or a complete world-model assessment.

The [protocol](../experiments/2026-11-09-breakout-denoising-real-history.md)
changes training-history delivery only: each update receives four actual past
images through a five-arrival call. Both image/event heads now use real context
every update. The previous mixed-history control used generated histories for
three of every four transitions. Target order and native persistent noise/RNG
sequence remain; optimizer updates and model/loss/sampler are unchanged. This
comparison tests that history change in the scale100 recipe; it is not a test
of every denoiser or budget, and no unique failure cause is established.

| Evidence | Result |
|---|---|
| Fresh training | 2,048 calls/updates;65,536 targets;224 unique arrivals;588 positive source-prefix targets |
| Actual initial parity | 135 tensors and20 per-draw forecast arrays exact |
| First update | All four loss components match;135/135 trained tensors exact before update2 |
| Frozen/restore | 135 tensors and20 per-draw arrays exact;no learning during restore |
| Train call/loop time | 157.905s/166.498s;control call137.973s |
| Data/host | No new emulator actions/frames;target AMD RX7900XT/Auto;separate user's process untouched |

Same RGB64/trainB32/inferB8,seed41001,LR1e-4/LaProp epsilon1e-8/AGC.3,
context-motion scale100/radius4/mean1,45 parameters/1,787,843 scalars/135 tensors,
three Euler samples and v2 format. Native source879ab527,binary9657af72 is
reused with its completed guarded weighted-loss value/derivative qualification;
no rebuild or native fixture repeat. Runner/protocol source6f130f10.
All129 declared and70 CPU analysis pins pass;
four CPU runner checks and2,048 finite six-component learning rows pass.
Mean image loss across first/last64 updates is0.848643→0.124014.
Shared-host activity and extra call overhead limit speed attribution.

Every supported ordinary endpoint has zero ball-color candidate pixels on every draw.
Every supported privileged one-step endpoint also has zero candidates on every draw.
Missing predictions fail. Eighteen unsupported ball endpoints per task/draw
remain excluded explicitly. The unchanged per-action/horizon ball,paddle,brick,
reward andending metrics, every supported/missing endpoint andregression stay
retained. Evaluated suffixes contain no positive reward/ending events, so they
cannot establish event timing or natural calibration. Accurate static bricks
and low average image loss cannot establish motion learning.

Failure under actual-history one-step evaluation puts the error before
accumulated ordinary-rollout drift. Training on real histories alone does not
repair this fixed-budget recipe; generated training history is not necessary
for its observed fitting failure. This does not identify a unique optimization,
representation or sampling cause. No unchanged extension, broad learner,
collector or automatic successor is selected. All completed trials stay closed.

The full goal still requires physical/event precision, natural calibration,
persistent removed-brick memory, late/tunnel/second-wall/above-top coverage,
untouched families andfresh confirmation. The existing full oracle lacks an
above-top endpoint. A tiny fit pass would not discharge these requirements.
Actor/critic, CDP comparison andgame campaign remain parked.

Retained root `/mnt/data/kindle-breakout-denoising-real-history-20261109`:
`review.json`, `pixel-review.json`, `learning.jsonl` inside `train/`, all frozen
checkpoints, guard receipts, target plan, source/runtime identities, and
`individual-endpoints.csv` (1,376 endpoints). `contact-lane0-causal.png` retains
16 lossless fixed lane0/draw0 frames with the same RGB64 camera. No old final
RGB/model query; only required cached actual-initial arrays are decoded.

Closure verifies221 pins in0.531s, two sealed guards/18 files and
one terminal successful service. No new build/cache is created. Observed-frame
brick cells are exact;observed paddle accuracy is1/.875/1/1,while all observed
balls remain missing. These observed-frame metrics do not establish forecasting.
