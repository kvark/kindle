# Fused learner: fewer host round-trips, modest speed gain

**Result:** one fixed synthetic comparison improves **212.70 → 184.43 ms/update
(1.153×)** over the shared-parameter baseline. Native-input N6 Pong improves
**17.36 → 20.16 steady actions/s at R256 (1.161×)**. The **3× target is not met**.
[Self-contained configs, metrics and curves](2026-09-27-fused-learner.json).

## Change and numerical evidence

Posterior T64 and imagination H15 now execute as complete GPU graphs. CPU-owned
RNG streams supply uniform noise; GPU Gumbel-max samples the unimixed categorical
distributions. Reward/value heads batch across imagined time. Shared parameters
and independent slow-critic EMA remain. CPU λ-returns, normalization and two-hot
targets are unchanged. No backend, F32 precision, optimizer, loss, BPTT, replay
ratio or model-size change. Sampling consumes more uniforms, so old trajectories
and checkpoints are **not bitwise-equivalent** to the new run.

Fusing alone barely helped: **212.70 → 211.00 ms**, about 0.8% less time. The
profile then exposed repeated per-block slicing/concatenation. Grouping the RSSM
inputs and GRU lanes removes that work without changing parameters or equations.

| Mean synthetic stage, ms | Shared control | Fused only | Fused + grouped layout |
| --- | ---: | ---: | ---: |
| Complete update | 212.70 | 211.00 | 184.43 |
| Posterior | 33.25 | 30.76 | 20.51 |
| Imagination/targets | 83.97 | 86.27 | 84.85 |
| World training | 69.63 | 70.03 | 55.24 |
| Behavior training | 21.67 | 21.76 | 21.69 |

Each arm uses 36 updates, first four excluded from timing: Dreamer12M,
B16/T64/full BPTT/M16/H15/R256, prediction-only, same synthetic replay/seed.
All **241 state tensors / 146 moments** are finite with matching schemas.
The first shared-parameter result's 227 ms baseline belongs to a different
pair; do not add percentage gains across unrelated windows.

- 16,384 categorical rows match an independent CPU Gumbel reference; frequencies
  match analytical unimix within six binomial standard deviations. Endpoints
  and exact ties are covered.
- Tiny fused posterior/imagination matches independent sequential graphs with
  identical draws, row resets and nonzero heads (`2e-4 × (1 + |reference|)`).
- Grouped RSSM layout and **all input gradients** match an independent scalar
  formula, including odd block counts and the production B16/eight-block shape.
  Worst relative L2 is **1.60e−7**, below the declared 2e−5 limit.
- Existing learning/checkpoint and independent-stream/replay checks pass.
  Current local checks: **94 Rust CPU tests, 854 Python tests**, formatting and
  release Clippy. The obsolete historical Large initialization fixture is
  removed; its archived results remain unchanged.

## Short Pong comparison

Each arm completes **6,144 actions / 1,186 updates**, seed 7301, six independent
streams, native 210×160 RGB, 18 actions, non-sticky published protocol. Exclude
the first 1,536-action interval, which contains only 34 warmup updates. The
remaining 4,608 actions take **265.50 s → 228.60 s**. Mean learner time is
**227.37 → 195.66 ms**. Aggregate simulated speed is **1.157× → 1.343× realtime**;
each of six streams runs at roughly **0.193× → 0.224×**. This is not six
independently realtime agents. Construction timing is reported but not used as
a speed claim; order/cache effects are not controlled.

Mean world loss is **865.4267 → 865.4237**, raw KL **2.19550 → 2.19171**, behavior
loss **5.10201 → 5.10880**, entropy **2.88865 → 2.88293**. Reward MAE differs 3.15%,
imagined reward mean 4.45%, return mean 7.48%; sampling changes accumulate.
Action-histogram total variation is **0.0363**. Both arms complete six natural
episodes at mean **−20.5**, no cutoffs; all unfinished tails remain recorded.
This is a short numerical/learning-statistics check, **not Pong competence or
multi-seed equivalence**. Minimum sampled Vulkan estimated headroom stays above
11.5 GB. The historical ~15.6 actions/s is a context reference, not a matched
baseline; upstream ~59 actions/s remains unreconciled.

Both Pong arms use Tiny `7fe9b252`, pretrained on 250,000 random-play RGB64
frames from Boxing, Pong, Freeway, Breakout and Qbert (45k train + 5k validation
per game). This is additional same-title experience. No action/reward aid is
used here. The synthetic test uses no frontend or pretraining.

## Profile, limits and next decision

The initial fused posterior has **10,435 dispatches, 6,779 data movement**;
imagination has 4,659/3,059 and world gradients 35,221/16,171. This motivated
the grouped layout, not a backend fork. Per-dispatch instrumentation inflates
wall time by **4–39×**: its kernel shares are not ordinary runtime shares or SM
utilization. World training uses four submissions, so its ordinary GPU timestamp
covers only the last chunk, not the full update. **SM utilization is unmeasured**.

A separate post-change profile confirms dispatch counts fall to **3,459
posterior / 3,024 imagination / 14,863 world gradient**. Optimizer-free ordinary
session wall medians are 17.08 / 42.02 / 53.31 ms, with behavior gradients
19.43 ms. These diagnostic sessions are not the end-to-end update benchmark.
The remaining gap is not explained solely by recurrent readbacks.

Imagination/CPU target construction is still expensive. F32 remains: the current
backend exposes F16 cooperative relaxation, **not a BF16 compute switch**.
BF16 weight loading is not BF16 training. Reduced precision needs a separate
numerical/learning comparison; it is not silently enabled to claim this target.
The fast screening recipe is the next iteration-speed deliverable, rather than
another unchanged long Pong campaign.

All completed guarded jobs exit cleanly and reap their children on RTX 5080 /
580.178.04, without separate NVML polling or recovery. Meganeura upstream
`ee3aea42` / Blade `fbb4f28c` were rechecked; dependency-only/external-capture
carries remain `367e53d4` / `7cca6377`.

The first sampler test failed because a 2-D input was passed to a flat max-pool
reduction, leaving rows uncomputed. Explicit flattening plus dispatch-coverage
tests fixes it; the failed binary/logs remain. Its guard also captured a
pre-spawn-source NVRM allocation warning, without Xid/device loss/reset. Origin
is unknown; source and journal receipt clocks must not be conflated. This is
not a GPU safety proof. Full correction history, declarations, profiles and
read-only extraction: [local run](../../runs/fused-learner-20260927.NSTOLm/).
