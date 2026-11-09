# Grouped denoised-history learning: fails the fitting test

The goal **prove the world model training on Breakout** remains unproven.
The [fixed-budget grouped-history test](../experiments/2026-11-09-breakout-denoising-generated-history.md)
scores **[13, 9, 10, 10]/110** ordinary endpoints and **[59, 60, 55, 59]/198** actual-history
one-step positions, versus **[35, 38, 34, 33]/110** and **[140, 149, 140, 140]/198** for
the actual-history-training control. It fails the original fitting test
(0/24 individual gates pass). No best-age
substitution or reliability claim. One reused development seed and dataset.

The [matched-noise diagnostic](2026-11-09-breakout-denoising-paired-history.md)
established that generated input history substantially degrades this control's
forecasts. This trial exposes training to its earlier denoised estimates; that
is a specific attempted remedy, not a test of all ways to learn robust history.
These estimates still contain noisy target information and are not the full
causal diffusion forecasts used at evaluation.

| Final ordinary horizon | Draw0 | Draw1 | Draw2 | Draw3 |
|---|---|---|---|---|
|1|6/8|5/8|5/8|5/8|
|4|1/6|0/6|0/6|0/6|
|8|0/8|0/8|0/8|0/8|
|16|0/8|0/8|0/8|0/8|

| One-step motion diagnostic | Draw0 | Draw1 | Draw2 | Draw3 |
|---|---|---|---|---|
|Detected /198|73|71|65|71|
|Within1px of previous position|6|6|4|6|
|Median movement projection|0.923|0.936|0.909|0.909|
|Median next-position error on detected/previous-supported pairs|0.417|0.417|0.430|0.440|

Motion projection is dot(predicted−previous,target−previous)/squared_norm of the
true displacement (copy0, correct1). Conditional detected-pair precision is not
recall. Missing balls fail. All864 one-step rows,1,376 combined endpoint rows,
pixel-candidate diagnostics, physical/event metrics and fixedlane0/draw0 lossless
16-frame clip remain retained. Unsupported18 endpoints per task/draw stay explicit.

Only call grouping changes:513 calls (first five arrivals,511 eight-arrival
calls,last seven) replace2,048 five-arrival calls. Each native SGD target order
and exposure remains identical, verified against retained locations and
target_arrivals arrays. Native sigma/pixel RNG draws retain their order, one
draw batch per update. Same2,048 updates/65,536 targets,224 unique source arrivals
and588 positive prefix exposures. Four actual prefix frames start each call;
up to three later history frames are earlier denoised estimates. This changes
history, residual anchors, motion importance and event conditioning together;
it does not isolate one causal training mechanism. No target-derived importance,
object/RAM label, pretrained encoder, new data or emulator actions/frames.

Same fresh seed41001,45 parameters/1,787,843 scalars,RGB64/trainB32/inferB8,Auto,
rate.001,LaProp epsilon1e-8/AGC.3,context-motion100,residual/half-max-noise modes,
sigma_data.05/history normalization.5 and three Euler evaluations. Source
3f1115ae; unchanged native7871f08b/SHA
af13a03eada23d60aa483e111f68a63d6e16c208daa76ec80ffdc8d7a61474a1.
No rebuild or backend primitive. Existing18 Rust CPU/402 packaged/4 runner checks,
strict Clippy and native math receipts are reused rather than rerun. Native
Meganeura+Blade perform all learning/inference; Python prepares/analyzes only.

| Verification | Result |
|---|---|
| Own initial weights/full rollout |135 tensors and20 per-draw arrays exact to control before learning |
| First update before update2 |Six finite reports/correct total,45 changed parameters/135 finite tensors; event losses/57 weights and moments exact |
| Final event state |57/57 tensors differ; expected conditioning change, not an event improvement claim |
| Frozen/restore |135 tensors and20 restored per-draw output arrays exact |
| Loss ledger |513 finite rows,2,048 updates/65,536 targets |
| Declared/CPU inputs |43/73 pins |
| Native-call/loop time |155.067s/162.249s versus control175.803s/184.417s |

Shared-host timing includes native CPU/GPU/readback and exports; no isolated
kernel speed claim. Every age0/509/1021/1533/2048 and loss row is retained.
The final budget point governs the verdict. Observed sigma.5 denoising is a
separate noisy-current-image task, not RSSM reconstruction or future prediction.
Evaluated suffixes contain no positive rewards or endings, so natural event
timing/calibration remain unproved. Static wall accuracy does not establish
persistent removed-brick memory.

The original CPU preparation assumes the wrong saved plan schema and raises an
assertion before any native work. Its source/output are retained. Separately
declared prepare_v2.py verifies the actual locations/target_arrivals schema with
the same exact target-order requirement; no learner, query or assessment is
repeated and no failed comparison is relabeled.

The CPU observer subsequently attempts a nonexistent local runner path during
its first guard audit. Its original script, progress log and failed subprocess
output remain preserved. The separately declared terminal_v2.py uses the exact
retained runner already used by the native queue; both actual guard audits run
once and pass. Native stages were already terminal successful. No model work,
evaluation gate or result changes.

Both guarded stages and the service are terminal successful;18 sealed guard
files pass. The original launch invocation is c987c6f854d644f3b3abdd0f461186b0.
No build cache was created. No extension, collector, broad run or automatic
successor is selected. All prior trials/failures/repairs remain closed/retained.
Actor/critic, CDP comparison and five-game work stay parked. Full physical/event
precision, natural calibration, changed-brick memory, late/tunnel/second-wall/
above-top coverage, untouched families and fresh confirmation remain required;
the current full oracle lacks above-top endpoints. No fitting score proves the
full goal.

Root `/mnt/data/kindle-breakout-denoising-generated-history-20261109` retains
scripts/source/config/plan, original failure and repair, all checkpoints/losses,
raw predictions, guards, CPU reviews, individual-endpoints.csv,
contact-lane0-causal.png and motion-review.json.

Closure verifies149 pins in0.586s, two sealed guards/18 files
and one terminal successful service. No disposable build cache was created.
