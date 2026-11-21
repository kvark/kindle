# Six-step sampling: better one-step ball counts, worse image fidelity

The goal **prove the world model training on Breakout** remains unproven.
The [frozen sampler intervention](../experiments/2026-11-09-breakout-denoising-six-steps.md)
improves actual-history one-step ball matches from **[140, 149, 140, 140]/198**
to **[156, 165, 163, 161]/198** with identical weights and Gaussian noise.
Ordinary matches change **[35, 38, 34, 33]/110 → [30, 42, 40, 38]/110**:
three draws improve, one regresses. All0/24 strict fitting gates still fail.

**Retain the three-step production sampler.** Six steps increases mean pixel
error about3–4× and creates more false balls. It is not a general world-model
improvement. The result establishes that sampling changes these physical
predictions, without showing that sampling is the only training bottleneck.

| Task, all four draws | Mean pixel MSE3 | Mean pixel MSE6 | False ball detections3/6 | Correct paddle positions3/6 |
|---|---|---|---|---|
|Actual-history one-step|0.000065882|0.000219505|5/9|767/764 of864|
|Ordinary rollout|0.000564893|0.002308476|7/25|324/318 of512|

Every origin contains eight equal-size images, so the MSE means weight images
equally. False detections are cases where the original conservative detector
cannot support a true ball but finds a predicted ball. These repeated clip/draw
rows are descriptive counts, not independent game trials or calibrated rates.
All per-origin paddle/brick/image metrics remain in review.json.

| Six-step ordinary horizon | Draw0 | Draw1 | Draw2 | Draw3 |
|---|---|---|---|---|
|1|7/8|6/8|6/8|6/8|
|4|3/6|4/6|3/6|3/6|
|8|2/8|4/8|3/8|5/8|
|16|0/8|0/8|0/8|1/8|

| One-step paired outcomes | Both correct | Three only | Six only | Both wrong |
|---|---|---|---|---|
|Draw0|135|5|21|37|
|Draw1|148|1|17|32|
|Draw2|137|3|26|32|
|Draw3|138|2|23|35|

One-step supported detections increase143/151/140/142→166/168/169/167 of198;
some new detections are misplaced. Ordinary detections rise59/64/57/48→70/80/65/74
of110, while position accuracy remains low. More visible red pixels do not by
themselves establish correct physics. No detector threshold changes or
low-contrast positions manufactured from target coordinates. Missing balls fail;
18 unsupported endpoints per task/draw remain unscored.

The intervention uses the existing rho7 grid with six positive sigma nodes
[5.0, 1.817629098892212, 0.5566728711128235, 0.13390222191810608, 0.022333577275276184, 0.0020000000949949026], then0, versus5/.28308183/.002/0. Same residual anchor,
EDM coefficients, quantized denoising output, Euler equation and seed stream.
Extra denoiser evaluations consume no new random draws. Six evaluations per
future frame replace three; no inference wall-time ratio is inferred. The
9.616s native duration includes restore, both modes, production
parity, exports and frozen save on the shared host. CPU analysis takes
1.257s. No learning, build of a wheel, new emulator actions/frames,
collection, labels or pretrained encoder. Inference stays native Meganeura+Blade.

Source1e66499e adds only a test collector and CPU checks; production sources are
byte-exact to the retained7871f08b runtime. The trained sigma_data.05 checkpoint
stays at age2048. Before any six-step query, the collector's default mode must
match all108 cached teacher batches/864 images and64 ordinary batches/512 images
bit-for-bit. Four complete production predict calls then match20 saved per-draw
RGB/observed/event arrays exactly, covering the full16-step rollout. All these
checks pass, and135 frozen checkpoint tensors remain exact. Expected outputs
and target frames never enter generation; only observed prefix RGB/actions and
submitted future commands do. Only this selected current checkpoint is queried.

All51 declared/29 CPU pins pass. The single service is terminal successful,
with one sealed nine-file host guard. Invocation4b59a1aa3d6641f49e9928f1dff17902.
19 focused CPU checks pass, four GPU tests stay ignored until explicitly selected;
strict release test-target Clippy passes. No new differentiated graph/backend
primitive, so existing math qualification is reused rather than expanded.

The initial CPU schedule assertion incorrectly expects a node within a factor
of two of.05. Preserve the failing source, executable and log. A separate CPU
repair checks independently calculated F64 grid values and narrower bracketing;
the grid, generation, physical gates and strict native bit-parity remain
unchanged. Test-target Clippy separately flags two pre-existing default-config
initializations; equivalent struct literals fix style without changing test
values/assertions, and the final collector is rebuilt and CPU-checked. Preserve
that original log/source too. An initial shell redirection fails before the
build directory exists; no compiler starts until the directory is created.
None of these failures is relabeled or followed by a repeated native assessment.

The own719,946,264-byte apparent build cache is removed after packaging;
all source/executables, original failures, raw arrays, frozen weights and evidence
remain. Retain2,752 endpoint rows and16 exact lossless animation frames at fixed
lane0/draw0, with a fixed horizon8 poster. All previous trials/failures/repairs
stay closed. No learner, extension, collection or automatic successor selected.

This is one reused development checkpoint and eight related clips across four
diffusion draws, not four independent learners or held-out reliability. The
intervention collects no event-head outputs, so it establishes no event timing
or natural calibration improvement. No new six-step own-initial learning-gain
claim. The earlier three-step model's own-initial controls remain retained.
Full physical/event precision, natural calibration, persistent changed-brick
memory, late/tunnel/second-wall/above-top coverage, untouched families and fresh
confirmation remain required. The full oracle still lacks above-top endpoints.
Actor/critic, CDP comparison and five-game work stay parked; no fitting score
completes the full goal.

Root `/mnt/data/kindle-breakout-denoising-six-steps-20261109` retains
protocol/config/checkpoint references, source archive, standalone collector,
raw predictions, guards, reviews, individual-endpoints.csv and
sampler-lane0-causal.png. No production sampler/API/checkpoint format changes.

Closure verifies103 pins in0.696s, one sealed guard/nine files
and one terminal successful service. Own build cache is absent; retained source,
executables, raw outputs and checkpoint evidence remain.
