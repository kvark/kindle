# Lower-noise training: improved one-step visibility, worse rollouts

The goal **prove the world model training on Breakout** remains unproven.
The [bounded lower-noise learner](../experiments/2026-11-09-breakout-denoising-low-noise.md)
gets **[161, 161, 158, 155]/198** correct actual-history one-step positions versus the
control's **[140, 149, 140, 140]/198**, but ordinary matches regress
**[35, 38, 34, 33]/110 → [18, 20, 23, 23]/110**. It fails the declared fitting test: 3/24 individual
gates pass (three first-horizon checks), while all longer-horizon and aggregate
criteria fail. **Retain log-noise mean−1.2, not this−3 recipe.** Four diffusion
draws share one reused development learner/data/seed; no reliability claim.

| Task, all four draws | Control pixel MSE | Lower-noise pixel MSE | False balls control/new | Correct paddle control/new |
|---|---|---|---|---|
|teacher|0.000065882|0.000083306|5/17|767/772 of864|
|ordinary|0.000564893|0.000920496|7/56|324/266 of512|

More visible balls do not establish a good simulator. One-step detections rise
to172/174/170/168 of198; none match the previous supported ball position within
one pixel. Detected/previous-supported median movement projection is.995–.999
(copy0, true step1), median next-position error.333–.388 RGB64 pixels. But later
ordinary frames have10–22 ambiguous red-candidate cases of110 per draw and many
misplaced balls. Conditional localization precision is not total recall.

| New ordinary horizon | Draw0 | Draw1 | Draw2 | Draw3 |
|---|---|---|---|---|
|1|6/8|8/8|8/8|8/8|
|4|1/6|1/6|2/6|1/6|
|8|0/8|0/8|1/8|1/8|
|16|0/8|0/8|0/8|0/8|

A separate CPU event audit exposes another failure in the isolated denoiser.
Across the512 saved ordinary event predictions, every target reward and ending
is zero, yet decoded scalar reward error is large:

| Ordinary event metric | Control | Lower noise |
|---|---|---|
|Decoded reward RMSE|65.6043666|59.9126949|
|Positive-reward probability Brier|4.81856046e-08|2.114511e-07|
|Ending probability Brier|3.5088778e-12|3.3044289e-12|

Both have0/512 positive-reward and ending alarms at probability.5. These zero-only
scores are not natural-frequency calibration or evidence of positive-event
recognition. Ending risk is1−continuation/(1−1/333). The event-head's57 final
weights/moments remain bit-exact; different generated images change its inputs.
The scalar RMSE difference is not an event-head learning improvement.

The [native scalar decoder](../../kindle/src/dreamer/distributions.rs) averages
probabilities over exponentially wide scalar support. Small distribution tails
are a plausible contributor to the large scalar error, but this run does not
retain logits needed to attribute it. This is a failed prediction, not proof of
a decoder implementation bug or an assessment of the retained CDP agent.

Event coverage must be scoped explicitly. Source rewards are1 at arrival4/lane4
and arrival5/lane1; there are no terminals. Ordinary exported suffixes16..31
are zero-only. Teacher RGB queries5..31 include the latter positive target but
discard reward/ending outputs. The original review's evaluated_positive_rewards=0
field therefore applies only to ordinary event exports. The separately retained
coverage-clarification.json states that scope, preserves the original review,
and changes no score/gate. No native query is repeated to complete missing
positive-event evidence. The initial unpublished report prose incorrectly says
0/24 individual gates pass; the saved review and horizon table already show
3/24. A separate CPU-only report correction fixes that prose, preserving the
original report/source. No score, gate, native query or learner changes. Per-horizon event metrics and source indices remain.

Only unforced training lognormal mean changes−1.2→−3;std1.2/bounds.002..5,
half-max alternation and Gaussian draw count/order remain. Expected fraction
below sigma.05 changes3.363%→25.071%; these are analytic expectations, not measured
exposure counts.32,768 forced sigma5 targets remain. This changes optimization
weighting across noise levels; aggregate image losses are not matched. It does
not isolate a unique cause of contrast, hallucination or history drift.

Same fresh2,048 five-arrival calls/updates,65,536 targets,224 unique arrivals and
588 positive prefix exposures, exact target/pixel-normal order,seed41001,
RGB64/trainB32/inferB8,Auto,rate.001,LaProp epsilon1e-8/AGC.3,context-motion100,
residual/half-max modes,sigma_data.05/history normalization.5, actual histories
and three Euler evaluations. No new emulator actions/frames, data collection,
object/RAM labels, pretrained encoder, weights, differentiated graph or loss
expression. Learning/inference stay native Meganeura+Blade; Python prepares
and analyzes only.

| Verification | Result |
|---|---|
| Source/native |fecc251b /83146163393626bb5c36855a35d38a3b9af6635d7713058e1cca326f961e3d64 |
| CPU |22 focused Rust,40 packaged config/restore,four runner checks; strict release test-target Clippy |
| Default-mean v5 fixture |135 initial/frozen tensors,20 full initial arrays,135 first-update tensors and six losses exactly match cached control before learner launch |
| New learner initial |135 tensors and20 per-draw full rollout arrays exact before learning |
| First main update before update2 |Six finite reports/correct total,45 changed weights/135 finite tensors; event losses/57 tensors exact |
| Final event-head state |57 weights/moments exact |
| v6 freeze/restore |135 tensors and20 restored arrays exact |
| Loss ledger |2,048 finite rows/65,536 targets |
| Declared/CPU analysis |140/97 pins |
| Learn-call/loop intervals |161.227s/169.159s versus control175.803s/184.417s |

The default fixture's one qualification update is separate from the main budget;
it takes2.522s including creation/query/save. Learn-call timing also includes
input packing and native CPU/GPU/readback; loop timing includes periodic exports,
not initial/final assessment. Shared host; no isolated kernel speed claim.
All ages0/509/1021/1533/2048 and losses remain; no best-age selection.

Optional log_noise_mean defaults−1.2/omitted and retains the original default
sigma function, v1/v2/v3/v4/v5 and unchanged three-step inference. Non-default
requiresv6, persists in restore, rejects invalid ranges/headers before GPU.
The40 packaged checks cover native configuration/restore parsing; no new command
line parser is introduced. Existing weighted-loss native values/derivatives are
reused because the complete graph and EDM coefficients are unchanged. Observed
sigma.5 denoising remains a distinct noisy-current-image task, not RSSM
reconstruction or future prediction. All original physical/event regressions,
1,376 endpoints and16 fixedlane0/draw0 lossless frames remain.

Default fixture/train/restore and both services are terminal successful, with
three sealed guards/27 files. Main invocation17f6f731b6b5421b8000418e66561475;
default-fixture invocationd86d686ae1cc48a28715f76f89a40689. Own
792,536,079-byte apparent build cache is released after packaging, retaining
native/wheel/source/validation executable and all checkpoints/evidence. All
earlier trials/failures/repairs stay closed. No extension, broad learner,
collection or automatic successor selected. Actor/critic, CDP comparison and
five-game work stay parked.

Full physical/event precision, natural calibration, persistent changed-brick
memory, late/tunnel/second-wall/above-top coverage, untouched families and fresh
confirmation remain required. The full oracle lacks above-top endpoints. No
fitting score completes the full goal. Root
`/mnt/data/kindle-breakout-denoising-low-noise-20261109` retains all scripts,
source/native/wheel, config/plan, checkpoints/losses, guards, raw arrays,
comparison-review.json, coverage-clarification.json, individual-endpoints.csv
and contact-lane0-causal.png.

Closure verifies285 pins in0.771s, three sealed guards/27 files
and two terminal successful services. Restored v6 metadata retains log mean−3
at age2048. Own build cache is absent; all useful artifacts remain.
