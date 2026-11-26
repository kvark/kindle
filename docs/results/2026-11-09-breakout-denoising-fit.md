# Breakout denoiser: basic fitting remains unresolved

The unchanged conditional denoiser fails its fixed eight-trajectory fitting
check. Every final draw gets **0/110** supported ordinary16-step endpoints and
**0/198** supported one-step endpoints with actual past images. Observed-frame
denoising at sigma.5 gets0/8 ball detections on every draw. The model reconstructs
the brick cells correctly there but does not recover the small moving object.
This failure is present before long generated-history drift could explain it.

This is a negative early fitting result, not proof that all budgets/denoisers
fail, nor an isolated attribution to capacity, noise conditioning, optimizer,
loss balance or the three-step sampler. The full world-model goal stays active
and unproven. Actor/critic, CDP comparison and the game campaign stay parked.
No extension, broad assessment or successor is selected.

[Protocol](../experiments/2026-11-09-breakout-denoising-fit.md).
Root `/mnt/data/kindle-breakout-denoising-fit-20261109`. Experiment declaration `ab14c014`; unchanged native source
`0c705e12`, SHA256 prefix `db6b433d`, reused from the closed broad trial.
No rebuild or completed backend fixture, collector or old learner is repeated.

## Test and result

The [broad denoising trial](2026-11-09-breakout-denoising-world.md) loses the ball
even on fitting material. This diagnostic separately tests its basic fitting
ability using the eight exact32-arrival contact clips with starts
43/27/43/43/28/49/43/67. All224 real four-frame/action conditions are distinct;
there is no contradictory exact duplicate in this finite set. This does not
establish observability or deterministic futures for the general game.

Fresh2,048 updates use513 calls, trainB32 as four noisy copies of eight clips,
RGB64, Auto, development seed41001, LR1e-4/LaProp epsilon1e-8/AGC.3. The unchanged
1,787,843-weight/45-parameter U-Net and separate transition event CNN train with
uniform EDM residual MSE and the original event losses. Seven rolling blocks
cover224 target arrivals with65,536 exposures,292–296 each. Two positive prefix
rewards produce588 training targets. The evaluated suffix contains no reward
or ending positives. No object/RAM labels or pretrained weights enter learning;
no new emulator action/frame is collected. Recorded RGB retains its qualified
native max-pooling and single GPU resize.

| Final task | Supported endpoints per draw | Within1px, each of four draws |
|---|---:|---:|
| Ordinary16-action generated-history rollout |110/128|0|
| One action from actual past images |198/216|0|
| Observed noisy-frame denoising at sigma.5 |8/8|0|

Ordinary forecasts get actual16-frame prefixes and then generated frames only.
The separate privileged one-step test predicts source arrivals5..31 from their
actual past, never the target image. Target4 is excluded because the API's
observed-denoising diagnostic requires five observed arrivals. It is not a
27-step forecast or evidence of unassisted long-horizon competence. Missing balls
fail;18 unsupported positions per draw remain untested on either future task.
All1,376 individual endpoints are retained, with every paddle/brick/image metric
and reward/continuation output. The CSV explicitly separates source arrival
index from forecast horizon; the original review's teacher horizon field holds
the source target index. This clarification changes no score or model query.

All ordinary checkpoint curves at0/509/1021/1533/2048 remain retained. First/
last32-call mean denoising training loss falls0.772248 to
0.092168; this is not physical precision. Final observed brick
accuracy is1.0 on each draw and paddle within1px ranges.625–.875, while ball
precision stayszero. Ordinary1/4/8/16 checks all fail. The same failure with
actual-history one-step inputs means image-history refresh alone does not
rescue this tested recipe at this budget. Basic conditional denoising/sampling
fit remains unresolved; a unique cause is not established.

The [older pixel RSSM contact diagnostic](2026-11-08-breakout-world-contact-fit.md)
gets110/110 supported endpoints on each draw. Reuse only its closed data and
cached report, with no old-model query or old forecast RGB decoding. That model
has8,852,675 weights, a longer latent history, motion CE100 and a different
supervision workload; equal nominal2,048 updates are not equal data, capacity or
compute. Its success does not prove that this smaller denoiser must fit in the
new budget or establish an architecture ranking.

## Verification and limits

Training-call/loop wall time is**133.853825s/141.289141s**
(2m14s/2m21s). Loop includes checkpoints, first-state checks and ordinary curves;
initial/final privileged exports are outside it. These are wrapper timings,
not isolated GPU kernel measurements. CPU outcome review takes
10.080679s. No useful learning-speed gain is established.

Three CPU runner checks pass, covering budget/coverage, lane/action alignment
and future-image poisoning. Exact135 initial tensors match the retained actual
initial control. The first update changes all45 weight tensors; all513
six-component rows are finite, totals/ages/exposures agree, and all135 final/
frozen tensors agree. A separate restored process reproduces all five stacked
four-draw arrays (20 per-draw arrays) exactly, including images/events, without
updates. All142 declared inputs/69 CPU audit inputs pass.
Both nine-file host guards seal and the service is terminal; no real GPU fault,
reset/reload or NVML polling. The qualified runtime, wheel and native primitive
receipts are reused unchanged rather than rerun.

Original CPU preparation incorrectly assumes every source reward iszero and
fails before exposure counting or GPU allocation. Its source/log remain. The
separate repair preserves both positive prefix rewards and counts actual targets;
only the evaluated suffix haszero positives. No data is changed and no original
failure is relabeled passed. First-update review occurs before the second update;
there is no recurrence of the broad trial's launch ordering failure.

The fixed lane0/draw0 clip retains16 exact lossless frames with recorded RGB64,
actual-history one-step and ordinary rollout panels. All cases/draws are selected
in advance. All measurements are fitting-only, with reused development seeds.
Negative-only event evaluation cannot establish recognition/calibration of real
rewards or endings. The conservative below-brick detector cannot certify full-field
or above-top dynamics. No full-corpus gates are replaced by this easier check.

Precise physics/events, persistent removed-brick memory, natural calibration,
late/tunnel/second-wall/above-top coverage, untouched families and fresh
confirmation still govern the [full goal](../kindle_single_life_dreamer_plan.md).
Neither this failed fitting check nor a future memorization pass completes it.
