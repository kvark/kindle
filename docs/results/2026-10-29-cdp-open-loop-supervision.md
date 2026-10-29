# Three-step world supervision lowers frozen Qbert quality

## Full comparison retains the control

The separately selected fresh learner completes **200,000 actions /799,709
emulator frames /12,477 updates in30m15s** (1815.270s), within the35-minute
training cap. Fixed-final greedy mean falls **11984.375→9377.083**, a21.76%
decrease against the retained control. Sampled mean also falls16.95%.
Both new modes clear23/24 first pyramids, but miss the original mean15000 gate.
Retain the control; no extension, retry or seed confirmation follows this result.

| Frozen assessment | Retained control | H3/scale1 loss | H3 own initial |
| --- | ---: | ---: | ---: |
| Greedy mean, primary |11984.375|9377.083|80.208|
| Greedy first pyramids |24/24|23/24|0/24|
| Sampled mean |6759.375|5613.542|177.083|
| Sampled first pyramids |24/24|23/24|0/24|
| Original full-game gate |Fail|Fail|Fail|
| Training time |30m27s|30m15s|Zero experience|

The first three natural episodes per eight lanes supply24 episodes per arm,
with no guidance, labels, truncations or learner updates. All four arms together
execute99,200 actions /396,641 frames in59.555s. Their extra completed episodes
and partial tails remain in the evidence and do not change selection. Candidate
first-pyramid coverage is20.833/21 greedy and20.875/21 sampled.

Configuration remains12M48/Auto/N8/B16/T16/H15/R16,centered CDP,coefficient1,
actor-unimix0,full18,original rates,seed4013/evaluation4000004013. Guidance stays
at20% episodes with continuation/cap1920,expert scale100,all18 action weights1,
world expert gradients and all-state corrective labels. Optional H3/scale1
world supervision is the intended learning change. No bounded-check experience
or prior checkpoint state is reused. Native/runner revisions differ explicitly;
all357 initial tensors and both initial semantic traces match the control.

Training actually overrides87,421 actions (43.7105%), compared with the
control's42.958%. The remaining112,579 actions receive passive corrective labels.
All12,477 updates report expert supervision, with2,991,348 labeled replay rows;
these are repeated samples, not unique demonstrations. The44 completed guided
episodes average15313.636, whereas148 unguided online episodes average2273.649.
Those changing-policy training returns are not frozen competence.

All new auxiliary reports are finite and nonzero, with mean total47.2208,
prediction.0936888,reward.346874,continuation.029539 and valid fraction.998013.
This verifies loss execution, not improved forecasting or task performance.
The training comparison does not itself measure forecast accuracy; the separate
frozen comparison below now supplies that evidence.
Recorded update time is45.86% imagination/targets,28.91% world training and16.54%
posterior preparation. Total training is.64% shorter than control, but differing
trajectories and native/runner revisions prevent an isolated overhead claim.
See `full/completed-update-timing.json` for both complete timing sums.

All242 declared pins, configuration/counter/numerical checks,100k recovery at
6,227 updates,357 matched initial and unchanged frozen tensors,parameter changes,
initial semantic traces,full frozen outcome replays/videos,training pixel/action/
label replay,host guards,curves,coverage,exposure and fixed comparison pass.
The training replay takes614.126s. Main service
`kindle-open-loop-200k-20261029` starts07:58:25 UTC/source`fc3a21a` and exits
success08:41:03 on October29; CPU postprocessing exits08:41:29. All four services
are terminal success and the final collector passes. No GPU stage is repeated
and no host recovery occurs.

This reused development seed/cohort does not establish that all multi-step
supervision is harmful, or identify why this particular loss lowers performance.
It does reject this exact candidate as the next retained recipe. The five-game
goal remains unfinished. The separate [completed Breakout readouts](2026-10-29-cdp-breakout-representation.md)
find useful position/motion information but no simple action readout that beats
the recorded policy; they do not establish a causal control bottleneck.

Evidence: `/x/Temp/kindle-open-loop-20261029/full/qbert-open-loop-h3-200k-seed4013`.
The [full protocol](../experiments/2026-10-29-cdp-open-loop-qbert-200k.md) remains
unchanged as a declared input. Curves and comparison plots are retained outside
Git and visually checked.

- Learning review: `73134dd1749f6152e61013773c7b3b3b4fe3f018b6d4f165bb27052e315ee077`.
- Comparison: `90e8dbb2cfb2720dea1abd695561f7f7e83eeef6f086077c8d746d1c6d60d6d3`.
- Completion: `cebc1086ea832ea3f0f4470a1f268a6f0863ceb1e4be29c0dc929e6e38d251f3`.

## Frozen forecasts worsen overall, with a small early-game improvement

The separately selected comparison finds that **H3/scale1 does not improve
reward forecasts overall on the control's recorded gameplay**. Fifteen-step
cumulative reward MAE rises37.499→40.793 (+8.78%). The regression is concentrated
after the first pyramid; early-game error improves slightly. Lower gameplay
scores are therefore not accompanied by better aggregate reward predictions in
this measurement. Retain the control; no loss variant or new learner is selected.

| Measurement on identical recorded inputs | Retained control | H3/scale1 |
| --- | ---: | ---: |
| One-step prior reward MAE, all test actions |2.592|3.004|
| Observation-corrected reward MAE, all test actions |2.274|2.887|
| One-step positive-reward AUC |.978|.976|
| 15-step endpoint positive-reward AUC |.938|.931|
| 15-step reward-sum MAE, all origins |37.499|40.793|
| 15-step reward-sum MAE, before first pyramid |2.347|2.035|
| 15-step reward-sum MAE, after first pyramid |45.259|49.350|
| 15-step reward-sum MAE, unrelated actions |69.538|68.081|

Both models still predict recorded rewards much better than the saved actual
initial model's15-step MAE107.570. The fixed common-origin comparison also loses
accuracy at the trained three-step horizon: MAE7.096→8.121 (+14.45%). At horizons
1/3/5/10/15, control cumulative MAE is2.569/7.096/11.583/24.234/37.499 versus
candidate2.926/8.121/13.222/27.014/40.793. Endpoint AUC is mixed at intermediate
horizons: control.976/.966/.954/.945/.938 versus candidate.974/.961/.962/.951/.931.
Do not collapse that distinction into a claim that every forecast metric worsens.

Before the first pyramid, one-step prior MAE improves.176→.095 and corrected
MAE.143→.094. After it, prior MAE worsens3.123→3.643 and corrected MAE2.743→3.500.
The latter shows that the measured regression is not confined to accumulating
error during unobserved rollouts. It does not identify which head, representation,
training-distribution change or competing gradient causes the difference.

One-step terminal Brier score changes.00058415→.00058552, versus.00059036 for
always continuing. There are only16 test terminations; sampled15-step endpoints
contain onlytwo. Their apparently high AUC is insufficient evidence of improved
death-risk calibration. The first episode in each lane supplies calibration
baselines; the next two supply16 reused evaluation episodes. One-step support
is27,102 actions, and common15-step support is1,681 origins:304 before/1,377 after
the first pyramid. Complete baseline, phase, horizon and support metrics remain
in the evidence; the cohort and selection are unchanged.

Both final checkpoints re-observe **48,256 actions /192,980 emulator frames**
each, with zero updates:208.483s control /207.348s candidate. Two128-action checks
take3.570s/3.336s. Both use packaged nativeab80486/SHA b4cdf26. The control exactly
reproduces all152 arrays and eight raw-frame hash trajectories from its original
native447fe50 capture, and its summaries/profiles also match exactly. Its short
check reproduces19 saved arrays. Runtime compatibility is verified rather than
assumed. Candidate/control target arrays and all eight frame-hash trajectories
match; all357 tensors remain finite and unchanged per capture.

All65 current pins,39 resolved historical pins,config/source/counter/prefix
checks and host guards pass. The narrow Python comparison option permits only
the declared H3/scale1 training-loss difference; restored config still matches
checkpoint config exactly. It changes no agent setting. All34 forecast CPU tests
pass with native import prohibited. The previous helper bytes and hash resolver
are preserved, with original declarations unchanged.

Service`kindle-open-loop-forecast-20261029`,source`a3fbd03`,runs **09:42:29–09:49:38
UTC October29**,7m8.704s total. Terminal collection passes; no GPU stage repeats,
host recovery or active GPU allocation remains. Original training costs remain
30m15s candidate/30m27s control. The inspected plot and its receipt are retained
outside Git.

This is one reused development learner on the control policy's trajectories.
Candidate-policy state distribution, latent-target accuracy and causal policy
effects are unmeasured. Scalar re-observation uses different posterior draws
from the original vector actor. Unrelated-action predictions are sensitivity
controls without counterfactual outcome labels. H3 also supervises latent
features, so reward/continuation measurements do not evaluate every loss
component. The evidence rejects this exact candidate as a better retained
recipe; it does not reject multi-step supervision generally or justify another
training allocation automatically.

[Fixed forecast protocol](../experiments/2026-10-29-cdp-open-loop-forecasts.md).
Evidence: `/x/Temp/kindle-open-loop-forecast-20261029`,including both captures,
full summaries/profiles,compatibility check,terminal review and
`forecast-comparison.png`/SVG.

- Declaration: `baf3da7356e8a160749a41bb76d11fd43f242e4b2ad7fc1f69760782b5afb157`.
- Review: `042fc482357244500d41d01ad9babf328f66593f14289bb596b64cbe1e4df24f`.
- Completion: `788cd24f600b6e73dedadb1589708d02f55ff9514669d3b958360f9688b1d9d4`.

## Preceding numerical and early-cost checks

Optional prior-rollout supervision is implemented, default off. The fresh
Qbert12M bounded check completes **8,192 actions /32,754 emulator frames /489
updates in70.207s**. Frozen restore completes1,024 actions /4,096 frames in.792s,
with zero updates and no completed natural episodes. This establishes execution
and early cost; task benefit remains unmeasured.

The retained same-size short check took69.886s. The new check is.46% longer,
but different experience and execution times prevent an isolated overhead claim.
Linear200k projection is28m34s; adding10% gives **31m25s**, below the fixed35-minute
selection threshold. This is not a guarantee of full-run duration. No full
learner, extension or retry is allocated by this result.

## What changes

From the first posterior state in each replay chunk/row, execute three native
prior transitions using recorded actions without intervening observations.
Add detached centered CNN targets at steps2–3, and reward/continuation targets
at steps1–3. The existing deterministic CDP first-step term is not duplicated.
Use the existing straight-through categorical estimator with an independent
random stream. No trainable parameters are added.

Components average all rows, including masked zeros, and their supervised
horizons; existing component weights500/1/1 and new global scale1 apply. Masks
retain a valid terminal arrival, exclude reset/end crossings and remain zero
after a boundary. Targets and centering mean are detached. The initial
posterior/CNN, dynamics, prior and prediction heads receive gradients; no new
behavior loss is added. The original teacher-forced/expert losses remain intact.

The option is serialized only when enabled and exposed through fresh-only
`--open-loop-length` / `--open-loop-scale`. Checkpoints retain it; restore rejects
overrides. See the [exact protocol](../experiments/2026-10-29-cdp-open-loop-supervision.md).

## Validation and bounded results

Native source `ab80486` passes135 Rust CPU tests,611 packaged Python tests and
strict native/binding Clippy. CPU checks cover configuration/default compatibility,
RNG isolation, boundary masks, action ancestry, detached gradients, unchanged
parameter layout/inference and ordinary/staged graphs. CPU tests make zero
native construction attempts;10 native-construction cases are excluded.

Two separately guarded GPU checks pass: masked continuation values/derivatives
against F64; and rollout world-gradient routing, a smooth reward-head finite
difference, future-target/state invariance, action sensitivity and zero masks.
All189 declaration pins remain unchanged at completion. Service exits07:30:58
UTC October29, whole3.222s. Straight-through paths are checked for gradient
routing, not claimed to be derivatives of discrete samples.

Legacy frozen compatibility restores the retained trained12M Qbert model with
the option absent:1,024 actions in.823s,zero updates,357 unchanged tensors.
All128 transition rows match the retained greedy trace after excluding only
finite wall-clock timestamps. The original service exits07:36:27 at a CPU
comparison that mistakenly included `unix_time`; a separate CPU-only audit
passes07:37:39 with all29 pins unchanged and no GPU repetition.

The fresh check keeps12M48/Auto/N8/B16/T16/H15/R16,coefficient1,actor-unimix0,
full18,original rates,seed4013,20%episode continuation/cap1920/expert100/all18
weights1/world gradients/all-state corrective labels. Only the optional learning
loss is intended to change; native/runner revisions are explicitly different.
Evaluation uses4000004013 with greedy actions and guidance disabled.

All44 declared pins,357 matching actual-initial tensors,357 invariant frozen
tensors,config/source/counter checks,finite reports,CNN/reward/actor/value changes,
complete training pixel/action/label replay and both host guards pass. First
update occurs at376 actions; residual scheduler credit is.5. All489 reports
contain finite,nonzero auxiliary losses with audited weighted reductions.

| Auxiliary metric | Mean across489 updates | Minimum–maximum |
| --- | ---: | ---: |
| Scaled total |220.1032|42.3664–519.3414|
| CDP cosine distance |.431149|.076504–1.026136|
| Reward cross entropy |4.40201|2.16797–5.54126|
| Continuation BCE |.126851|.020558–.795089|
| Valid target fraction |.997912|.916667–1|

All489 expert reports remain supervised;117,261 labeled replay rows equal their
weight sum. Maximum raw world/actor expert-loss difference is8e−7. The episode
guidance rule actually overrides4,341 actions (52.99%);3,851 actions have passive
labels. Guided online returns are not frozen competence. Both learner and
evaluation seeds are reused development inputs.

Main service starts07:41:36/source`bade642` and succeeds **07:43:21 UTC October29**,
whole1m45.501s including construction,evaluation and CPU replay/audits. The
terminal collector passes with no repeated GPU stages. CPU preflight fixes for
a future checkpoint stub and premature empty output paths are preserved in
their original scripts/logs; they performed no GPU work. Earlier CPU compile
and interpreter-path errors are also retained. No training settings or caps change.

## Runtime and evidence

Root: `/x/Temp/kindle-open-loop-20261029`. Use its `runtime` directory and
`native-check/atari_vector.py`; the packaging receipt pins native,wheel,Python
sources and validation executable. Native SHA256:
`b4cdf26c5334ea7fde0e85a708809d5af11552e8de6f7a321d81e65437e2638a`.
The separate posterior-mode draft is retained as a patch,exact source copies and
hash receipt under `parked-posterior`,excluded from this runtime. Both disposable
build caches are released (2,959,631,566 file bytes); wheels,validation executable,
checkpoints and evidence remain.

- Gradient review: `6806137d0b7eb40d0d910259dfcf9524d4691f7d0f1fa55ff910b79f0cebeff8`.
- Compatibility review: `ffd2140d5fe62be0521d6327ba6962d35d1a75bb373f32a5a257ac85d6a146c7`.
- Bounded review: `3be34e68fe15eb9873c74ea55b18a56ba885dbc2afde6310fc31c9cacd79067b`.
- Completion: `f72896741831138dc3fc9cbf92906d15b21a471da1c5bf8a0611ac07f3f2be9e`.
- Packaging: `4dea42a0229db9d8e0faf1ea08511b3a0318ded210ce62ca5a3eb7bd977ecb6a`.

The bounded check selected no full learner by itself. The subsequent
[separately declared200k comparison](../experiments/2026-10-29-cdp-open-loop-qbert-200k.md)
uses a fixed35-minute training cap and completes as reported above. The
five-game goal remains unfinished.
