# Coverage teaches waiting and improves contacts; the world remains incomplete

The [fixed data/sampling experiment](../experiments/2026-11-07-breakout-world-coverage.md)
improves several causal forecasts with **unchanged model, native code and losses**.
The model stops inventing launches under NOOP, learns an immediate FIRE response,
and improves several contact/life-loss branches. Last-life ending prediction and
long forecasts remain inadequate; some free-flight, reward and brick metrics
regress. **The full world-model goal is unproven.** Actor/critic remains parked.

All selected phases are closed. Retain both development checkpoints and the new
fitting data, without treating this as a universally better replacement. Do not
repeat this learner/collector or extend it unchanged. The next justified hypothesis
is direct forecast supervision, not a capacity increase or a sampling-noise fix;
no successor learner is selected here.

## Fixed change and cost

Control is the [motion-H0 world](2026-11-07-breakout-world-motion-attention.md).
Keep8,852,675 parameters,RGB64/12M/B8/T32/full BPTT/Auto,seed41001,4e-4 rates,
reconstruction100,KL1/.1,free-nat1,mixing.01,motion cross-entropy,H0 and external
reward/continuation1. All204 initial tensors and18 initial-development arrays
match the control exactly. This is a reused matched initialization, not a fresh
learner seed or independent confirmation. No pretrained encoder or object targets.

Add16,384 fitting actions/65,492 emulator frames from eight new451001-based
streams, disjoint from retained fitting/development/challenge seed families.
Long non-FIRE waits, sustained random controls and deliberate paths to game-over
yield44 natural endings,2,979 complete waiting windows across eight lanes and
max96 ball-absent non-FIRE decisions. Preparation loop/checkpoints cost19.248s,
excluding native initialization; age0/all204 tensors stay unchanged. All18 commands,
native image hashes, action/reset/frame alignment and coverage gates pass.
Supplement max score14 and no above-top-row ball states: later-game coverage
remains missing. Labels/phase tags select data but never enter model inputs/losses.

Use4 retained-uniform +2 supplement-uniform +1 waiting +1 terminal-focused window
per update. This changes volume, behavior coverage and sampling together; it does
not isolate their individual causes. Source plan is pinned before construction.
It samples32,849 of32,850 available arrivals, exposing10,909 terminal targets
versus2,649 previously,34,440 positive-reward and10,488 natural-reset rows.
2,097,152 replayed arrivals are not new environment interactions. The two fitting
corpora together contain32,768 executed actions/130,995 emulator frames.

Native training **529.515s (8m49.515s)** versus530.143s; full measured training/
evaluation/checkpoint loop796.900s versus734.272s. The new fitting evaluation has
more ending origins. Both timers exclude native construction; other project
processes were present. These are elapsed costs, not isolated speed improvements.
No build/port/precision experiment or actor/critic update occurs.

## Matched original development

All dense-development source arrays and locations match,128 origins/one fixed
draw. Retain misses and the fixed final8,192-update checkpoint. Four-draw periodic
curves are retained separately; no favorable draw/checkpoint replaces the endpoint.

|Measurement|Retained motion H0|Coverage H0|
|---|---:|---:|
| Observed balls within1 resized pixel |107/112 (95.54%)|108/112 (96.43%)|
| Observed ball detections |110/112|111/112|
| Observed paddle within1 pixel |97.66%|98.44%|
| Observed brick balanced error |.206631|.223954|
|1-step ball within1 pixel|93/109|94/109|
|4-step|35/109|33/109|
|8-step|8/107|14/107|
|16-step|3/114|3/114|
|32-step|1/95|0/95|

On the same10 outcome-conditioned development ending slices, one-step ending
Brier improves.444408→.231761. These selected positives do not establish natural-
frequency calibration or separate perception, head fitting and rollout errors.
Long forecasts and unseen-brick memory remain weak despite preserved perception.

## Frozen fixed-plan development

Reuse the exact [closed oracle challenge](2026-11-07-breakout-world-branches.md),
without recollecting it or rerunning old models. Evaluate only the new final model,
history16 primary/32 secondary,eight fixed draws,84 groups,234.494s. All204
parameters/moments and age8,192 remain unchanged. These are reused development
origins, with33 distinct native contexts across the seven groups; repetitions
and duplicate source histories are not independent examples. No untouched-test,
calibration or reliability claim follows from this one learner and eight draws.

**Waiting is learned on this challenge.** Under NOOP, supported false-ball
detection at8 decisions falls95.3125%→0%; ball feature energy falls1.16702 to
approximately0. FIRE1 one-step draw matching improves0→68.36%, and proper ball
feature energy1.26433→.05084. Fire-and-move aliases and long launch trajectories
still need work; aggregate serving32-step energy worsens.41671→.56708.

**Contact forecasts improve.** Equal-plan primary ball-feature energy, lower
better, includes an absorbing ending atom weighted by model survival:

|Origin group|8-step retained→coverage|32-step retained→coverage|
|---|---:|---:|
| Free flight |.00950→.02059|.51840→.30597|
| Paddle approach |.33091→.09393|.85103→.25625|
| Brick approach |.02434→.02275|1.00856→.13811|
| Wall approach |.09905→.06435|.36877→.29655|
| Multi-life approach |.22109→.00453|.28347→.01652|
| Last-life approach |.23812→.07104|.66838→.15896|

This scores a reported finite mixture of decoded RGB-mean features, not full image
likelihood. The detector covers a restricted below-brick region. Low scores when
both feature labels are absent/unsupported do not prove correct physical position;
brick-boundary and later-horizon results need the image panels. History32 helps
some cases and hurts others; it does not consistently fix the world.

**Ending remains insufficient.** RIGHT ends every selected last-life oracle branch
by32 decisions. Mean model probability rises **.581%→50.211%**, but remains far
below100%; RIGHT Brier improves.988414→.389513. Across all four last-life plans,
ending Brier improves.522471→.144566, failing the declared.10 gate. Predictions
are strongest on the three origins that also end under NOOP, and still weak on
the five additional RIGHT-induced endings; do not claim an isolated head/lives/
contact cause. Primary-history16 non-ending-group mean32 ending probabilities
all remain below.621%,
passing the5% false-alarm gate. There is no life-loss penalty.

Reward changes are mixed. Free-flight8 positive-event Brier worsens.213156→.402050;
paddle-approach32 improves.524527→.090997; last-life32 improves.133930→.000173.
Do not promote on a selected event or trajectory. These categories alter event
frequencies and eight oracle samples cannot establish population calibration.

## Metric failure and additional CPU evidence

The declared FIRE one-step90% pair-match gate is **invalid for this oracle sample**.
Its oracle leave-one-out rate is62.946%. A separate CPU unit-disk covering audit,
allowing any prediction position, proves an empirical maximum of **78.125%**
within1 pixel, averaged over the supported origin targets. Any randomized forecast
is a mixture of positions and cannot exceed that bound on this fixed pair score.
The original failed gate remains false; it is not relabeled passed.68.36% matching
also does not imply superiority over the oracle: modal predictions can raise
pair matching without calibrated distributions. Future protocols should use proper
scores and oracle uncertainty, not repeat this unattainable threshold. No new
passing threshold is retrofitted. The ending gate independently still fails.

A separate CPU decomposition of the retained model, requiring every model/oracle
ball detection and live oracle endpoints, separates squared centroid bias, model
spread and oracle spread. At16-step free flight,31/32 origin-plan cases have
3.336-pixel RMS centroid bias,1.913-pixel model spread and zero oracle spread.
These are descriptive complete cases, not independent examples or a causal
attribution. They show that sticky randomness alone cannot explain those errors;
direct forecasting accuracy remains a useful next target. No prediction reruns.

## Identity, failures and closure

Root `/mnt/data/kindle-breakout-world-coverage-20261107`.
Native/source3fa37b6d, SHA256
`eafe8359ae5019ea0a5a7517b7acfe75aacaf64b4e84a3bcd9ec4548cf939dc5`;
exact retained motion runtime. Python learning source1f5358d5, archive
`source-1f5358d5.tar.gz`, SHA256
`54a6b4af8c3648b1903e111784c6eea28e1b5f3f15634a7fc0b1f70f50e02f8e`.
Eight affected CPU tests and the original8,192-plan/count equivalence review pass.
Collection35,learning80,branch482 input pins verify at closure. All68 logical
parameters change,8,192 nine-loss rows are finite and weighted totals agree.
204 initial/frozen tensors,18 initial-development arrays and18 final-restore arrays
are exact. All four host guards/sealed audits pass; no GPU fault, numerical
failure, host recovery or repeat of a completed GPU stage occurs.

The **original learning queue is protocol-failed**, not relabeled complete. Its
expected total32849 mistakenly equaled the unique sampled count; total is32850.
Combined lane11 (supplement lane3) arrival2053 is unsampled. Training itself
finishes and the native guard passes. A separate CPU post-train audit verifies
the completed checkpoint/source plan/counts, followed only by the unstarted
restore under a new declaration. Preserve the first CPU comparison SyntaxError
and its separate CPU-only repair, plus the invalid original FIRE gate/review.
Do not restart the failed queue or repeat the learner to fix bookkeeping.

`closure-review.json`, `learning-review.json`, `comparison-repaired.json`,
`branch-comparison.json`, `gate-review.json`, `fire-metric-review.json`, native/RGB
data, checkpoints, all failures/logs and arrays remain outside Git. The fixed
`branch-forecasts.png` panel shows oracle/model draw0,lane0,hist16; oracle ended
endpoints and model alive weights are marked. `development-curves.png` retains
the periodic four-draw comparison. Source/native identities stay pinned.

All stages close. Coverage helps without increasing native training cost, but
does not solve the full world. Preserve both models/data and the original failures.
The next experiment must separately justify forecast supervision and its checks;
no unchanged extension, automatic learner, broad seed campaign or actor/critic
resume is selected. Later-game/second-wall coverage, removed-brick memory,
untouched tests and fresh learner confirmation remain unfinished.
