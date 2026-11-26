# Breakout terminal supervision is present but sparse

The retained learner receives **5,376 terminal replay rows out of 3,194,112
rows (0.1683%)** across 12,477 updates. Its source experience contains 263 natural
game endings in 200,000 actions (0.1315%). Terminal targets are present in the
existing continuation loss; the data do not support an explanation that endings
are excluded from replay. Repeated exposures do not establish coverage of every
individual ending.

In the last 50,000 actions, terminal rows fall to **0.1034% of replay**, and
**76.352% of updates contain none**. Some terminal-containing batches show clear
recognition of endings: 205 have an entire conservative terminal-prediction mean
interval below .5, versus 10 entirely above .5 and 524 unresolved. The target is
zero. These are batch means from changing models and replay mixtures, not counts
of independently recognized game endings.

The [fixed CPU audit](../experiments/2026-11-04-cdp-breakout-terminal-replay.md)
takes **.493s**, with no model execution, emulator replay or learning. Retained
gameplay stays 70.750 sampled /143.667 greedy in 21m28s, with zero of24 two-wall
successes. The original 864/90%-two-wall and five-game goal remain unfinished.

## Exposure and what the existing summaries establish

| Training actions | Updates | Observed game endings | Terminal replay rows | Replay terminal fraction | Updates without terminals | Terminal prediction mean bounds |
|---|---:|---:|---:|---:|---:|---:|
|1–50,000|3,102|107|1,938|.2440%|1,643|[.425484,.993211]|
|50,001–100,000|3,125|63|1,505|.1881%|1,891|[.329224,.940108]|
|100,001–150,000|3,125|50|1,106|.1383%|2,185|[.184174,.789071]|
|150,001–200,000|3,125|43|827|.1034%|2,386|[.106423,.669123]|
|Whole run|12,477|263|5,376|.1683%|8,105|[.299810,.886492]|

The bounds aggregate terminal prediction sums and divide by terminal exposures;
they are neither point estimates nor statistical confidence intervals. The
whole-run terminal-count histogram is 8,105 batches with zero, 3,493 with one,
766 with two, 102 with three, 10 with four and one with five.

| Training actions | Terminal-containing batches | Mean bound entirely below .5 | Entirely above .5 | Unresolved |
|---|---:|---:|---:|---:|
|1–50,000|1,459|0|399|1,060|
|50,001–100,000|1,234|12|187|1,035|
|100,001–150,000|940|102|37|801|
|150,001–200,000|739|205|10|524|
|Whole run|4,372|319|633|3,420|

Late below-.5 batches contain 225 terminal exposures; above-.5 batches contain
20. Most terminal-containing batches remain unresolved by these summaries.
Overall replay MAE falls from .017294 in the first window to .001767 in the last;
reported world continuation BCE falls from .045152 to .022457. These overall
metrics are dominated by nonterminal rows. The BCE output is a separate training
calculation and is not algebraically combined with the prediction/MAE summaries.

Exact native447fe50 source confirms a sigmoid continuation head, terminal-arrival
target zero, nonterminal target F32(1−1/333)=.9969969987869263, and continuation
loss scale1. Intermediate life losses remain nonterminal. The reported prediction
mean and MAE use the same replay-posterior array. Every terminal count matches a
unique exact F32 target-mean enumeration and an independent integer estimate;
all reward-class denominators equal256.

The [protocol](../experiments/2026-11-04-cdp-breakout-terminal-replay.md) derives
conservative terminal bounds from the prediction mean and MAE. It explicitly
allows nonterminal predictions above gamma and 5e-5 numerical error in each
mean. Treating their algebraic combination as an exact terminal mean would be
incorrect. All per-report rows and unrounded results remain in the evidence root.

## Consequence for the investigation

Terminal supervision exists, is rare, and shows some fitting in replay. The
[frozen one-step diagnostic](2026-11-04-cdp-breakout-ending-feedback.md) still
predicts mean .952850 continuation at 24 selected game endings. Its prior
forecasts and these changing replay-posterior predictions use different states,
draws and cohorts; their difference is not a measured posterior-to-prior gap.

The useful next question is whether the final model recognizes the same terminal
arrival after observing it, compared with its prediction before that observation.
That would better distinguish an arrival-representation/head problem from a
forecasting problem. This audit does not select a new loss, weight, sampler,
native change or learner. Simply increasing the whole continuation loss would
also increase the weight of abundant nonterminal examples. No causal optimization
or representation defect, population calibration or gameplay gain is established.

## Validation and closure

Sixteen CPU tests pass, covering every possible terminal count, 1,000 synthetic
bound cases, F32/JSON rounding, nonterminal overshoot, extreme counts and malformed
inputs. Preflight verifies the previous 554 pins and39 artifacts, original
training completion/replay hashes, checkpoint configuration and four exact native
source files. All606 execution pins remain unchanged.

One .493s analysis runs on one CPU core at nice10, under the fixed60s cap;
its wrapper takes3.668s. Independent Decimal arithmetic recounts all12,477
reports, joins every saved row to the original log and reconciles all263 terminal
transitions with natural episodes. The3.710s closure completes
**03:17:31.882 UTC November4**, source`bf2011ee`, analysisPID1102416. All22 final
artifact hashes verify. No failure, repair or completed-stage repeat occurs.

Root `/x/Temp/kindle-breakout-terminal-replay-20261104` is closed. Native and
emulator imports are blocked; no GPU descriptor, learner or GPU allocation is
active. Original training, checkpoints, captures and collectors remain unchanged.
Do not repeat the analysis or closure; check live state before future work.

| Artifact | SHA256 |
|---|---|
| Declaration |`3a71ce3b137d69b5328c7ddff1bed7da6fd7c1f3d25d8d194de171ef61e39dae`|
| Analysis |`15718623934828dfb18e85b3063979108e25360a957ac623a506d07d2e902955`|
| Per-report rows |`526d759a3ff785786354ee54365a1a75cacd6b4586936ec00dd66062a227d43c`|
| Independent terminal summary |`661b62d0040bfe3d5a425fb5f9d7a78dcd1cff514fbf9060ea3c4b0a5b4aa38b`|
