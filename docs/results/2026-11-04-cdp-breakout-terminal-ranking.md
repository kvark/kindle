# Terminal replay changes ranking as well as output levels

The terminal-mixture learner changes how it ranks game endings against intermediate
life losses. Posterior ROC AUC falls **.952691→.897569**, with paired difference
**−.055122** and lane-bootstrap95% interval **[−.141927,.008257]**. The interval
crosses zero: this small reused cohort does not establish a decline or equality
in discrimination. It provides no demonstrated ranking gain.

There are **261 reversed posterior pair orders out of2,304**. The control correctly
ranks194 pairs that the candidate gets wrong; the candidate fixes67 control errors.
No pair is tied. Consequently, a strict monotone shift of all continuation outputs
cannot fully describe the observed change. The earlier terminal-error improvement
is accompanied by altered rankings, not solely a new operating point on identical
rankings. This does not identify the cause of weaker gameplay.

Retain the existing sampler, continuation1 and70.750 sampled/143.667 greedy. The
mixture remains65.708/85.250, all0/24 two-wall. The original864/90%-two-wall and
five-game goal remain unfinished; no correction, threshold, learner or successor
is selected.

## Fixed comparison

The [protocol](../experiments/2026-11-04-cdp-breakout-terminal-ranking.md) uses
only the saved candidate/control outputs from the [matched arrival diagnostic](2026-11-04-cdp-breakout-terminal-mixture-arrivals.md).
Each of24 game endings is compared with each of96 intermediate life losses.
Lower continuation is the correct ordering for an ending; exact ties receive
half credit. No new model execution, emulator replay, fit or threshold selection
occurs. Posterior is the fixed primary; one-step prior is secondary.

| State | Control AUC | Candidate AUC | Candidate−control | Paired lane95% interval | Reversed pair orders |
|---|---:|---:|---:|---|---:|
|Posterior, primary|.952691|.897569|−.055122|[−.141927,.008257]|261/2,304|
|One-step prior|.791667|.734809|−.056858|[−.205306,.085938]|707/2,304|

For priors,419 previously correct pairs become wrong and288 previously wrong
pairs become correct, also with no ties. Both AUC estimates decline, but both
paired intervals cross zero. The fixed bootstrap draws eight whole lanes with
replacement5,000 times, seed20261104. It recomputes pooled AUC using lane
multiplicities, including all cross-lane pairs; it does not average per-lane AUC
or treat2,304 pairs as independent evidence.

The original below-.5 counts reproduce exactly:

| State | True game endings, control /candidate | Intermediate-life false ending indications, control /candidate |
|---|---:|---:|
|Posterior|5/24 /17/24|0/96 /7/96|
|One-step prior|0/24 /13/24|0/96 /21/96|

These figures retain the earlier distinction: terminal MSE improves substantially,
while intermediate-life MSE worsens. This ranking check does not change those
results. The outcome-conditioned cohort contains only life-loss endpoints, with
20% true game endings; it excludes ordinary ongoing frames. Neither AUC nor these
threshold counts establish population calibration or a production acting rule.
Different native versions, learned states and experience remain confounded. The
role of remaining-life information, shared representation gradients and the use
of continuation in imagined behavior remains unresolved.

## Validation and preserved output failure

Eleven CPU tests pass, including perfect/reversed/tied scores, strict monotone
invariance, exact pair-transition counts, malformed joins and agreement between
lane weights and explicit resampling. Native-model and emulator imports are
blocked; no GPU descriptor is opened. Preparation verifies all1,052 previous
pins and52 completed arrival artifacts. All1,097 new execution pins remain intact.

One analysis takes **2.082s**, including input verification, on one CPU core at
nice10. Process1118658/source`9deeeb19` completes **06:03:28.640 UTC November4**;
its wrapper completes06:03:30.669 with returncode0. No learning or GPU work occurs.

The independent verifier recounts all9,216 model/state pair orders, both paired
bootstrap intervals and the threshold counts, verifies inputs/artifacts, and
writes its completed summary at06:03:45.658. It then **exits1 while printing** that
summary because its final `json.dumps` call lacks an import. Preserve the original
script, completed arithmetic summary and failed execution receipt; do not describe
the original closure process as passing.

A separate **1.992s CPU output repair** verifies the already written summary,
all17 saved artifacts,1,097 pins and successful analysis identity, completing
**06:04:44.530**. It repeats neither analysis nor arithmetic recount. Closure status
is `complete_with_cpu_output_repair`. No tolerance change, fit, model execution,
emulator replay or completed-stage repeat occurs.

Root `/x/Temp/kindle-terminal-ranking-20261104` is closed. No learner or GPU
allocation is active. Do not rerun the analysis, original closure or repair;
check live state before future work. Existing captures and checkpoints remain
unchanged.

| Artifact | SHA256 |
|---|---|
|Declaration|`a4b377997e2ff40730fdadca4200cddbba25ac7d2795be61c8bfdf4063c1fa5f`|
|Ranking analysis|`87ec532339e874e78c862e5edbeae4103026850f6be9920715b3f4415eaddca0`|
|Main completion|`aa451a51095eccd7efa895b8a86cbd8a72c619120d4a84d7d3129b1709ed09ed`|
|Original arithmetic summary|`d3dc8fff54d061e67b220a7776a65d34d058a3ade344ce032fedc3e7f281ea77`|
|Original closure failure|`0391ed9ca5db9cae13140983dcf598c6b24c7f16e36afd511902aeb60f2c7c24`|
|Separate output repair|`dcd6173fd69743b674376916036b5b08bb9294f47f29527b7fce5ccb6665b8f1`|
