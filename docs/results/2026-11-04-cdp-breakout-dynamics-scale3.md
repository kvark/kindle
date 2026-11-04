# Stronger prior fitting lowers frozen Breakout scores at the same cost

The [fresh200k comparison](../experiments/2026-11-04-cdp-breakout-dynamics-scale3-200k.md)
finishes **200,000 actions /799,607 emulator frames /12,477 updates in21m33.324s**.
Increasing dynamics KL scale1→3 lowers greedy143.667→105.667 and
sampled70.750→62.208, with all zero of24 two-wall successes. Training takes0.42%
longer. Retain dynamics scale1; the original864/90%-two-wall gate and five-game
goal remain unfinished. No extension, coefficient sweep, second learner or
automatic successor is selected.

## Full 200k comparison

| Measurement | Retained scale1 | Candidate scale3 |
|---|---:|---:|
| Training time |21m27.910s|21m33.324s|
| Frozen greedy mean |143.667|105.667|
| Actual-initial greedy mean |2.208|2.208|
| Greedy gain/min above own initial |6.590132|4.799649|
| Frozen sampled mean |70.750|62.208|
| Actual-initial sampled mean |1.417|1.417|
| Sampled gain/min above own initial |3.230038|2.820253|
| Two-wall successes, either final mode |0/24|0/24|

Primary greedy score falls26.45%, and initial-subtracted gain/min falls27.17%.
The paired lane difference is−38.000 points,95% interval[−69.167,−8.083]; the
gain/min difference is−1.790483, interval[−3.243235,−.399936]. Sampled score falls
12.07%; its−8.542-point interval[−16.964,.667] crosses zero. The locked preference
rule fails. These10,000 seed0 bootstrap resamples cover eight reused evaluation
lanes, not independent training seeds or broad reliability.

Both actual-initial semantic traces reproduce the retained control:799 sampled
and794 greedy events. Each of four arms selects the first three natural episodes
per eight lanes, with zero updates/truncations and357 invariant tensors. Together
the captures take88,584 actions/354,176 frames/56.403s. Full recorded-action,
outcome, frame-count and boundary replays/videos pass. Ordinary frozen logs lack
live pixel hashes; no equality to unavailable live images is claimed.

### Losses and cost

Late raw KL rises from11.815355 to13.872131 despite the stronger fitting weight.
The candidate's final50k clipped dynamics KL also averages13.872131, versus
11.815389 retained. The publication reader independently recounts all12,477
candidate reports and reuses the already verified retained review. These are
means on each learner's changing replay, with jointly changing representations
and shared recurrent gradients. They do not establish held-out prediction error
or explain the gameplay regression. This result gives no support for adopting
the higher coefficient, or for claiming that a larger weight fixes prior fitting.

| Action window | Retained raw KL | Candidate raw KL |
|---|---:|---:|
|0–50k|6.215493|7.323053|
|50–100k|11.217215|12.776863|
|100–150k|11.457282|13.793472|
|150–200k|11.815355|13.872131|

Candidate imitation cross-entropy on labeled rows falls1.208124→.568945 across
the run; the retained late mean is.560514. Replay fitting is not frozen competence.
Actual teacher overrides are84,836/200,000 (42.418%), versus43.0585% retained.
Guided online returns remain separate from the frozen scores.

Construction takes2.550s separately. Recorded stages take1,148.626s learning,
63.181s acting,62.778s environment,12.402s observation,1.086s reset and.552s
checkpointing. These wrapper timers do not isolate GPU kernels. Startup,
recovery and publication preparation CPU work co-hosts with training; retain its
time without correction. The100k verifier takes.458s (.514s observer wrapper),
its separate journal/receipt review.076s and publication preparation review.300s.

### Completion and evidence

Full helper,plot and comparison qualification passes3.867s; all five actual
queued commands pass.523s with no native-agent/emulator construction.

All145 current/338 retained pins,357 initial tensors matching bounded/retained
controls,47 pre-update events,finite weighted losses,correct cadence and
100k/6,227 read-only recovery pass. First update is376; residual credit is.5.
CNN,reward,continuation,actor and value arrays change. The108.214s training replay
verifies all recorded pixels,actions,labels and overrides. Guards,curves,coverage,
exposure and comparison pass. The curve was visually reviewed after completion.

The main service launches18:01:29.253 UTC November4 at source
`b222c0c8b112e92520efc3877f2d3c48552a3e36`, PID1169640/invocation
`5606b6db548e450bbbe94b4565f3a2b7`, and reports completion18:27:07.476.
Original collectorPID1169643/invocation`d88f36a9084b42909401fd104227af19`
completes18:27:46.942 after2.464s terminal audit. Recovery succeeds earlier.
The separate2.635s publication review completes18:28:08.346, verifies55 completion
artifacts and original journal identities, and independently checks means,
preference and the original gate. All three original services are terminal success.
No full-run stage fails or repeats. The earlier exploratory reader and canonical
Python import failures and their CPU repairs remain preserved below.

Native447fe50/SHA35ad0db,teacher and retained settings stay fixed; private
runner52593379 exposes the coefficient. Shared recurrent gradients and subsequent
experience change, so this is not an isolated prior-head intervention. The bounded
experience is not transferred. Root
`/x/Temp/kindle-breakout-dynamics-scale3-20261104/full` is closed; no GPU allocation
is active at review. Do not restart the learner or collectors. Check live state
before any separately justified work.

| Full artifact | SHA256 |
|---|---|
| `declaration.json` |`1c584615c7cc11224224c1c88b0fa679d974e48f8e6361bfe8d88e5ea335269a`|
| `recovery-verification.json` |`5c07f606c5f94b26f71c328a9dd8406bf28fa8a7da9a062d352427ec933f1172`|
| `candidate/learning-review.json` |`5c11048f46f656b693a12d018c886557aee1ea115ea32c7effacdbba2d16924a`|
| `candidate/guidance-replay.json` |`0acd8f4e7ef9d4eeeb9925e94aecf534a6b14053f676acd66e2e78d679441da1`|
| `comparison.json` |`1ec6b49dbbd55da344c3a873e41dbca8fdd36f900199ac775c6042e32e96f174`|
| `completion.json` |`c69abfe2b7a64d45770c077bce459e91dd0b7fe9e8d8d47607bd69b8f1ae4e2c`|
| `terminal-summary.json` |`35f09ea473d198611346017edfd131563da67c17ba74494100454ed9eb3fd056`|
| `publication-review.json` |`bb31472bc04952d04527ff3036b54f2e173fc634085875335bfe1f3fe4dd96d7`|

## Earlier bounded execution and cost check

The [bounded dynamics-scale3 protocol](../experiments/2026-11-04-cdp-breakout-dynamics-scale3.md)
completes **8,192 actions /32,719 emulator frames /489 updates in46.855s**.
The padded200k projection is **20m58.323s**, below the30-minute cost limit.
Frozen restore completes1,024 actions/4,096 frames in.881s, with zero updates,
357 invariant tensors and no natural completed episodes. This establishes
execution and early cost, not gameplay benefit or a reliable speed improvement.

This pass selected the separate fresh200k comparison reported above. At bounded
publication its preparation was pending; no check experience transferred to the
fresh learner. The retained result remains143.667 greedy/70.750 sampled, all
zero of24 two-wall successes.

### Hypothesis and change

Increase only `loss_scales.dynamics` from1 to3. Native dynamics KL fits the
prior to a stopped posterior; the coefficient also changes gradients through
shared recurrent dynamics. Posterior KL scale.1 and both free-nat floors1 stay
fixed. This is not an isolated prior-head change or proof of a causal bottleneck.

The retained200k log's final50k actions have raw KL mean11.81535483168 and
clipped dynamics mean11.81538893456. The mean clipping increment is only
.00003410288. A.317s CPU review covers all12,477 reports, validates cadence and
recounts all four window means using independent Decimal arithmetic. These
aggregate measurements give little support for the free-nat threshold as the
main late fitting limit; they do not reveal the fraction or importance of
individual clipped states. Raw KL is not a gameplay-error measure.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,dynamics4e-4,behavior8e-5,policy/value/replay-value1,
continuation1,CDPfuture500,disagreement1,expert100/unit weights/world gradients,
all-state corrective labels,20% episode guidance/cap1920,sampled collection and
posteriors,actor-unimix.01,replay100k,full18/sticky.25/repeat4/noops0. Native
pooled RGB and one GPU64 resize remain. Seed1009 is fresh; frozen seed4000001009.

Native447fe50/SHA35ad0db and the teacher are unchanged. The private runner
SHA52593379 adds only `--dynamics-scale` to exact retainedcf4ea238; the repository
runner exposes the same option. Defaults and checkpoint-controlled restore stay
unchanged. No new native graph rule, parameters, build or runtime overlay.

### Execution and qualification

All76 current/338 retained pins,357 matching initial tensors,47 shared pre-update
events,489 finite reports,correct cadence and weighted world losses pass.
Maximum weighted-total discrepancy is.00009793 within tolerance. CNN,reward,
continuation,actor and value weights change. All357 final/frozen arrays remain
finite and byte-identical after evaluation. The4.622s CPU replay verifies every
training pixel/action/label/override and both host guards pass. Actual overrides
are3,759/8,192 (45.886%); guided training returns are not competence.

Training construction takes2.473s separately. Learning takes40.489s,acting2.499s,
environment2.550s,observation.504s,reset.097s,checkpoint.258s. Retained bounded
training takes47.690s, so observed total time is1.75% lower. This historical
short comparison and wrapper timers do not establish a backend speedup or
isolate GPU kernels. No timing correction is applied.

Both canonical and private CLI suites pass12 cases: default/explicit-default/
scale3 configurations for RGB/CDP and malformed,restore/evaluation overrides.
Native constructors are blocked. Actual-command preflight passes1.458s with
zero agent/emulator construction,357-tensor schema,47 initial events,
489/6,227/12,477 schedules,negative fixtures and the actual terminal collector.

Preserve two preparation issues. The exploratory reader initially searched
spaced JSON text in a compact log, found zero learner rows and raised IndexError;
the corrected review parses JSON. The first canonical CLI qualification fails
before test collection because the retained Python package lacks a helper
imported by the current runner. The repaired harness resolves canonical Python
helpers from the repository while retaining guarded native metadata. No
completed test or GPU stage repeats; original source/stderr/receipts remain.

MainPID1169197/invocation`345290df574f4dc6b901129528802f72` launches17:55:34.789
UTC November4/source`d7f8847a13dae78407b2d842df5eb7268ef545c2`, succeeds
17:56:39.956. CollectorPID1169200/invocation`7b72d017e93f49178a88afca2f2f4b66`
succeeds17:56:46.344. The2.528s independent saved-log/tensor recount closes
17:57:15.344,retaining92 artifact hashes. All original services succeed,with
no repeated GPU stage. Root `/x/Temp/kindle-breakout-dynamics-scale3-20261104`
is closed; do not restart its check or collectors. Check live state before work.

| Artifact | SHA256 |
|---|---|
| `declaration.json` |`8c862cc8be0e1a0d7affc1f9a1e55e15bd70be2808f65202a1e6e7b4ff8e03f4`|
| `preflight.json` |`62a923735565dd39668f6ebb75320a7eeec2b17f4b2b67baa3fe7be1d0ad57a7`|
| `kl-review.json` |`17426d00919c5a70e614d2c136007c0a8877f4c7a0a8c16d0d16d7d290ea4f8e`|
| `check/review.json` |`7bf0c786fe168fb2b213293b0d9c75b4d05875e7586f7de07590856a98e166b6`|
| `check/completion.json` |`6b4a004e6e36a049d1eb2008e35267a7f3897308e7bcf5d105ca925aff853035`|
| `terminal-summary.json` |`f96da3420ab3470b56a7f110ade7fd677fc41d704487153c9f67ae5688620645`|
