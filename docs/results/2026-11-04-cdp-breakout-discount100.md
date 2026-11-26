# Shorter return discount lowers frozen Breakout scores

The [fresh200k comparison](../experiments/2026-11-04-cdp-breakout-discount100-200k.md)
finishes **200,000 actions /799,569 emulator frames /12,477 updates in21m23.435s**.
Reducing return discount horizon333→100 lowers greedy143.667→70.250 and
sampled70.750→56.083, with all zero of24 two-wall successes. Training takes0.35%
less time. Retain horizon333; the original864/90%-two-wall gate and five-game
goal remain unfinished. No discount sweep,extension,second learner or automatic
successor is selected.

## Full 200k comparison

| Measurement | Retained333 | Candidate100 |
|---|---:|---:|
| Training time |21m27.910s|21m23.435s|
| Frozen greedy mean |143.667|70.250|
| Actual-initial greedy mean |2.208|2.208|
| Greedy gain/min above own initial |6.590132|3.180916|
| Frozen sampled mean |70.750|56.083|
| Actual-initial sampled mean |1.417|1.417|
| Sampled gain/min above own initial |3.230038|2.555641|
| Two-wall successes, either final mode |0/24|0/24|

Primary greedy score falls51.10%, and own-initial gain/min falls51.73%. All
eight greedy lane means are lower. The paired score difference is−73.416667,
95% interval[−97.209375,−48.207292]; gain/min difference−3.409216 has interval
[−4.516728,−2.234156]. Sampled score falls20.73%; its−14.666667-point interval
[−27.916667,−1.708333] also lies below zero. The locked preference rule fails.
These10,000 seed0 resamples cover eight reused evaluation lanes, not independent
training seeds or broad reliability.

Both actual-initial semantic traces reproduce the retained controls:799 sampled
and794 greedy events. Each of four arms selects the first three natural episodes
per eight lanes, with zero updates/truncations and357 invariant tensors. Together
the captures take79,480 actions/317,755 frames/51.674s. Full recorded-action,
outcome,frame-count,boundary replays and videos pass. Ordinary frozen logs lack
live pixel hashes; equality to unavailable live images is not claimed.

### Interpretation and cost

The native configuration changes only return horizon333→100. Imagination stays15,
and continuation targets,imagined/replay credit and subsequent experience change
together. This comparison does not identify which of those changes explains the
regression. New held-out critic calibration was not measured, so it does not
establish whether the shorter-horizon value target became easier to fit.

Replay imitation cross-entropy on labeled rows falls1.193900→.546030 across
the run, slightly below the retained late.560514, despite worse frozen play.
These own-replay measures do not establish generalization or competence.
Actual teacher overrides are88,312/200,000 (44.156%), versus43.0585% retained.
Guided online returns remain separate from the frozen scores. Reused1009/cohort
and historical control timing remain development limitations.

Construction takes2.607s separately. Recorded stages take1,138.607s learning,
63.075s acting,62.891s environment,12.461s observation,1.096s reset and.581s
checkpointing. These wrapper timers do not isolate GPU kernels. Retain CPU
co-host time without correction:startup.360s,100k verifier.635s (.715s wrapper),
separate recovery receipt/journal review.077s and publication preparation.005s.

### Completion and evidence

Full helper,comparison and plot qualification passes3.667s; all five actual
queued commands pass.541s with zero native-agent/emulator construction.
All151 current/338 retained pins,357 initial tensors matching bounded/retained
controls,47 pre-update events,finite weighted reports,correct cadence and
100k/6,227 read-only recovery pass. First update376,residual.5. CNN,reward,
continuation,actor and value arrays change. The107.869s training replay verifies
all recorded pixels,actions,labels and overrides. Guards,curves,coverage,
exposure and comparison pass. The curve was visually reviewed after completion.

MainPID1173630/invocation`ba245c2b52b44d84b58322ec2008dbc3` launches18:43:09.877
UTC November4/source`11477bc7ae422ab3266b913036c4eae6258668ca`, completes
19:08:30.521. Original collectorPID1173633/invocation
`9021e661bc79456bb66e71bdb46aafc1` completes19:08:42.614 after2.515s terminal
audit. Recovery succeeds earlier. The separate2.919s publication review closes
19:09:01.626, verifying55 completion artifacts,original journal identities,
24-episode means,config,cadence,preference and the original gate. All three
original services are terminal success. No full-run stage fails or repeats.
The earlier CPU source-name assertion and repair remain preserved below.

Native447fe50/SHA35ad0db and teacher remain unchanged; qualified runner112be6c4
exposes the discount. No build,new parameter,experience transfer or optional
loss is added. Full root `/x/Temp/kindle-breakout-discount100-20261104/full` is
closed, with no active GPU allocation at review. Do not restart the learner or
collectors; check live state before any separately justified work.

| Full artifact | SHA256 |
|---|---|
| `declaration.json` |`b49d48e3cd307d10ea27bba54c30495785cbd50a6f004868d9cf1d0f9f8f7096`|
| `recovery-verification.json` |`64971cf98005270543329b05d64e920f19a84e14c94bb1b64f8a7be5bd59f502`|
| `candidate/learning-review.json` |`0e2af56b128a1209dc4b99c92d1f5e5d8b770fb257631759db45eeabbd56d433`|
| `candidate/guidance-replay.json` |`7594ec9d97a0116adfea164ffc5a8f1f3c96572b48b62c9632daecdecf9f3176`|
| `comparison.json` |`1dd5ac2c42079453ba50916f457af4f464dc53cf01e4e0d664c9cc53490a3c84`|
| `completion.json` |`031f895c3c30e56bf206d5c5c94fb3037423240d4ded9c3a127bcf82dc203ccc`|
| `terminal-summary.json` |`4f7a9647a32f3c16d15454a84655e80e690cc3ce1974be54170a84e45341b53c`|
| `publication-review.json` |`545c34ce0bd432f38d9fbc2eadf75a52332f285b7c9524a0daefa6655e090d2a`|

## Earlier bounded execution and cost check

The [bounded discount100 check](../experiments/2026-11-04-cdp-breakout-discount100.md)
completes **8,192 actions /32,729 emulator frames /489 updates in47.055s**.
The padded200k projection is **21m03.687s**, below30min. Frozen1,024 actions/
4,096 frames take.874s with zero updates,357 invariant tensors and no natural
episodes. This establishes execution and early cost; gameplay is unmeasured.

This pass selected the separate fresh200k comparison reported above. At bounded
publication its preparation was pending. No bounded experience transferred;
the original864/90%-two-wall gate and five-game goal remain unfinished.

### Change and rationale

Only native `horizon` changes333→100; imagination length stays15. The retained
critic's complete-return sampled RMSE9.231 trails its earlier-episode constant's
7.855. Near-zero reward forecasts on selected rescue branches do not distinguish
survival. The hypothesis is that more local credit may improve learning despite
losing distant reward information. Neither this rationale nor the bounded loss
establishes a causal bottleneck or gameplay benefit.

Exact native447fe50 source routes this coefficient to continuation targets,
CPU replay returns and GPU target parameters. CPU F32 arithmetic gives:

| Quantity | Retained333 | Candidate100 |
|---|---:|---:|
| Gamma |.9969969988|.9900000095|
| Constant-discount weight at15 actions |.955890|.860058|
| At100 actions |.740261|.366033|
| At333 actions |.367327|.035198|
| Half-weight distance, actions |230.471|68.968|

These describe constant discount only; learned continuation,lambda,resets and
bootstrap also affect targets. The arithmetic is not a new GPU target capture.
Continuation training,imagined/replay credit and subsequent experience change
together, so this is not an isolated critic intervention.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,dynamics4e-4,behavior8e-5,lambda.95,policy/value/replay-value1,
continuation1,dynamics-KL1,posterior-KL.1,free-nat floors1,CDPfuture500,
disagreement1,expert100/unit weights/world gradients,all-state corrective labels,
20% episode guidance/cap1920,sampled collection/posteriors,actor-unimix.01,
replay100k,full18/sticky.25/repeat4/noops0,native pooled RGB/one GPU64 resize.
Fresh seed1009/frozen4000001009. Native447fe50/SHA35ad0db and teacher stay fixed;
private runner112be6c4 adds ten CLI lines to retainedcf4ea238. No build.

### Verification and completion

Both canonical/private CLI suites pass14 cases. Default/explicit333 configs
match;100 changes only horizon. Boundary cases preserve nondegenerate F32
discount, and invalid/restore/evaluation overrides fail before construction.
The1.560s preflight checks both actual commands,15 config corruptions,357-tensor
schema,47 pre-update events,correct schedules,labels,caps,replay paths and the
actual terminal collector with zero native-agent/emulator constructions.

All82 current/338 retained pins,357 matching initials,47 pre-update events,
489 finite reports and weighted world totals pass; maximum total discrepancy
is.00012224 within tolerance. First update376,residual.5. CNN,reward,continuation,
actor and value weights change. Full5.572s training replay verifies every pixel,
action,label and override; both guards pass. Actual overrides3,854/8,192
(47.046%); guided training returns are not frozen competence.

Construction takes2.454s separately. Learning40.560s,acting2.520s,environment
2.568s,observation.508s,reset.091s and checkpoint.268s are wrapper timings,
not isolated kernel costs. Total is1.33% below the historical47.690s check;
this one short comparison is not a reliable speedup.

The original CPU source-review assertion expects an unformatted continuation
input name, while native code uses `continuation_target_{local_time}`. It fails
before arithmetic,CLI tests or GPU work. The original source,stderr and execution
receipt remain in `source-review-failure.json`; the exact-name repair passes.
No completed test or GPU stage repeats.

MainPID1173227/invocation`971be793e4f04cf8b16eae74a996d908` launches18:38:15.667
UTC November4/source`a5c9093a10106980bf2d7d828703761a34a04209`, completes
18:39:21.965. CollectorPID1173230/invocation`5f9070e02900441884f892ac73ba1126`
completes18:39:27.158. Independent2.481s closure finishes18:39:55.770,retaining
102 artifact hashes. Both original services succeed; no GPU stage fails/repeats.
Root `/x/Temp/kindle-breakout-discount100-20261104` is closed through the bounded
check. Do not restart it or its collectors. No GPU allocation is active at
closure; check live state before separately selected work.

| Artifact | SHA256 |
|---|---|
| `declaration.json` |`c7e4ac6419567b8cd426234c9d95abbf9f752a89d750fae24a8bd220216c0fd9`|
| `preflight.json` |`ed760bb1430e2b59fb679ea5f84783387edf24f408d6c177e303831cd2480f95`|
| `discount-review.json` |`b302b383b8f681f3edce3029d1d2f70ecd10af5354d180fc55d4405a8d78ad31`|
| `check/review.json` |`c64c6901776efc05082a20a23f58ee57b0e6d7bfc467fb99999366d5a8496c3c`|
| `check/completion.json` |`e5029018046c0d85c357924ae7351b53be089f67f0b6f70afb5a8257e5e5036e`|
| `terminal-summary.json` |`8c615c244a81f076ca7bb493c37c7219ea98e7d86b24c374ca34d9652556097f`|
