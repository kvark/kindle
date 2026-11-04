# Shorter return discount passes bounded execution and cost checks

The [bounded discount100 check](../experiments/2026-11-04-cdp-breakout-discount100.md)
completes **8,192 actions /32,729 emulator frames /489 updates in47.055s**.
The padded200k projection is **21m03.687s**, below30min. Frozen1,024 actions/
4,096 frames take.874s with zero updates,357 invariant tensors and no natural
episodes. This establishes execution and early cost; gameplay is unmeasured.

Select the separate [fresh200k comparison](../experiments/2026-11-04-cdp-breakout-discount100-200k.md)
after this pass. No bounded experience transfers. At bounded publication full
preparation is pending; retain143.667 greedy/70.750 sampled and the original
864/90%-two-wall gate. The five-game goal remains unfinished.

## Change and rationale

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

## Verification and completion

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
