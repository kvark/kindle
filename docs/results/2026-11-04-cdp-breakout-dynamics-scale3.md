# Stronger prior fitting passes bounded execution and cost checks

The [bounded dynamics-scale3 protocol](../experiments/2026-11-04-cdp-breakout-dynamics-scale3.md)
completes **8,192 actions /32,719 emulator frames /489 updates in46.855s**.
The padded200k projection is **20m58.323s**, below the30-minute cost limit.
Frozen restore completes1,024 actions/4,096 frames in.881s, with zero updates,
357 invariant tensors and no natural completed episodes. This establishes
execution and early cost, not gameplay benefit or a reliable speed improvement.

Select a separate [fresh200k comparison](../experiments/2026-11-04-cdp-breakout-dynamics-scale3-200k.md)
after this pass. Preparation is pending at bounded publication; no full learner
is running yet. The retained result remains143.667 greedy/70.750 sampled, all
zero of24 two-wall successes. The original864/90%-two-wall gate and five-game
goal remain unfinished. No check experience transfers to the fresh learner.

## Hypothesis and change

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

## Execution and qualification

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
