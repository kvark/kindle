# Return-lambda1 lowers both frozen Breakout scores

The [fresh 200k comparison](../experiments/2026-11-03-cdp-breakout-lambda1-200k.md)
scores **45.667 sampled / 90.292 greedy**, below the retained lambda0.95
control's **70.750 / 143.667** by **35.45% / 37.15%**. Training takes
**21m52.277s**, 1.89% longer than the control. All four candidate arms have
zero of 24 two-wall successes; the original 90% gate still fails. Retain
lambda0.95. No extension, coefficient sweep, confirmation or automatic
successor is selected. The five-game goal remains unfinished.

| Measurement | Retained lambda0.95 | Lambda1 |
| --- | ---: | ---: |
| Sampled frozen mean | 70.750 | 45.667 |
| Greedy frozen mean | 143.667 | 90.292 |
| Actual-initial sampled / greedy | 1.417 / 2.208 | 1.417 / 2.208 |
| Training time | 21m27.910s | 21m52.277s |
| Initial-subtracted sampled return/min | 3.230 | 2.023 |
| Initial-subtracted greedy return/min | 6.590 | 4.027 |
| Final 50k replay expert cross-entropy | .560514 | .536404 |

Both learners use fresh 200,000-action lives and 12,477 updates; the candidate
executes 799,566 raw frames. Fixed frozen arms select the first three natural
episodes in each of eight lanes, with guidance and labels off, zero updates
and no truncations. Initial-subtracted return per minute falls **37.36% sampled
/ 38.89% greedy**. The four candidate captures total 82,720 actions / 330,692
frames / 78.455s, including extra episodes and partial tails excluded from
the fixed score cohort.

Only lambda changes from .95 to 1. Both imagined actor/value and real-replay
targets change, while discount, endpoint bootstrap and actual last/terminal
cuts remain. This does not isolate intermediate-value bias, imagined-reward
error or variance as the cause. Reused learner seed1009/evaluation seed4000001009
make this a development comparison, not independent reliability evidence.
Native447fe50 and the teacher remain unchanged; the qualified runner9eb21cca
differs from controlcf4ea238 by ten added CLI lines. The default remains .95.

All **448 pins**, 100k/6,227 retained recovery, 357 control/bounded matching
initial and invariant frozen tensors, 47 pre-update transitions, 800 sampled
and 795 greedy paired initial semantic events, 12,477 finite supervised
reports, full guided pixel/action/label replay, frozen replays/videos, guards,
curves, exposure and comparison checks pass. Recovery retains weights and
metadata only, omitting replay, live belief, games and RNG.

The original main, postprocessing and collector succeed **04:39:49.589 /
04:40:12.443 / 04:40:16.846 UTC November 3**, launch source958d622. All five
original services succeed. Terminal confirmation completes **04:40:46.018**
in 1.987s (2.008s wrapper). The original test-environment import failure and
prelaunch collector-date failure/repair remain preserved. No full-run stage,
observer, replay, collector or reader fails or repeats.

The fixed CPU coverage reader completes **04:40:51.786** in 2.956s (2.976s
wrapper), with 50 diagnostic/448 full pins and 282,720 actions / 1,130,258
frames reconciled. It reuses completed control counts without another replay.
Native training actions at pre-action scores216–431 rise **1,262 → 1,434**
(.631% → .717% of all training actions); neither learner visits432+. These
score bins describe exposure, not verified wall boundaries or causal coverage.
The modest increase and lower expert loss coexist with worse frozen gameplay.
Actual overrides are 90,797 (45.3985%); guided online last50 return81.54 is not
unaided competence. Keep all CPU co-host timings without wall-time correction.

Root `/x/Temp/kindle-breakout-lambda1-20261103/full` is closed. Recovery,
terminal and coverage readers are complete; do not repeat completed stages.
No Kindle allocation is active at closure; check live state before new work.

| Full-comparison artifact | SHA256 |
| --- | --- |
| Declaration | `aa991176cb6cc06fd585f52ceb3c2d4ccdb96e915208a8d0c6268e892a0410c8` |
| Learning review | `197989c673c72cea92c2d80c48ad61b497f605a2078394e2e0d274809496dc86` |
| Completion | `17c06d5d8d3998191d2add3d4aaa0b67ec42f11782c6155acd4bb5930b00e6f8` |
| Terminal summary | `4145274a5451ab098d0ce0a67207e55b40c3d5b38fb2b01db4935e141447ab11` |
| Coverage review | `eabe5271bad6879381c114286204cc72491070eba1988d1baa7616ded02524ad` |

## Bounded qualification before the full comparison

The fresh lambda1 candidate completes **8,192 actions /32,727 frames /489
updates in50.121s**. A1,024-action/4,096-frame frozen check takes.868s with
zero updates and no natural completed episodes. The conservative200k projection
is **22m26.028s**, below the30-minute cap. This established execution and early
cost before gameplay was measured. The bounded closure preceded the separate
full comparison above.

The [fixed protocol](../experiments/2026-11-03-cdp-breakout-lambda1.md) changes
only return lambda.95→1 against the retained replay-value1/behavior8e-5 control.
The same coefficient affects imagined actor/value targets and real-replay value
targets. Continuing traces use all available rewards before the final bootstrap,
while existing last/terminal cuts remain. It is not complete-episode Monte Carlo
training or removal of all model/critic bias; variance and imagined-reward errors
may increase. No isolated mechanism or causal defect is established.

Keep12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16, encoder/base4e-5,
behavior8e-5,dynamics4e-4,replay-value1,expert100 with unit weights/world gradients/
all-state corrective labels,sampled collection and posteriors,actor-unimix.01,
action-effects1,replay100k,warmup1000,20% episode guidance/cap1920,seed1009.
Full18/sticky.25/repeat4/noops0,native pooled frames and one GPU64 resize remain.
No prior weights or experience are reused.

Native447fe50/SHA35ad0db and the retained teacher/runtime stay unchanged. The
private runner adds exactly ten lines exposing `--return-lambda`; omission keeps
the native default, and invalid values or restore/evaluation overrides fail
before environment construction. RunnerSHA9eb21cca; runtime remains
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. There is no native build,
shader, optimizer, dependency, precision or checkpoint-format change.

The current runner passes515 Python tests using the existing qualified crop
package for test imports. The exact retained-runner overlay passes22 focused
tests and eight separate early-refusal cases. Four hand-calculated CPU cases
execute the unchanged return helper extracted verbatim from native447fe50:
continuing lambda1/endpoint dependence, terminal cuts, timeout cuts and.95/0
mixtures. Both pinned production call sites use the shared config field. This
small standalone Rust test does not rebuild the learning runtime or exercise
the unavailable resident-target GPU path. The initial ordinary-environment test
collection fails because `kindle._pixel_guidance` is absent; its exact sources
and failure receipt remain preserved. No test or GPU stage had run at that point.

Bounded preflight validates two actual commands,19 config cases,357-tensor
schema,338 retained-control pins and the376-action first update/489-update
cadence. All **385 selected pins**, finite reports,357 matching initial and
invariant frozen tensors,47 pre-update semantic events, changed CNN/reward/
actor/value tensors,complete guided pixel/action/label replay and host guards
pass. Actual overrides number3,839; guided returns are not unaided competence.

Original mainPID972191/invocation8e98c2a1b71d4f4fac22179cf7b58216 launches
04:06:32 UTC November3/sourcef466942 and succeeds **04:07:39.518**. Original
collectorPID972194/invocatione57b4accf7f648ac9eac2cde52232bd0 succeeds
**04:07:43.412**. The read-only terminal summary completes once **04:08:34.276**
in1.619s (1.638s wrapper), checking both original identities, all pins, completion
files, counters and cost arithmetic. No bounded learner, frozen stage, replay,
collector or final reader fails or repeats. At bounded closure, no Kindle
allocation remained active; the separately selected full comparison is now closed.

Root `/x/Temp/kindle-breakout-lambda1-20261103/native-check` is closed. Do not
repeat completed bounded stages. The separately selected full comparison retains
100k recovery andfinal/actual-initial sampled andgreedy24-episode assessments
against70.750/143.667 and the original two-wall gate. Its original CPU preflight
fails on a stale collector-date suffix before model/environment construction;
the preserved failure anddate-only observer repair precede the passing2.247s
preflight andfull launch. No cap increase,sweep,extension or automatic successor
follows. The five-game goal remains unfinished.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `aa472fd56cbd1c735d027274d88021b8208f3b84be8e455b843214534e594e2c` |
| Bounded review | `bd4724ac2006d0c136dd3e8042abaffb01ae8ac874b9243449bdb36d859cfe20` |
| Completion | `14b35911a3ce41bb7d76eae1733f9267a70ffeaaad1036ec3e10cad2918b0310` |
| Terminal summary | `21e6df091e9167ccce238e277a21cc416a10983c60a4e314fe7248c89e945948` |
| Exact-source return check | `25c81d0cf8676bee0929464c47101a4f57789048984a3a4a5e6590771e360132` |
