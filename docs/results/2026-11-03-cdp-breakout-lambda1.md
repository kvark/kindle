# Return-lambda1: bounded checks pass and the full comparison is active

The separately selected [fresh200k comparison](../experiments/2026-11-03-cdp-breakout-lambda1-200k.md)
launches **04:13:38.340 UTC November3/source958d622**, with a30-minute training
cap and448 pins. MainPID972800/invocation44e2a4cbb446493f9db029c194c27300 and
progress,recovery,postprocessing andcollection observers are active. Startup
matches357 initial tensors against both retained control and bounded candidate,
plus47 pre-update transitions. All241 reports through4,232 actions are finite;
the first six updates follow376,392,408,424,440,456 actions. Startup checking
takes1.500s CPU wrapper time, retained without training-time correction.
Full frozen gameplay and100k recovery remain pending. Do not restart or overlap
this learner; root`/x/Temp/kindle-breakout-lambda1-20261103/full`.

The fresh lambda1 candidate completes **8,192 actions /32,727 frames /489
updates in50.121s**. A1,024-action/4,096-frame frozen check takes.868s with
zero updates and no natural completed episodes. The conservative200k projection
is **22m26.028s**, below the30-minute cap. This establishes execution and early
cost; gameplay benefit remains unmeasured. The bounded closure preceded the
separate full selection above.

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
collector or final reader fails or repeats. No Kindle allocation remains active.

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
