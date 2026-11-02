# Breakout 25M capacity: bounded execution and cost check

Select one fresh 8,192-action check of the existing 25M preset using the retained
12M recipe. The [12M control](../results/2026-11-02-cdp-breakout-replay-value1.md)
scores 70.750 sampled / 143.667 greedy in 21m27.910s, with zero two-wall
successes. Earlier 4M→12M capacity/batch growth improved frozen gameplay, but
that comparison used the weaker encoder rate. The stronger current recipe has
not tested 25M. This is a test of useful capacity at measured cost, not evidence
that inadequate capacity causes the remaining control errors.

Change only the model preset, 12M→25M. Deterministic state grows 2048→3072,
hidden/head width 256→384, categorical classes 16→24 (32 variables), and CNN
depth 16→24. Encoded RGB64 width grows 1024→1536. Several network components,
initial values and visited states change together; this cannot isolate the
encoder, recurrent memory or policy head. Do not require matching initial values
or pre-update trajectories across different architectures.

Keep centered CDP, Auto precision, native pooled RGB with one GPU resize to64,
N8/B16/T16/microbatch16/H15/R16, encoder/base4e-5, behavior8e-5, dynamics4e-4,
replay-value1 with imagined bootstrap, expert100/unit weights/world gradients,
all-state corrective labels, 20% episode guidance/cap1920, actor-unimix.01,
action-effects1, replay100k and warmup1000. Preserve full18/sticky.25/repeat4/
noops0. Reuse development seed1009 and frozen seed4000001009; neither is an
independent confirmation. No weights, experience, belief, games or RNG transfer.

Use the unchanged native447fe50/SHA35ad0db and matching retained Breakout runtime
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Copy the exact qualified
replay-value1 runner, with no native build or production changes. Derive the
357-tensor schema from native network formulas and reproduce the actual 12M
schema first. Verify all actual 25M tensors against the new schema.

Train 8,192 aggregate actions / 489 updates within180s, first update376 and
residual credit.5; save initial and final checkpoints. Restore once for1,024
greedy frozen actions within60s, with zero updates and no guidance or labels.
Require exact config/counters/source identity, finite reports/tensors, changed
CNN/reward/actor/value parameters,357 invariant frozen tensors, full guided
pixel/action/label replay and host guards. CPU preflight exercises both actual
CLI paths before construction and rejects undeclared config changes. Keep120s
inner/150s outer replay,180s audit,480s main and600s collector caps. Check live
state; never overlap GPU jobs or restart successful/failed stages blindly.

The fixed cost screen is `training_seconds * 200000 / 8192 * 1.10 <= 2700`.
A possible larger-model life has a **45-minute** ceiling, reflecting its larger
capacity; this is not an extension or cap change of any completed 12M run.
The eventual comparison must report both absolute gameplay and initial-subtracted
return per training minute against12M. Higher cost alone earns no promotion.
This check measures execution and early cost only. It does not assess competence
or select a full learner. No automatic promotion, capacity sweep, retry, cap
increase, extension or reuse of check experience follows.

If separately selected after this check closes, a full life would use fresh
200k actions /12,477 updates,100k/6,227 recovery and fixed final/actual-initial
sampled and greedy24-natural-episode arms. Match that life's initial tensors to
this 25M check, not12M. Preserve the original90%-two-wall task gate and the
five-game goal. Root `/x/Temp/kindle-breakout-capacity25m-20261102/native-check`;
service `kindle-breakout-capacity25m-check-20261102`.

CPU preflight passes in1.060s (1.092s wrapper): both actual CLI paths stop
before construction,17 actual config cases reject undeclared changes, and the
unchanged cadence gives376 first update /489 check updates /6,227 at100k /
12,477 at200k. All338 retained-control pins verify. The independently derived
12M schema exactly matches its actual checkpoint; the 25M schema has357 tensors,
342 changed shapes and **36,608,401 trainable parameters**. The exact retained
runner is unchanged. No native or environment construction occurs. This is CPU
preparation only; actual 25M execution, learning, restore and cost remain unmeasured.

## Bounded result

The check completes8,192 actions/32,722 frames/489 updates in84.864s;
1,024 frozen actions/4,096 frames take1.343s with zero updates and no natural
episodes. All380 pins,357 actual schema/finite/invariant frozen tensors,
finite supervision,parameter changes,full guided replay and guards pass.
Main/collection succeed10:39:11.653/10:39:18.624 UTC November2, sourcea8144ec;
terminal confirmation10:40:52.368. No failures or stages repeat.
Conservative200k projection37m59.071s passes45min, but short training costs
77.95% more than12M. [Result and receipts](../results/2026-11-02-cdp-breakout-capacity25m.md).
Both services are terminal; no allocation or full learner is active at closure.
Do not repeat this check,collector or confirmation. A full comparison remains
a separate decision; no gameplay benefit is established.
