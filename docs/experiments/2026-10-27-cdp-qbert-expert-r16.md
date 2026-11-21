# Qbert expert imitation: more experience at lower replay ratio

**Completed:**500k actions /1,998,773 frames /62,470 updates in28m20s. Frozen
3752.083 /21 of24 first pyramids versus initial182.292 /0. All audits and
comparison pass; the original15k/90% gate still fails. No extension or next
allocation is selected. Launch source`bad8f55`; no recipe change after launch.

Selected one fresh **500k-action/R16** learner on reused seed4013 after its
completed200k/R32 result:3153.125 and17/24 first pyramids in20m07s. The same
expert recipe on1009 scores1083.333/2. Both improve over their actual initials,
but neither passes the original mean15000/90%-first-pyramid gate.

This candidate changes the action budget200k→500k and replay ratio32→16
together:2.5 times the experience and62,470 versus49,939 updates. The hard
training cap is30 minutes including construction. Prior unaided1009 at500k/R16
was weaker than200k/R32; improvement with expert imitation is a hypothesis,
not an established benefit. Seed4013 is selected after its stronger outcome,
so this is reused development evidence, not independent confirmation.

## Fixed settings and controls

Keep4M/RGB48/Auto/N8/B8/T16/H15/coefficient1/actor0/full18, original rates and
disabled actor/critic gradients. Native/CLI79c9a7f and all runtime flags remain
unchanged. Keep native max-pooled RGB and one GPU resize. Five percent of
episodes use the existing first-pyramid pixel teacher, capped640 actions;
expert scale1, NOOP weight.05, other weights1. Actual guided-action exposure
must be measured. The CPU continuation prototypes are not applied.

Fresh learner/guidance seed4013, frozen seed4000004013. No checkpoint, replay,
belief, game or RNG is restored. The actual zero-experience initial must match
all346 tensors of the completed4013 reference. First update248, then
256:2/264:3/272:4/280:5/288:6; final62,470 updates and zero residual credit.
Audit recoveries at100k/12,470,200k/24,970,300k/37,470 and400k/49,970 updates.

Evaluate the fixed final and actual initial policies with guidance disabled:
first3 natural episodes per lane,24 selected per arm, zero updates/truncations,
all346 tensors unchanged; retain excess episodes and partial tails. Each arm
has a600k-action/300-second ceiling. Compare against the completed200k/R32
4013 life, including exact initial tensors and semantic initial evaluation
traces. Preserve the original15k/90% gate, full source/config/finite/counter
checks, explicit expert-label audit, full training/frozen pixel-action replay,
videos, host guards, curves, coverage and exposure analysis.

No extension, cap increase, earlier-checkpoint selection, warm start, broad
seed queue or automatic successor. The CPU audit cap is1200 seconds; the total
service cap is3660 seconds, which includes the separately bounded evaluations
and audits. An observation timeout cannot restart or stop the training service.

## Runtime and evidence

Owner:`kindle-expert-qbert-r16-500k-seed4013-20261027`.
Artifacts:
`/x/Temp/kindle-qbert-continuation-20261027/qbert-4m-48-r16-expert1-cap640-500k-seed4013`.
Runtime:`/x/Temp/kindle-expert-20261026/runtime`.
Native SHA256:
`7c07d6702e37ca11aa20c1b35b48951f0ed01e0e1dfe72e187eac52253f8455e`.

The RX7900XT/RADV host remains on boot`a8648671`, kernel7.0.0-38, AMD module
`9BDBC866F55BF296F1B7FA2`, Mesa26.0.8. No host recovery or runtime rebuild is
performed. The existing host guard stops on genuine faults or numerical failure.

All67 declaration pins pass. The queue normalizes exactly to the completed
4013 queue after changing only budget, ratio, cap, expected counts and output
paths. The comparison reader passes against the completed control without
constructing an agent. Declaration SHA256:
`cf7006677c45e58c3aa3be8c446313c7230f6971fa9cdaabe61c3ca156e629d3`.

Selection evidence:
[completed two-seed comparison](../results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-transfers-on4013-with-substantial-seed-variation),
[continuation diagnostics](../results/2026-10-25-cdp-cheap-qbert.md#after-the-first-pyramid-movement-continues-but-the-teacher-is-limited).


## Launch and verified startup

Service starts00:17:08 UTC, main PID61973, invocation
`66573c6168d640428dfcdafa9d82ca17`. All67 pins stay unchanged. Actual initial
matches all346 completed4013 tensors, and all31 transitions through action248
match before the first learning update. Nothing is restored. The first six
updates match248:1/256:2/264:3/272:4/280:5/288:6.

The complete17,136-action prefix has2111 finite reports,1920 trusted actions
and633 reports containing expert labels. Last progress16,384/2018 is58.195s
with zero debt. These establish execution and provenance, not task benefit.
Read-only recovery, progress and completion monitors run as
`kindle-expert-qbert-r16-{recovery,progress,post}-20261027`.
Startup review SHA256:
`fa85b4a73f4ebe41fea8be66bb6d59b346cec84c0ddf95d71bcea061259b5195`.


## Completed recovery checks

| Checkpoint actions / updates | Audit UTC | Finite tensors | Complete prefix actions / finite reports |
| --- | --- | ---: | --- |
|100,000 /12,470|00:23:17|346|100,864 /12,577|
|200,000 /24,970|00:28:48|346|203,072 /25,354|
|300,000 /37,470|00:34:18|346|303,744 /37,937|
|400,000 /49,970|00:40:19|346|409,944 /51,212|

The same PID61973/invocation continued through completion. The400k audit
checks progress409,600 /51,170 at1386.221s, with zero debt. Recovery checks
establish execution; the fixed final outcome is reported below.

Independent score-color CPU prototypes complete on one affined core at nice10
while this learner ran. Their111.935s total measured collection/replay time
is a co-host timing caveat; this is not an isolated backend speed comparison.
They never construct an agent or change any of the67 active pins, recipe or
caps. A subsequent hop timing check adds29.848s and produces an
identical trace; total CPU prototype time is141.783s. All prototypes remain
outside production and unselected. See
[results and failed attempts](../results/2026-10-25-cdp-cheap-qbert.md#score-color-prototypes-remain-below-the-game-gate).

Recovery review SHA256s:
100k`7edd98a8c1d0e10f1be6517d98c0cd21a33a452fe3ae593f7b01cebc2f039e7c`;
200k`3868d0a42c39bf435212b1c80e7442504af34bb4767b5c0340006500b40daa5c`;
300k`439dd6525474fa7dea6f0f46a9fd9c3407fadf8cdaea446cfe7aa89c85d2864c`;
400k`a344db5d4303764631eb2c825f9ef6f2f1e2ab41030d4ff914884d413c3feef9`.


## Completed outcome

Training finishes the full500k budget in1700.485s, within the30-minute cap,
with62,470 finite reports, zero debt and346 finite checkpoint tensors. All67
pins, four recovery audits, full training/frozen pixel-action replay, explicit
expert labels, source/config/counters, guards and comparison pass. Initial
parameters match all346 control tensors, and both initial evaluation traces
have the same1140 semantic events. No GPU retry or CPU audit repair occurs.

Final/own-initial each select exactly24 natural episodes, zero updates and
truncations,346 unchanged tensors and retained partial tails. Frozen final
3752.083 /21 first pyramids /20.333-of21 coverage versus initial182.292 /0
/5.167. Final20,256 actions /80,984 frames take8.244s; initial8728 /34,875
in3.565s. The200k/R32 control scores3153.125 /17 /20.167 in20m07s training.
The gain costs40.88% more training time; neither15k nor90% is reached.

The main service exits successfully00:48:34 UTC,31m25.790s total including
setup, evaluation and CPU audits. Postprocessing completes00:48:42. All
services and observers are terminal; no extension, confirmation or successor
is selected. [Full result and limits](../results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-improves-first-pyramid-success-with-higher-cost).

Learning review SHA256:
`3cad7a4285536a7d484759f3f483250636a82dac9fb66d031749f2c8b1dcaa40`.
Completion:
`391554dfef3c781c9a432088d4f59c64e5b622594ff3bb9eea1b2ee5a8e84622`.
Comparison:
`58879af670c951c03fce61c5a07a2d50233a3643a4a6f1242eca4ff0e5ed9d8a`.
