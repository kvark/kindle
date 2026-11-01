# Native imitation decay passes numerical and bounded execution checks

The optional schedule can hold strong early imitation and fade its direct loss
weight later. It is a qualified mechanism, not an established gameplay gain.
The retained Breakout encoder2e-5 learner remains the stronger compact control:
sampled25.292/greedy42.625 in22m27s, with zero of24 two-wall successes.

The [protocol](../experiments/2026-11-01-cdp-expert-imitation-decay.md) follows the
[local action interventions](2026-10-30-cdp-encoder-representation.md#local-action-interventions-limit-the-teacher-disagreement-diagnosis):
teacher disagreement is not necessarily an inferior action. Earlier gradient
attribution found large imitation contributions. Together these motivate testing
a fade, but do not establish that imitation causes the remaining control gap.

## Implementation and numerical qualification

Source`65ed45f61c9871b38a6a1b1b02cef254a573492a` adds optional
`expert_imitation.decay={start_update,end_update}`. The one-based LearnReport
counter holds the base weight through start, linearly reaches zero at end, and
stays zero. Actor and optional world imitation use the same weight. Raw weighted
cross entropy, label counts and target-weight sums remain unchanged; scheduled
reports also contain the applied weight. Existing optimizer moments, online
teacher guidance and corrective labels remain. Thus zero current imitation
weight neither erases earlier optimizer history nor removes action guidance.

Omitting the option retains legacy serialization and graph construction. Restore
uses the saved schedule and counter. The CLI permits only fresh-run schedule
overrides and checks native capability before environment construction.

All148 Rust CPU checks,804 packaged Python checks and strict workspace/binding
Clippy pass. The first focused Python run had five fixture failures because it
used ordinary guidance without corrective masks; the exact test and output are
retained. Changing that fixture to the existing corrective-label fixture gives
38 focused passes; production audit code did not change for that repair. The
packaged CPU suite records zero native construction attempts.

The guarded native fixture compares F64 weighted masked CE and derivatives at
weights100,50,0 with an independent quadratic auxiliary loss. All original loss
and gradient bounds pass; raw CE is exactly unchanged across weights and the
auxiliary gradient survives at zero. Three synthetic optimizer steps take.72s
(test duration); the guarded command takes1.485s. There are zero learner updates
or gameplay actions. All216 pins and the host guard pass. The scalar input has
the same `[1]` shape as Meganeura's scalar constant and scalar reductions.

The isolated runtime is `/x/Temp/kindle-expert-decay-20261101/runtime`, native
SHA`cac21ef6949f55c7ef3ff8d5a8a2d34eeb8c13f5106c217f2c289f2fae3b9028`.
The wheel, native test executable, CPU logs and build receipt are retained.

## Bounded Breakout check

The fresh check uses12M64/Auto/N8/B16/T16/H15/R16/expert100/encoder2e-5 with
compressed decay128→256. Guidance remains20% episode selection, capped at1920
actions, with corrective labels on all states. Seed1009 and the teacher are
retained. No prior experience is reused.

| Stage | Actions | Emulator frames | Learner updates | Measured time |
|---|---:|---:|---:|---:|
| Fresh training | 8,192 | 32,724 | 489 | 46.991s |
| Frozen greedy restore | 1,024 | 4,094 | 0 | .902s |

All489 finite reports have the exact scheduled weight, including100 at128,
50 at192 and zero at256 through489 (234 reports). Raw world/actor expert losses
agree within8e-7 at worst. All357 initial tensors exactly match the retained
control and47 pre-update semantic events match. CNN, reward, actor and value
parameters change during training. All357 final tensors remain invariant in
frozen restore, which retains the schedule and learner counter.

The full training pixel/action/label replay and both host guards pass. There are
3,889 actual overrides (47.4731%),4,303 passive labels and22 natural training
episodes. The short frozen arm completes one natural zero-return episode; it is
not a frozen competence assessment. The conservative early200k projection is
21m02s, below the30-minute gate, not a promise of full-run time or learning gain.

Service`kindle-breakout-expert-decay-check-20261101` succeeds02:22:36 UTC
November1, original PID728135/invocation`b71a50b6f3be4cc2b231086027ad8e70`.
Terminal collection completes02:22:48. All416 pins pass, including173 original
control pins without historical substitutions. No GPU stage repeats or bounded
service failure occurs. Native and runner differ from the retained447fe50 control;
matching initialization and unchanged teacher do not establish isolated causality.

Evidence root: `/x/Temp/kindle-expert-decay-20261101`. The bounded trial is under
`native-check/breakout-12m-64-encoder2e5-expert-decay-check`:

- Declaration SHA`e6bc847c40a544499614774de9eefbd3c892cd37621632e15028c5fd0069b019`.
- Review SHA`e7447e6fc54702bf6459fea8bec23387f12bfd4c70a885ca30be4d424bd9602a`.
- Completion SHA`066d52329425581447ddad9c03aff876d31e05ffc2dbd20e76b3ba8931aacee9`.

No full learner is selected by this qualification. A separate bounded comparison
would be needed to test whether later fading improves frozen gameplay per minute.
Do not reuse this check's experience or restart its completed stages. The service
is terminal, no GPU allocation remains active, and the five-game goal is unfinished.
