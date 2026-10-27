# Qbert expert imitation: more experience at lower replay ratio

Select one fresh **500k-action/R16** learner on reused seed4013 after its
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
