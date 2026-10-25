# T64 fits narrowly, but misses the efficiency gate

**Keep T16.** T64 passes the production/accounting/frozen-restore checks but
reduces cost per replay position by only4.60%, below the predeclared10% gate.
Its minimum sampled Vulkan estimated headroom is2.05GiB, just56MiB above the
2GiB requirement. Do not turn this into a sequence-length sweep or learning
campaign. This says nothing about whether longer BPTT could improve policy
quality; the short preflight does not test that hypothesis.

[Declaration](../experiments/2026-10-09-cdp-sequence-preflight.md) ·
[Compact data](2026-10-09-cdp-sequence-preflight.json) ·
[Raw audit](../../runs/cdp-sequence-preflight-20261009.EBJhphnr/audit.json).
The service finishes October9 at16:54:08 UTC in3m46s; CPU audit passes in1.45s.
No worker remains. All five DreamerV3 quality targets remain open.

## Equal trained-position window

Same qualified native7311547d/Meganeura31026833/Blade56f0565 and12M centered
CDP/N8/B8/R32/H15/action-effects coefficient1. Only sequence/full-BPTT length
changes. Each arm gets4,096 actual Breakout actions, seed103. Compare every
update in the fixed last2,048 actions, after both replay warmups:

| Measurement | T16 | T64 |
| --- | ---: | ---: |
| Measured updates |512 |128 |
| Replay positions |65,536 |65,536 |
| Mean full update |109.82ms |419.06ms |
| Cost per replay position |0.8580ms |0.8185ms |
| Imagination / update |60.81ms |240.34ms |
| World training / update |32.93ms |121.95ms |
| Minimum sampled estimated headroom |11.36GiB |2.05GiB |

Imagination cost per position barely falls (about1.2%); world-training cost
falls about7.4%. This is not enough to justify the memory pressure on an
efficiency argument. B512 imagination uses the existing large-batch RSSM
implementation; no grouped threshold, precision policy, model capacity or
exploration change is bundled into the test. Per-dispatch instrumentation and
concurrent heavy work are absent. These are one-pair production measurements,
not measured SM utilization or a universal throughput result.

Whole training-loop time is108.22s versus83.29s, but the latter processes fewer
total replay positions because it warms up later:963 versus193 updates,
123,264 versus98,816 positions. Its apparent whole-run gain is not the
steady-work gain. Final training debt is0 versus.5, exactly reconciled by the
configuration-aware replay ledger, not silently discarded or forced to zero.
Initialization and frozen checks add to the service wall time.

## Integrity and limits

All four guards pass with zero new warnings. Initial346 tensors match exactly
between arms; both1,024-action frozen restores make zero updates and preserve
all346 saved tensors. Finite metrics/checkpoints, expected device, per-stream
actions/rewards/resets/replay and source/native/config identities pass. Total
8,192 training actions/1,156 updates plus2,048 frozen actions. Every episode
and unfinished tail remains. No game score is presented as competence evidence.

The old convenience learning reporter assumes one update per four actions.
It is not used here: the independent configuration-aware vector auditor
reconstructs actual training credit for both sequence lengths. No production
code, checkpoint migration, new learner budget or historical repin is needed.

Next inspect imagination's cost on the retained12M checkpoint with one bounded
diagnostic capture, not another optimizer sweep. Timestamp instrumentation
perturbs timings; use it for attribution, never substitute its times for the
ordinary measurements above. Do not automatically retry the rejected projection
rewrite, relax numerical gates or postpone learning indefinitely on optimization.
