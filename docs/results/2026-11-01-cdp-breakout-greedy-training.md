# Breakout greedy collection passes its execution and early-cost check

The [bounded check](../experiments/2026-11-01-cdp-breakout-greedy-training.md)
completes8192 actions /32,740 frames /489 updates in46.633s. Frozen1024 actions /
4076 frames take.874s withzero updates;its eight natural episodes all scorezero.
This establishes execution and early cost,not competence. The conservative
200k projection is1252.339s (20m52s),below the30-minute gate.

Only live action collection changes sampled to greedy in the retained12M64/
Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100 recipe. Actor unimix.01,all-state
corrective labels,world imitation gradients,20% episode guidance/cap1920,teacher,
losses,posterior and imagined sampling remain unchanged. Native447fe50 and exact
qualified runneraba4300d are reused without a build. Seed1009/evaluation4000001009
are development inputs,not independent confirmation.

All252 pins,357 matched initial and invariant frozen tensors,489 finite reports,
correct first update376/residual.5,changed CNN/reward/actor/value,full training
pixel/action/teacher-label replay and host guards pass. All47 pre-update semantic
events deliberately differ from sampled control. Training executes3079 overrides
(37.5854%);guided returns are not unaided competence. The original service succeeds
07:51:36 UTC November1 and terminal collection07:51:45,sourceccb4387.
OriginalPID765731/invocation`706343652a1e4e1aac60099e83b198e8`. No original stage
fails or repeats. The prior.648s CPU preflight is retained without repetition.

Root `/x/Temp/kindle-breakout-greedy-training-20261101/native-check`,trial
`breakout-12m-64-b16-r16-encoder4e5-greedy-expert100-check`. All services are
terminal at closure. Do not rerun this check or collector. No check experience
may be reused. Full gameplay benefit remains unmeasured;any full comparison
requires a separate selection and declaration.

| Artifact | SHA256 |
|---|---|
| Declaration | `5469d5903091f73d2c9276042364db7a265cd33f84c534f06496b81a6373fc3b` |
| Review | `191f7fa6163afe9d667350ded1408280de04af22765da5526f657eb4fe951d9b` |
| Guided replay | `ffe00013c78a9d05d9f26b63b809d58a06fc886b40b911fc01bd776a0b66bce4` |
| Completion | `7c353908b4526a69dddadd812782b57197f8add64863ed3e17192b69b421967c` |
