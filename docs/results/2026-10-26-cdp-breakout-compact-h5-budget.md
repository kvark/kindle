# Compact H5 reaches the time cap without two-wall success

The fresh compact H5 recipe stops at the declared 45-minute cap after
**487,736 actions / 1,948,581 emulator frames / 121,873 updates**. A separately
declared frozen assessment scores **8.458 versus its actual initial 1.375**,
with **0/24 two-wall successes**. The 500k target is incomplete. Faster execution
and more experience produce only a modest score gain over the H15/200k control;
this does not justify an unchanged extension or establish task competence.

[Declaration and conditional cutoff assessment](../experiments/2026-10-26-cdp-breakout-compact-h5-budget.md).
Fresh1009/eval4000001009, centered CDP/action-effects1,4M/RGB64/Auto,
4,131,729 parameters, N8/B8/T16/H5/R32, full BPTT/microbatch8, replay100k,
actor_unimix.01. Full18/sticky.25/repeat4, native max-pooled frames and one GPU
resize; no external aid, RAM input or reward shaping. Only H15→5 changes in the
native config versus the compact mixture control, but the experience budget
also changes. All346 initial tensors match. This is a reused development seed.

## Cost and cutoff

At the shared196,608-action/49,091-update prefix, H5 takes1,086.206s (18m06s)
versus H15's1,325.662s (22m06s): **18.06% less wall time**, or1.220× throughput.
Native binary, GPU, model, resolution and initial tensors match. The horizon
and learned trajectories differ; this is a recipe-cost observation, not a
hardware speedup or an isolated horizon-learning result.

The cap includes construction. Recorded training takes2,697.229s (44m57s);
the service journal records45m01.488s including setup and shutdown. The worker
handles the stop, saves the exact cutoff checkpoint and exits0. The guard and
controller exit with a time-budget failure, which remains in the evidence.
The transient unit is then collected: a later `systemctl show` reports default
inactive/success fields, while the retained journal proves controller exit1.
Do not describe the original job as successful or the action target as met.

First update248, zero training debt,1,775 natural training episodes, final online
last50 mean9.32. The100k/200k/300k/400k recovery checkpoints pass before the exact
487,736-action checkpoint replaces them. All121,873 numeric reports, complete
source/configuration/counters and finite346-tensor checks pass. The cutoff audit
finds no recorded kernel-fault or validation-error event. No training resumes.

## Frozen assessment

| Policy | Training actions | Selected natural episodes | Mean score | Two-wall successes |
|---|---:|---:|---:|---:|
| H5 cutoff |487,736|24|8.458|0|
| Own actual initial |0|24|1.375|0|
| H15 mixture control |200,000|24|7.083|0|

The unchanged864-point two-wall/90%-success gate fails. The endpoints have
different experience budgets and do not isolate horizon effects. The earlier
retained12M H5 warm start also regressed, but belongs to a separate lineage.

The separate phase starts06:28:51 UTC after the cutoff audit and terminal
training service. It performs zero training actions or updates. Final/initial
frozen collection uses11,288/5,944 actions and45,101/23,717 emulator frames in
5.524s/2.992s. It collects31/29 natural episodes; the first three per lane remain
the selected24 per arm. Seven/five excess episodes and all partial tails stay in
the complete replays/videos. Zero truncations and all346 source/frozen tensors
unchanged in each arm; every source file hash also remains unchanged.

Both frozen host guards and full CPU replays pass; the frozen service exits.
The complete-budget postprocessor refused the missing successful training review
after the cutoff. Its error is retained; the separate comparison accepts the
explicit cutoff and completed frozen review. Action/frame/time curves and frozen
outcomes are independently reconciled and viewed, with the cutoff labelled.

## Evidence

Runtime root: `/x/Temp/kindle-cheap-20261025`. Training:
`breakout-4m-64-h5-r32-unimix001-500k`; frozen phase:
`breakout-4m-64-h5-r32-unimix001-500k-cutoff-frozen`.
Native`bfb8c31`, runner`1ed07af`, replay`e853818`; native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Training source at launch`8910efe`; frozen-phase source`bd652bc`.

- Cutoff review:`680e5cf47e0500e7e8444bfd641acbf440b06c64ba4120cdec7413726936d69d`.
- Frozen review:`8c174f12cce59866990f9d5f920d8c89e60b7dd3884b67b8dd2701e86215f4ef`.
- Final metadata:`4d3f976871231a152f17edc855c2565b660f665f711fbbf212c0a2f65653145c`.
- Comparison:`af7506d72131c83ddd9df7fd95b20219317b27bfe9a0a38f5f40edb3ed1ddb95`,
  under `horizon-budget-comparison/` in the training directory.

Retain the failed budget outcome and unchanged task gate. No extension, earlier
checkpoint selection, cap increase or broad seed campaign follows.
