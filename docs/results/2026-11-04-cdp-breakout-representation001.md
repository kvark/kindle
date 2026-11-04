# Lower representation KL weight passes bounded execution and cost checks

Reducing the existing representation KL weight from0.1 to0.01 completes
**8,192 actions /32,719 emulator frames /489 updates in47.300s**. The retained
bounded control took47.690s. The conservative200k projection is **21m10.262s**,
within the30-minute gate. This establishes execution and early cost; it does
not establish better gameplay or a speed difference. No full learner is selected.
The original Breakout864/90%-two-wall gate and five-game goal remain unfinished.

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-representation001.md)
changes only `loss_scales.representation`. Dynamics KL1, both free-nat floors1,
native447fe50/SHA35ad0db and the corrective runtime remain unchanged. Keep
CDP12M64/Auto/N8/B16/T16/H15/R16, replay-value1/expert100, base/encoder4e-5,
dynamics4e-4, behavior8e-5, sampled collection/posteriors, actor-unimix.01,
20% guided episodes/cap1920 and all-state corrective labels. All optional
prior-continuation, terminal-weight and multi-step losses stay off.

The hypothesis is that weaker posterior regularization might help representation
and control. Retained full-run raw KL means6.215/11.217/11.457/11.815 over successive
50k windows show an active aggregate penalty, while older two-batch gradient
measurements show opposition to imitation in some groups. These observations
do not identify a causal limitation. The new weight adds no parameters or
computation, and can change learned states, optimization and later experience.

## Execution and early fitting

All706 current and338 retained pins pass, resolving the exact historical replay
helper bytes. All357 initial tensors match the retained learner and all47
pre-update semantic events agree. First update376, final489 and residual credit.5
match the fixed cadence. Every report is finite; independent weighted-world-total
recount has maximum absolute error.000106542. All inspected CNN, reward,
continuation hidden/output, actor and value parameter entries change.

Frozen restore completes1,024 greedy actions /4,096 frames in.872s with zero
updates,357 invariant tensors and no completed natural episodes. Full training
pixel/action/label replay and host guards pass. Actual teacher overrides are
3,906/8,192 actions (47.6807%); guided online scores are not frozen competence.

| Unweighted learner-report mean | Retained weight0.1 | Candidate weight0.01 |
|---|---:|---:|
|All489 raw KL|2.687052|3.941760|
|All489 floored representation KL|2.699389|3.943247|
|All489 expert cross entropy|2.124006|2.122716|
|Last100 raw KL|3.453213|3.998163|
|Last100 expert cross entropy|1.416821|1.415500|

These are each learner's changing replay, not a fixed held-out comparison.
The loss summaries neither establish gameplay benefit nor explain a mechanism.
Construction takes2.466s separately; learning-wrapper time is40.577s of47.300s,
acting2.483s and environment2.531s. These timers do not isolate GPU kernels.

## Qualification, failures and closure

The runner exposes an optional positive finite F32 override for fresh training;
omission preserves defaults and restore rejects overrides. Both canonical and
private qualified runners add exactly ten lines. No native build occurs.
CPU checks cover default/explicit/reduced configurations, invalid values,
restore/evaluation rejection, actual queued commands, loss arithmetic, cadence,
label/guidance bounds, replay argv, checkpoint corruption and terminal collection
before native-agent or emulator construction.

The original canonical test run records34 passes and one failure: a new negative
fixture incorrectly spells greedy evaluation `--evaluate-greedy`. Its exact
source and exit1 receipt remain preserved. Only that fixture changes to the
supported `--evaluate --greedy`; the failed case then passes, and the first
packaged run passes all35 targeted cases. Production source is unchanged by
the repair. Preflight passes in1.539s. The completed GPU and audit stages have
no failures or repeats; the earlier CPU fixture failure is not erased.

Source`ae529140`, mainPID1152316/invocation
`2e74adbf10b54bd9ba6fcc6691f39bd7` starts09:02:55 UTC November4. Main succeeds
**09:04:06.215**, original collector succeeds **09:04:18.505**, and independent
6.424s CPU closure finishes **09:04:41.233**, verifying89 saved artifacts.
A separate.025s saved-report summary completes09:05:13.803. Both services are
terminal success. No GPU allocation remains active; check live state before work.

Evidence root: `/x/Temp/kindle-breakout-representation001-20261104`.
Declaration SHA256`b3e0c169c1ae88e5e2d5edf0cb9f27a697affbf1adda52bf98109d73f60087c6`;
completion`67b7262d6acf9ad29dcf66d96d00201c4930ab607454bf55762b579797b99353`;
terminal summary`5b392230c202ecd3c9c3d69d4aeff8a6c2f360f4c25f8f2f4e3c47e00f07a69d`.
Original failure, repair, historical-source snapshots, checks, checkpoints,
replay, guards and closure receipts remain retained. Do not repeat completed
stages or transfer this bounded experience. Any full learner needs separate
selection,100k recovery and final/actual-initial sampled/greedy24-episode
assessments against retained143.667/70.750. No extension, sweep or automatic
successor is selected.
