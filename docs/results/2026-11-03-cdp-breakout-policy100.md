# Stronger imagined-policy weighting passes execution and early cost checks

The [bounded policy100 check](../experiments/2026-11-03-cdp-breakout-policy100.md)
completes **8,192 actions/32,728 frames/489 updates in46.910s**. The1,024-action
frozen greedy arm takes.889s, with zero updates and no natural episodes. The
conservative200k projection is **1,259.778s (20m59.778s)**, below30min.
This establishes execution and early cost; gameplay benefit remains unmeasured.

Only imagined-policy loss weight1→100 changes. Score-function and entropy terms
scale together, while expert100/world imitation,imagined-value1/replay-value1,
actor/critic representation gradients off,12M64/Auto/N8/B16/T16/H15/R16 and all
other retained settings stay fixed. Native447fe50/SHA35ad0db and teacher remain
byte-identical. Private runner5a86e224 adds12 CLI lines. Current actor-gradient
dominance is not measured, and this does not predict an isolated mechanism.

All396 pins,357 control-matching initial/invariant frozen tensors,47 pre-update
semantic events,exact cadence,finite reports,changed CNN/reward/actor/value,
full guided pixel/action/label replay and host guards pass. Behavior total loss
reconstructs as `100*policy + value + replay_value + 100*expert` on all489 reports;
maximum absolute difference is3.06e-5, within the fixed tolerance. This verifies
reported weighting, not derivatives or optimizer correctness.

All489 reports have supervision, with117,049 repeated labeled replay rows and
the same total weight. Maximum raw world/actor expert-loss difference is8e-7.
Actual teacher overrides are3,733/8,192; online guided returns are not competence.
No bounded experience will transfer to a later learner.

Source `e5ebbaeebfb1d9021cf7841ad3696d7446785cc2`; original mainPID999583/
invocation17ae4bd7576f4022bcd23d308898587c succeeds **09:30:47.023 UTC November3**.
Original collectorPID999586/invocation85c31fea96c4446ea76d4923fe2fea17 succeeds
**09:30:52.508**. Independent terminal reader completes once **09:31:21.424**
in1.927s (1.944s wrapper), verifying original identities,artifacts and pins.
No bounded stage fails or repeats. Two preceding CPU arithmetic-fixture failures
and corrections remain preserved as detailed in the protocol; no native agent
was constructed in preparation.

Root `/x/Temp/kindle-breakout-policy100-20261103/native-check` is closed; do not
repeat qualification,learning,restore,replay,collection or terminal reading.
No allocation is active at bounded closure. A separately fixed
[fresh200k comparison](../experiments/2026-11-03-cdp-breakout-policy100-200k.md)
is now selected to measure final/actual-initial sampled and greedy gameplay
against70.750/143.667,all0/24 two-wall, with100k recovery and30min training cap.
The five-game goal remains unfinished.

Full CPU preparation passes **09:33:15.246 UTC November3** in2.406s (2.521s
wrapper):18 compiled scripts,five actual commands,38 config cases,19 startup
header cases,four observer-parser cases,14 recovery-body cases,six retention
cases,two collector-retention cases and ten command-validation cases. All396
bounded/338 control pins and357 shapes match. Actual bounded/full loss arithmetic
is identical; its qualified cases are retained without repetition. No emulator
or native agent is constructed, and no full preparation failure occurs.

| Artifact | SHA256 |
| --- | --- |
| Bounded declaration | `897f8faf5ca7655220dca54d0b29e2f3000b10e59f5922e8d568528caf284f2c` |
| Review | `0ebbce59bb9c476b8567a245dcdbfaa6580f9f0df24f5de8f035fadd2a30adc3` |
| Completion | `df88afd36854d25e222f3728c53bf52b1a34d2fef01cf8293f1f88000404189b` |
| Independent terminal summary | `a3a320fca7bb48a4dcd52f476e27e5566f784ea3a1f01b860aab6f3d992900ed` |
