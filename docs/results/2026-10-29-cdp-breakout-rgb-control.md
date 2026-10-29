# Breakout reconstruction passes the bounded execution and cost check

The existing native reconstruction model completes **8,192 aggregate actions /
32,726 emulator frames /489 updates in59.857s**. A1024-action/4096-frame greedy
frozen check takes.866s,with zero updates and no completed natural episodes.
The fixed conservative200k projection is **1607.487s (26m47s)**,below the
35-minute cap. This is execution and early-cost evidence,not task competence
or a guarantee of full-run cost.

All required execution checks pass after a separate CPU-only schema-name repair.
The original service remains recorded as failed at its CPU validator; both GPU
stages succeeded and neither was repeated. A full learner is separately selected
in the [200k protocol](../experiments/2026-10-29-cdp-breakout-rgb-control-200k.md);
no full run has launched at this report's bounded-check completion.

## Configuration and measured checks

Keep12M64/Auto/N8/B16/T16/H15/R16,full18/actor-unimix.01,action-effects1,
base/actor4e-5,encoder6e-6,dynamics4e-4,warmup1000,replay100k,seed1009 and
evaluation4000001009. Training uses20% episode guidance,cap1920,expert100,
all18 action weights1,world gradients and all-state corrective labels from the
unchanged edge-reset teacher with actual-action feedback. Native447fe50 and
the matching runtime remain unchanged; no build is required.

Relative to the retained CDP learner,only centered true→false,reconstruction
loss0→1 and future loss500→0 change. A copied experimental runner preserves
the split rates and exploration that stock RGB CLI would change. Native
max-pooled frames receive one GPU resize to64;sticky actions.25/repeat4 remain.

The model has14,457,556 trainable parameters and399 checkpoint tensors,
versus16,334,353/357 for CDP. All342 shared initial tensors match exactly.
All399 initial/final tensors are finite,and399 remain unchanged after frozen
evaluation. Positive reconstruction/zero future loss,489 finite supervised
reports,changed CNN/reward/decoder/actor/value,source/config/counters,full
training pixel/action/corrective-label replay and both host guards pass.
The first update is376,with half an update of residual credit at8192 actions.

There are3898 overrides (47.583%),4294 passive labels and3976 actual/teacher
disagreements. All489 reports contain supervision;117,064 labeled replay rows
equal their weight sum. Maximum raw world/actor expert-loss difference is8e-7.
These observations do not establish useful unaided gameplay.

## Original CPU failure and separate repair

Service`kindle-breakout-rgb-control-check-20261029` starts17:55:59 UTC October29,
source`2261a7a`,PID324293/invocation`039efed1d74040a887b71166978f0e70`.
Training completes17:57:04 and frozen evaluation17:57:08. Both guards pass.
The service then fails17:57:08 in the CPU checkpoint-schema comparison,after
1m09.841s total wall time.

The generated schema incorrectly prefixed38 decoder optimizer tensor names
with`world.adam_m.`/`world.adam_v.` instead of`adam_m.`/`adam_v.`. Shapes and
parameter counts were correct. The earlier CPU preflight checked shared shapes
and aggregate counts but did not catch this naming error; its original receipt,
schema and controller remain unchanged.

A separate`cpu-repair` directory corrects only those38 names. It verifies the
canonical decoder names against the already completed Qbert RGB schema,the
moment shapes against their parameters,and all399 actual initial tensor shapes
and finiteness. A native-import guard wraps the original audit with the corrected
schema helper. The complete CPU audit/replay takes7.138s and succeeds17:59:54;
terminal collection succeeds18:01:33. All156 original inputs and176 repair pins
remain unchanged. The original failed service/journal are retained; no native
stage,initialization,training or frozen evaluation repeats.

Use the corrected schema/helper in`native-check/cpu-repair` for the separately
selected full comparison. Do not copy the original erroneous schema. GPU and
host recovery were unnecessary. No allocation remains active at bounded-check
completion.

[Fixed bounded protocol](../experiments/2026-10-29-cdp-breakout-rgb-control.md).
Evidence root:`/x/Temp/kindle-breakout-rgb-control-20261029/native-check`,trial
`breakout-rgb12m-64-b16-r16-expert100-check`. Logs,weights,repair evidence and
runtime artifacts remain outside Git.

- Original declaration: `6b66ca1b8af25b0b84e0b0bec6b6108e27be022246008abd076e2a496d156254`.
- Completed execution review: `52bd0cc559a0350b6b2746253d0700d834c1f3fb8e95b0809a5079b6c5fc4a36`.
- CPU repair: `a8be9c8492e558258db5f6396ac7c2d543733d9f19afba9fc6c4e3baf7193dbb`.
- Completion: `bd43647a5445ccda18e9634f9f5c3ec01a1bc1462a3ac087d5f1537945d565f1`.
