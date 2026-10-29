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


## Full-controller CPU preflight

The fresh200k controller passes CPU preflight at18:13:34 UTC October29 in
1.188s:five actual commands parse,22 actual audit/comparison config
cases,six header-readiness cases,four observer parser cases,five recovery
cases,40 resolved paths and14 compiled scripts pass. All156 bounded inputs,
176 CPU-repair inputs and132 retained CDP inputs remain unchanged. The copied
corrected schema validates all399 finite bounded initial tensors and342 shared
initial values against CDP;100k/200k schedules are6227/12477 updates. Native
construction is prohibited and no game is created. The prior failed bounded
service remains failed;the successful CPU repair is checked separately.

Receipt:`full/full-preflight.json`,SHA256`42c423d80d90ecd5519390ac57c2b6e67c4ebea4337fe7f41ee7231276cca63f`.
The fixed protocol,controllers,corrected schema,original failure and repair
evidence are pinned before launch. The selected training cap remains35min;
this preflight is execution evidence,not a task outcome.


## Fresh200k comparison active

The selected fresh life starts18:14:45 UTC October29/source`531ce35`,main
PID324938/invocation`23da92f687b74243812a8007db61ac5a`,under
`kindle-breakout-rgb-control-200k-20261029`. Its212 immutable inputs include
the corrected schema and prior failure/repair. Training has a35min cap and
100k recovery;four final/actual-initial sampled/greedy arms,full replays,
curves,exposure and comparison follow automatically. No check/control
experience is reused. The main and three observers are active.

Startup18:15:01 passes:399 finite initial tensors,399 matching bounded RGB
initial values,342 shared CDP initial values,expected updates at376/392/408/
424/440/456,1800 observed actions and90 finite supervised reports with positive
reconstruction/zero future loss. All212 pins and original process identity
match. These are startup checks,not frozen task results;training and audits
remain unfinished. No extension,second seed or successor allocation selected.

Root:`/x/Temp/kindle-breakout-rgb-control-20261029/full`,trial
`breakout-rgb12m-64-b16-r16-expert100-200k-seed1009`.
Declaration:`cf3a4f45cd1ad497ce48d5acbae8314310c0bc630ac85bb567aab2959d1c8a45`.
Launch:`d0b4787d58888f3daea9a46468fa834e965ba3ae4f2e137ff7ae9dd7c499b303`.
Startup:`eda45ab6a2585596f4fdcb2c0ec74215449ab8a2572ca17efe4b9f1a77f48bb8`.

The18:21:37 UTC CPU-only completed-life expert-fit diagnostic co-hosts for2.402s
on one low-priority CPU core. All212 active pins and original PID/invocation
remain unchanged;no native agent/game or GPU work is created. Retain this as
a timing caveat without wall-time correction. It reads the completed CDP
Breakout and Qbert lives,not the active RGB loss trajectory. See the
[diagnostic](2026-10-29-cdp-learning-diagnosis.md#breakout-still-has-substantial-training-imitation-error).


The100k recovery audit succeeds18:28:50 UTC October29:100,000 actions/6227
updates and399 finite tensors. Its complete log prefix contains102,312 actions
and6371 finite reports. All212 input pins remain unchanged and the original
main PID324938/invocation`23da92f687b74243812a8007db61ac5a` stays active.
The recovery observer exits successfully;main/progress/post continue. This is
checkpoint integrity evidence,not frozen competence. Recovery-review SHA256
`488f39780a830572131f4644d690e9203c1df4e4f01c9fde446b39b7e9de3913`.


## Automated terminal collection: CPU observer correction

A read-only terminal collector is added while the original learner continues.
Its first service fails18:35:31 UTC because systemd unloads a completed
successful recovery observer and clears its live`InvocationID`. The observer
incorrectly required that field to remain populated. No training,evaluation,
checkpoint or main-service input is changed;no GPU stage restarts.

The original script,declaration,failed service and journal remain at
`/x/Temp/kindle-breakout-rgb-control-20261029/terminal-collection`. A separate
`cpu-repair` checks active PID/invocation as before and validates terminal
success against the original invocation preserved in journal records. Eight
cases cover active/terminal/deactivating states,wrong PID/identity,multiple
journal identities and failures;all four actual service states also pass.

Service`kindle-breakout-rgb-control-collection-repair-20261029` starts18:37:31
UTC/source`cb09d72`,PID327156/invocation`94465c790a5448fe81cb000c83adc7df`.
It is verified live;all11 collection pins and212 learner inputs remain unchanged,
as does learner PID324938/invocation`23da92f687b74243812a8007db61ac5a`. It waits
for the four original services to succeed,then invokes the existing pinned
CPU collector once. An observation timeout never starts/stops/restarts a
service or changes an allocation cap. Do not manually repeat the exclusive
collector while this observer is pending or after completion exists.

Repair declaration SHA256`85fb7bdbcd43478fe7e1a6d7361b42f339089d4ce0c79703cd0d8fc1b28e3a79`;
preflight`9d6be67e223ddab91b811f048f46fb1d8bb6cadae1c3879ba98b5ad283873767`.
