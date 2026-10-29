# Fresh200k Breakout reconstruction comparison

After the [bounded check](../results/2026-10-29-cdp-breakout-rgb-control.md)
passes execution and its fixed26m47s conservative projection,select one fresh
200k-action reconstruction learner. Its original CPU schema failure remains
recorded;the separate CPU repair completes all checks without repeating GPU
work. Reuse the corrected schema/helper,not the original erroneous names.

Keep the exact checked12M64/Auto/N8/B16/T16/H15/R16/expert100 configuration,
full18/actor-unimix.01,all rates,warmup,teacher,20% episode overrides/cap1920,
all18 label weights1,world gradients,all-state corrective labels,native447fe50
and copied matched-RGB runner. Native max-pooled frames receive one GPU resize;
eight independent lanes,sticky.25/repeat4 remain. Only centered true→false,
reconstruction0→1 and future prediction500→0 differ from the retained CDP
control. No native build,additional CDP learner or further cost check.

Use seed1009/evaluation4000001009,200,000 aggregate actions/12,477 updates,
first update376,half-update residual credit,with a **2100s/35-minute training
cap**. Save100k recovery at6227 updates. Start fresh: no check/control weights,
experience,replay,belief,games or RNG are restored. The completed CDP reference
is22m19s training,sampled13.208/own initial1.417 andgreedy17.875/2.208,bothzero
of24 two-wall successes. It is evidence only,not a source of training state.

Require399 finite initial/final tensors,399 identical initial tensors to the
bounded RGB check,and342 matching shared initial tensors with CDP. Compare
complete own-initial semantic traces with CDP,excluding source/config/timing
identity. Audit exact config/source/counters,seeds,positive reconstruction and
zero future loss,finite reports,changed CNN/reward/decoder/actor/value,aligned
expert losses/weights,full training pixel/action/corrective-label replay and
host guards. Pin original failed-service evidence and the successful CPU repair.

Evaluate fixed final and own actual initial under sampled and greedy actions,
sampled primary as in the retained control protocol. Each arm selects first3
natural episodes per8 lanes=24,with600k-action/300s caps,guidance and labels off,
zero updates/truncations and399 invariant tensors. Retain all excess episodes
and partial tails; full outcome replay/video checks,curves,supervision exposure
and the paired comparison are required. Frozen source logs have no live pixel
hashes. Keep the original90%-two-wall gate (864 points);score improvements alone
are not a task-gate pass.

Caps:training2100s,each frozen arm300s,training replay900s,CPU audit1500s,
whole main service5100s,postprocessing5400s. Progress/recovery observers cannot
restart training. Stop on actual GPU faults/numerical failures;no driver
recovery,retry,cap increase,extension,checkpoint or inference-mode selection,
second seed or automatic successor. A cutoff is incomplete evidence.

Report sampled and greedy outcomes,own-initial gains,actual guidance fraction,
aggregate actions,emulator frames and wall costs separately. This changes the
objective and head geometry,not parameter count in isolation. Seed1009/cohort
and the known CDP result are reused development evidence. A favorable result
would not identify missing ball information as its cause; a negative result
would not rule out all reconstruction objectives. No broad reliability,CDP
superiority or five-game completion claim follows from this comparison.

Root`/x/Temp/kindle-breakout-rgb-control-20261029/full`,trial
`breakout-rgb12m-64-b16-r16-expert100-200k-seed1009`,service
`kindle-breakout-rgb-control-200k-20261029`. CPU controller preparation and
preflight precede any declaration/launch. Check actual live state first.

CPU preflight completes18:13:34 UTC October29 in1.188s. Five actual
commands,22 config cases,header/observer/recovery checks,40 resolved paths,
14 scripts,399 finite bounded initial tensors and342 shared initial values pass.
All156 original/176 repair/132 CDP pins remain unchanged;no agent/game created.
Corrected schema SHA256`4c9ea7443dfa36ed87db7b45b4005ee2e6b9c1550b1f7734a648db62f72b3f3a`.
Full preflight SHA256`42c423d80d90ecd5519390ac57c2b6e67c4ebea4337fe7f41ee7231276cca63f`.
Declaration and launch follow the fixed protocol;the earlier bounded failure
remains recorded and no GPU stage is repeated.

Selected declaration and launch complete18:14:45 UTC October29/source531ce35.
MainPID324938/invocation`23da92f687b74243812a8007db61ac5a`,212 pins;startup
18:15:01 passes399 finite/bounded-matching and342 shared initial tensors,
firstsix updates and90 finite reports. All services are active;training and
final audits remain pending. See the linked result for exact receipts.

The18:21:37 UTC CPU-only completed-life expert-fit diagnostic co-hosts for2.402s
on one low-priority CPU core. All212 active pins and original PID/invocation
remain unchanged;no native agent/game or GPU work is created. Retain this as
a timing caveat without wall-time correction. It reads the completed CDP
Breakout and Qbert lives,not the active RGB loss trajectory. See the
[diagnostic](../results/2026-10-29-cdp-learning-diagnosis.md#breakout-still-has-substantial-training-imitation-error).


The100k recovery audit succeeds18:28:50 UTC October29:100,000 actions/6227
updates and399 finite tensors. Its complete log prefix contains102,312 actions
and6371 finite reports. All212 input pins remain unchanged and the original
main PID324938/invocation`23da92f687b74243812a8007db61ac5a` stays active.
The recovery observer exits successfully;main/progress/post continue. This is
checkpoint integrity evidence,not frozen competence. Recovery-review SHA256
`488f39780a830572131f4644d690e9203c1df4e4f01c9fde446b39b7e9de3913`.
