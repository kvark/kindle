# Fresh Qbert200k with reduced wait-label weight

The [bounded check](../results/2026-10-29-cdp-qbert-noop-weight.md) passes all45
pins,357 matching initial/invariant frozen tensors,weighted loss accounting,
full guided replay and host guards.8192 actions/489 updates take70.160s; the
linear200k projection28m33s becomes31m24s with a10% margin, below35min. This is
execution/cost evidence, not task competence. The prior CPU import assertion
failure is preserved and corrected before GPU launch; no GPU stage repeated.

Separately select one fresh200k-action/12,477-update learner. Change only the
NOOP expert-label weight1→.25 from the retained strongest12M control. Keep
12M48/Auto/N8/B16/T16/H15/R16/centeredCDP coefficient1,actor-unimix0/full18,
original rates,20%episode continuation guidance/cap1920/expert100/world gradients/
all-state corrective labels,seed4013/evaluation4000004013. Other17 action weights
remain1. Native447fe50, the copied runner and Qbert teacher remain identical;
use `/x/Temp/kindle-breakout-corrective-20261028/runtime`. Frames retain one GPU
resize. No bounded/control weights, replay, belief, games, RNG or experience
are reused. This is a reused development seed/cohort, not independent confirmation.

The loss averages all replay rows, not the sum of weights. Reducing wait weight
changes relative class pressure and overall expert magnitude in actor and world
training together. It is not a pure class-balance change at fixed total strength.
The control's late-training passive post-pyramid wait/hop agreement95.044%/55.968%
motivates this test but does not establish a causal bottleneck or teacher accuracy.

The primary comparison is fixed-final greedy mean against retained11984.375/24
first pyramids in30m27s. Keep sampled final6759.375/24 as a separate secondary
comparison; do not switch primary mode based on outcomes. Evaluate candidate
final and actual-initial checkpoints with both sampled and greedy actions,
sampled posterior throughout. Each arm selects first3 natural episodes per8
lanes=24, with guidance/labels off, zero updates/truncations and retained partial
tails/excess episodes. Original mean15000/90%-first-pyramid gate remains fixed.
Compare both mean and success fraction, training wall time, actions and emulator
frames. Online teacher-guided scores are not frozen competence.

Save and audit100k recovery at6227 updates; final200k has12,477 updates, first
update376/residual credit.5. Require357 matching control/bounded initial tensors,
357 finite/invariant frozen tensors per arm, paired actual-initial semantic
traces, exact config/source/counters, finite reports, changed CNN/reward/actor/
value, aligned raw actor/world expert losses and valid weighted sums for every
report. The implied wait/hop row counts must be nonnegative integers with
`.25 * wait_rows + hop_rows == weight_sum`; they are inferred from reported
sums because replay indices are not independently logged. Require complete
training label/input-pixel/action replay, frozen outcome replays/videos, host
guards, curves, coverage, actual guidance exposure and completed comparison.
Ordinary frozen logs lack live pixel hashes; do not claim direct live-image
identity. Preserve original declarations and resolve historical input paths
through their existing verified archive map.

Caps:training2100s, each frozen arm300s/600k actions, training replay1200s,
CPU audit1800s, main service5400s and postprocessing6000s. Automatic read-only
progress and100k recovery observers use the exact parsed commands and fixed
deadlines. Observer timeout does not stop or restart training. Stop on real
faults/numerical failure; no driver recovery, retry, cap increase, extension,
checkpoint selection or automatic successor. A favorable result would need
separate seed confirmation; this run alone cannot complete the five-game goal.

Root `/x/Temp/kindle-qbert-noop-weight-20261029/full`, trial
`qbert-12m-48-b16-r16-expert100-noop025-200k-seed4013`, service
`kindle-qbert-noop025-200k-20261029`. Check live state before launch. The bounded
service is terminal success and the host is idle at selection.

## Launched and startup checked

Main service starts10:36:00 UTC October29/source`bcaf51a`,PID283869/invocation
`0522d3f2958a4f47880218d10aa18cac`. All230 declared inputs remain unchanged.
Startup verifies the exact selected config,357 finite initial tensors matching
both retained control and bounded check,and the first six updates at376/392/
408/424/440/456 actions. The inspected2376-action prefix contains125 finite,
supervised reports. No frozen result is available yet. Progress,100k recovery
and terminal postprocessing observers are all active,with parsed commands and
fixed deadlines. No preparation failure or GPU repeat occurs in this full run.

The five CPU command parses,14 compiled scripts,11 exact-config boundary checks,
eight header-readiness checks,three observer CLI checks and357 initial tensor
matches pass in.592s without native/environment construction. All45 bounded
pins and190 historically resolved control inputs verify. The original bounded
CPU import assertion failure remains separate; no completed check is repeated.

Declaration`bfb3e2864f87e5be79c20e21f5ea6278db76c0251483fcf12866ee60f9cc0c74`.
The copied`full/fixed-protocol.md` and all executable inputs remain pinned; this
status paragraph is outside that immutable declaration. Check live service state
before any additional GPU work.

## Midpoint recovery verified

The100k checkpoint audit passes at10:51:04 UTC October29,with6227 updates,357
finite tensors and matching config/metadata/tensor hashes. Its stable-file check
passes;the complete100,440-action log prefix contains6254 finite reports. The
recovery observer exits success without stopping training. The main learner
keeps PID283869/invocation0522d3f2958a4f47880218d10aa18cac and advances beyond
104k actions. All230 declared inputs still match;progress and postprocessing
observers remain active. This verifies recoverable weights,not frozen competence.

Recovery review`28cda2bfc94c4918de75f82898a6d1bf0a68ea4c9bf7ab305209c1153d37bcb1`;
checkpoint metadata`4ce18c54a9b9683b3978892ca0f7de3c8f32914b1d12ebecf4b52ee3a7d7a2c6`.
The200k budget,35-minute training cap and fixed-final assessments remain unchanged.

## Training and frozen execution complete; audits active

Training completes200,000 actions/799,700 emulator frames/12,477 updates in
1812.098s (30m12s),within2100s. Its guard passes at11:06:17 UTC October29. All
four frozen arms finish by11:07:29,with zero updates and passing guards:sampled
final30,664 actions/122,613 frames/18.254s,initial8728/34,881/5.324s;greedy
final43,280/173,074/26.175s,initial8432/33,690/5.090s. Total91,104 actions/
364,258 frames/54.843s. All230 declared inputs still match.

The same main service/PID283869 remains active with CPU audit child287674.
Checkpoint/trajectory replay audits and final postprocessing remain pending;
no completed comparison or gate claim is made yet. Progress/postprocessing
services remain active;the100k recovery observer already succeeded. No GPU
stage repeats or caps change. A separate read-only status inspection briefly
encounters an empty newly created greedy log;its failure/correction is recorded
in`monitor-tail-empty-file.json`,with no worker or pinned observer change.

## Completed comparison retains weight1

All audits and the terminal collector pass. Greedy9756.250/24 pyramids and
sampled6120.833/22 are below retained11984.375/24 and6759.375/24;own initials
80.208/177.083 clear none. Both candidate means miss15000. All230 pins,100k
recovery,357 initial/frozen tensors,paired initial traces,full replays,guards,
curves,coverage,exposure and comparison pass. Main exits11:18:14 UTC October29,
postprocessing11:18:33,all four services terminal success. Retain weight1;
no extension,confirmation,weight sweep or automatic successor is selected.

The separate1.000s CPU phase recount has10 unchanged inputs. On last50k passive
post-pyramid states,wait agreement95.044%→86.137% and hop55.968%→58.100%;before
that phase,both fall. These changing training states are descriptive,not causal
or held-out accuracy evidence. No GPU work or checkpoint changes occur.
[Completed result and receipts](../results/2026-10-29-cdp-qbert-noop-weight.md).
