# Qbert continuation guidance: bounded check and fresh learning comparison

Select one bounded native check after the Breakout expert comparison completes
with all audits and a negative frozen result4.000 versus control7.083. Do not
extend Breakout. The stronger CPU Qbert reference separately scores22050.000
with24/24 first pyramids and exact input/output/reset pixel/action/outcome replay.
It is a hand-coded reference on reused development episodes, not native learning.

Apply the exact three-file optional continuation patch after terminal Breakout
reviews and all46 preceding pins are verified.456 CPU tests pass with native
construction blocked;10 unchanged native-constructor validations excluded.
The original first-pyramid policy AST is unchanged. All24,733 saved first8
reference actions and pixels match the staged implementation. Native79c9a7f
is unchanged; retain previous runtimes for their historical logs/checkpoints.

The new `--qbert-guidance-policy continuation` is fresh/Qbert/full18/native-RGB
only and requires positive episode probability plus an explicit action cap.
It repeats a newly chosen hop briefly while its origin remains visible, avoids
nearby purple pixels when choosing a new hop, and uses stable score-digit/board
colors to continue. Inputs are native RGB and the rule's own history. No RAM,
rewards or task labels enter the policy. Its guidance kind and implementation
hash distinguish it in training/replay logs. The default remains first-pyramid.

## Selected check

- Centered CDP4M/RGB48/Auto/N8/B8/T16/H15/R32, coefficient1, actor mixture0,
  original rates, full18/sticky.25/repeat4/noops0; one GPU resize from native RGB.
- Fresh4013/eval4000004013,8192 aggregate actions /1987 updates, first248,
  residual0; training180s, frozen1024 actions/60s, CPU audit180s, service480s.
- Select25% episodes/cap640, expert scale1, NOOP weight.05 and other weights1.
  This is an execution-check schedule, not a full learning recipe.8192 actions
  exercise guidance continuation and release within the check.
- Require3,578,225 parameters, all finite reports/346 tensors, all initial tensors
  matching retained same-seed expert control, changed CNN/reward/actor/value,
  movement labels/nonzero expert loss,346 unchanged frozen tensors, complete
  guided input/action/outcome replay and passing host guards.

Artifacts:`/x/Temp/kindle-qbert-guidance-20261027`.
Trial`qbert-4m-48-continuation-check`, owner`kindle-qbert-continuation-check-20261027`.
Preparation`prepare_check.py`, controller`run_check.py`, isolated`runtime`.
Control:`/x/Temp/kindle-expert-20261026/qbert-4m-48-r32-expert1-cap640-200k-seed4013`.
The declaration pins exact source/runtime/scripts, control and preceding review.
DeviceRX7900XT0x744c/RADV, boota8648671/kernel7.0.0-38/Mesa26.0.8; existing host
guard. No native rebuild, retry, cap increase, extension, reuse of check
experience or automatic full-run promotion. A pass establishes execution only;
no full Qbert learner is selected by this declaration.

Patch SHA256`aeb61a4510b9122ae0fbf4ee8777eb86e15d2d082ebc7ee7dccbe2a3351ebfbc`;
validation`82b534c911aa4499cafd14f8d50d146f22d8e637ffd5f90145b3405a25bd4563`;
parity`2f2d4d44f9b90412fad4234c5ee9531f790d9e857f32af33e63fa357db3c29dc`.


## Completed bounded result

Service starts03:21:37 UTC/source3f3f7b5, PID80351,
invocationc1b17474007e431e8f9482bbec2357b2; exits successfully03:23:09.
8192 actions /32,739 frames /1987 updates take70.493s. All finite reports,
346 matched initial tensors, changed CNN/reward/actor/value,32 declared pins,
2987 trusted actions (687 hops),1968 nonzero expert reports, full guided replay
and guards pass. Frozen1024 /4096 frames take.531s with zero updates and346
unchanged tensors. No natural frozen episode completes. Review SHA256
`6f91901f01eb2101735282f6c2fb2a4bcff754863de1160edb3c29e393b5943c`.
No full learner or automatic successor is selected. See [evidence and limits](../results/2026-10-25-cdp-cheap-qbert.md#native-continuation-check-passes).


## Selected fresh 200k comparison

After the bounded check passes, select one fresh life with the same centered
CDP4M/RGB48/Auto/N8/B8/T16/H15/R32, coefficient1, actor mixture0, original
learning rates, full18/sticky.25/repeat4/noops0 and native79c9a7f. Seed4013 and
evaluation4000004013 are reused development seeds, selected after the stronger
old-teacher result. Restore no weights or experience from the check or controls.

Change the teacher from first-pyramid to continuation and the guidance cap from
640 to1920 actions. Keep5% episode selection, expert scale1, NOOP weight.05 and
other action weights1. These two changes are tested together. On the24 saved
CPU-reference episodes,640/1280/1920-action prefixes average4875.000/11311.458/
15026.042. This selects1920 to expose later-board rewards; prefix scores are
neither natural-episode scores nor native learning. The full CPU reference
averages22050.000. The same development episodes inform both choices.

- Train200,000 aggregate actions /49,939 updates, first248, residual credit0;
  save100k recovery at24,939 updates and final200k. Training cap30 minutes.
- Evaluate fixed final and own actual initial on first3 natural episodes per8
  lanes,24 selected each. Guidance off, zero updates/truncations; retain original
  mean15000/90%-first-pyramid gate,600k-action and300s caps per arm.
- Require all finite reports/346 tensors, matched initial weights against the
  retained4013 control and bounded check,346 invariant tensors per frozen arm,
  identical frozen initial semantic traces, complete guided input/action/outcome
  replay, frozen outcome replay/videos, guards, recovery, curves and coverage.
  Ordinary frozen logs have no live pixel hashes. Training trajectories need
  not match because the teacher changes.
- Record actual guided-action fraction, guided/unguided episode outcomes,
  actions after guidance release and repeated replay-label exposure. Online
  returns include teacher actions; changing replay cross entropy is not held-out
  imitation accuracy. CPU audit cap1200s; service cap3660s.

Primary control is the retained200k/R32/first-pyramid/cap640 life:3153.125,
17/24 pyramids,20m07s. Secondary500k/R16 control scores3752.083,21/24 in28m20s;
it also changes action budget and replay ratio. All three use the same native
binary, boot, driver and model config except the secondary replay ratio. The
Python runner/guidance implementation changes. No isolated teacher/cap effect,
independent seed confirmation or five-game reliability claim follows this test.

Trial:`/x/Temp/kindle-qbert-guidance-20261027/qbert-4m-48-r32-continuation-cap1920-expert1-200k-seed4013`.
Owner:`kindle-qbert-continuation-cap1920-200k-20261027`.
External `prepare_full.py`, `run_full.py`, `check_startup.py`, `finish_full.py`,
`compare_full.py` and `analyze_full_exposure.py` retain the declaration, audit,
comparison and actual-exposure receipts. Three commands parse, seven scripts
compile, and both retained-control readers pass without environment/native
construction. The initial plot-environment import failure lacked ALE; the
unchanged preflight passes under the Atari environment before any GPU stage.
Native binary and applied adapter remain unchanged. No retry, cap increase,
extension, checkpoint selection or automatic successor is selected.


## Live startup

The selected service starts03:38:49 UTC/source`8761ff2`, PID81404,
invocation`ab0b0ad39c914bd89ab77e9e94773971`. Declaration SHA256
`f88fa2fe0baee5449dd48baa2ee0a04f2ee0f19b591864cdf1c034cf000996d3`.
All56 pins match. The initial346 tensors are finite and exactly match both
retained4013 control and bounded check. First updates are248:1/256:2/256:3/
264:4/264:5/272:6. At the first check,2312 actions have515 finite reports and
zero guided actions; no selected episode has begun yet. This is startup evidence,
not a frozen outcome. Read-only progress,100k recovery and terminal CPU
postprocessing observers are attached; no next GPU allocation is selected.


## Recovery checkpoint passes

At03:53:05 UTC, the100k recovery contains the expected24,939 updates and346
finite tensors. The watcher verifies25,837 finite reports in the103,600-action
complete prefix; last progress102,400/25,539 at844.397s, zero debt. Recovery
receipt SHA256`a7b9132298a8ac152518a7dcf895cbc79ecf991861ce18b7d4d36d1e2acd1fcb`. The same PID81404/invocation remains
active; no frozen task outcome exists yet. Keep200k/30-minute cap and pending
fixed final/own-initial assessments; no resume or additional allocation.

CPU-only Breakout movement diagnostics co-host50.202s of successful arm replays,
plus an initial failed/repeated two-arm attempt whose timestamp interval is
11.050s. A separate fixed Pong reference fit and one development-match/replay
use13.225s. Each runs on one affined low-priority core and leaves all56 pins
unchanged. Both source-version mismatches and CPU repeats are retained outside
Git; no GPU stage repeats. These are wall-time caveats, not time corrections.


## Completed fresh result

The selected life completes200k actions /799,570 frames /49,939 updates in
1522.860s (25m23s), within30 minutes. Frozen4730.208/23of24 first pyramids versus
own initial182.292/0 improves primary3153.125/17 and secondary3752.083/21.
The original mean15000 gate still fails despite95.83% first-pyramid success.
All numerical/config/counter checks,100k recovery,346 matched initial and
unchanged frozen tensors, initial semantic traces,56 pins, guided input/action
replay, frozen outcome replays/videos, guards, curves, coverage, exposure and
comparison pass. Main service exits04:07:44 UTC; postprocessing04:08:08.
No repeated GPU stage, extension or successor is selected. Both services are
terminal with MainPID0/Resultsuccess. See [full evidence and limits](../results/2026-10-25-cdp-cheap-qbert.md#continuation-guidance-improves-frozen-learning-but-misses-the-full-gate).


## Selected seed1009 check

After the4013 life and CPU postprocessing finish with all audits, select one
fresh1009 life to check the promising recipe on the weaker old-teacher seed.
The4013 gain is4730.208/23 first pyramids versus3153.125/17, but mean15000 still
fails. Its old-teacher1009 counterpart scored1083.333/2 in20m11s. Both seeds
and evaluation cohorts were previously used in development; this is a focused
paired check, not held-out or broad reliability evidence.

Keep the exact continuation recipe, native binary, Python runtime and host:
4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original rates,
5%episode guidance/cap1920/expert1/NOOPweight.05. Change only learner/evaluation
seeds to1009/4000001009. Train200k aggregate actions /49,939 updates, first248,
residual0;100k recovery at24,939 updates and final200k,30-minute training cap.
Restore no weights, replay, live belief, games or RNG. The native binary and
optional policy already pass bounded and full execution; no rebuild or new
backend qualification is needed for this seed change.

Fixed final and own actual initial each select first3 natural episodes per8
lanes=24, guidance off, zero updates/truncations, original mean15000/90%-pyramid
gate. Keep600k-action/300s cap per frozen arm,1200s CPU audit and3660s total
service cap. Require all finite reports/346 tensors, initial tensors matching
the retained1009 expert control and original1009 native check,346 frozen
invariant tensors, paired frozen initial semantic traces, full guided input/
action/outcome replay, frozen outcome replay/videos, host guards,100k recovery,
curves, coverage, exposure and four-arm comparison. Ordinary frozen logs lack
live pixel hashes. Actual guidance fraction is separate from5% episode selection.

Trial:`/x/Temp/kindle-qbert-guidance-20261027/qbert-4m-48-r32-continuation-cap1920-expert1-200k-seed1009`.
Owner:`kindle-qbert-continuation-cap1920-200k-seed1009-20261027`.
External`prepare_seed1009.py`, `run_seed1009.py`, `check_seed1009_startup.py`,
`finish_seed1009.py`, `compare_seeds.py` and shared`analyze_full_exposure.py`
are pinned by the declaration. Three commands parse, seven scripts compile,
and all three retained-control readers pass without constructing a native agent
or environment. All56 primary pins remain unchanged. The paired comparison
includes old-teacher1009/4013 and continuation1009/4013, with original gates.
No retry, cap increase, extension, checkpoint selection, third seed or automatic
successor is selected. Only this fresh1009 allocation is selected here.


## Seed1009 startup passes

Service starts04:17:32 UTC/source`7560d2d`, PID85716,
invocation`bef8ea63356848a88f69884efb2f039a`. Declaration SHA256
`73a79ddbfec359c92dfac75e7367c91b0a87688f436940e4967abd5dbf05324d`.
All60 pins match; all346 initial tensors are finite and exactly match the
retained1009 expert control and native check. First248:1/256:2/256:3/264:4/
264:5/272:6 updates match. At3008 actions,689 reports are finite,376 actions
are trusted and443 reports contain expert labels. Read-only progress,100k
recovery and terminal CPU postprocessing observers are attached. No frozen
outcome is available yet; retain the declared cap and full audit scope.

## Seed1009 recovery and co-host analysis

At04:31:53 UTC, the100k recovery passes with24,939 updates and346 finite
tensors. The watcher checks25,375 finite reports in the101,752-action complete
prefix; last progress98,304/24,515 at824.058s, zero debt. The original service
remains active under the same PID/invocation; no frozen outcome yet.

During training, [saved4013 replay analysis](../results/2026-10-25-cdp-cheap-qbert.md#saved-replay-locates-the-remaining-gap-after-the-first-pyramid)
uses2.518s of single-process unpinned log analysis,12.301s of affined low-priority
emulator replay and.025s of affined summary work. It executes only saved actions
with zero native decisions or updates and leaves all60 active pins unchanged.
The work is a co-host timing caveat, not a wall-time correction. It changes no
training settings or evaluation gate and selects no successor allocation.

## Prepared higher-exposure and lower-replay check

While the1009 confirmation is still active, prepare an external CPU-checked
candidate at`/x/Temp/kindle-qbert-exposure-20261027`. No GPU allocation is selected
by this preparation. Its controller requires terminal success and complete
learning/postprocessing/exposure reviews from the1009 confirmation before it
can run. The active runtime and all60 input pins remain unchanged.

The candidate keeps4M48/Auto/N8/B8/T16/H15/coefficient1/actor0/full18/original
rates, continuation/cap1920/expert1/NOOPweight.05 and fresh4013/eval4000004013.
Change episode guidance5%→20% and replay ratio32→16 together. The hypothesis
is that more teacher experience on later boards can improve transfer while
fewer repeated updates reduce wall cost. Neither effect is established or
isolated. The original mean15000/90%-pyramid gate remains the eventual target;
guided online scores cannot satisfy it.

The proposed check is8192 aggregate actions /994 updates, first248/residual0,
then1024 unguided frozen actions with zero updates. Training/frozen/audit caps
are180/60/180 seconds, total service480. Require finite reports/346 tensors,
346 matched initial and unchanged frozen tensors, changes to CNN/reward/actor/
value weights, hop labels/nonzero imitation, exact guided input/action/outcome
replay and host guards.8192 aggregate actions do not reach cap1920 in a lane;
the completed continuation lives cover that unchanged release schedule.

Two commands parse and three scripts compile without constructing an agent or
environment. The retained R16 log independently verifies994 updates at8192
and first248:1/256:2/264:3/272:4/280:5/288:6. CPU preflight takes.031s on one
affined low-priority core. No full learner, cap increase, retry, extension or
automatic successor is selected, and check experience will not be reused.

## Seed1009 completes and the bounded exposure check is selected

The confirmation finishes200k actions /799,538 frames /49,939 updates in
1597.708s (26m38s), within30 minutes. Frozen3137.500/17of24 first pyramids
versus own initial180.208/0 improves the paired old-teacher1083.333/2, but
misses both mean15000 and90%-pyramid criteria. All audits,100k recovery,
346 matched initial/unchanged frozen tensors, paired initial semantic traces,
60 pins, replays, guards, curves, coverage, exposure and four-arm comparison
pass. Main service exits04:47:41 UTC; postprocessing04:47:57. No GPU stage
repeats or completed life is extended. See [paired results and limits](../results/2026-10-25-cdp-cheap-qbert.md#continuation-improves-the-weaker-seed-but-both-miss-the-full-gate).

After both services are verified inactive/MainPID0/Resultsuccess, select only
the prepared8192-action/994-update **20% guidance/R16 bounded check** above.
Keep every declared setting, cap and audit. Owner
`kindle-qbert-r16-p20-check-20261027`, trial
`/x/Temp/kindle-qbert-exposure-20261027/qbert-4m-48-r16-p20-check`.
All33 input pins and the existing runtime match. No full learner or automatic
promotion is selected; the check cannot establish natural-episode competence.

CPU preparation for a possible200k/24,970-update comparison is also retained
externally: three commands parse, eight scripts compile, the primary-control
reader passes and the retained R16 recovery checker verifies100k/12,470 and
200k/24,970. It takes.082s on one affined low-priority core during the completed
confirmation and changes none of its60 pins. That possible comparison would
retain the full frozen protocol and a20-minute training cap, but has no trial,
declaration, selection or launch yet. Passing the bounded check is not automatic
authorization for its controller to promote itself.

## Higher-exposure R16 bounded result

Service starts04:51:06 UTC/source`e920f7b`, PID89803,
invocation`579681656ed744a5a22a2389d06c6d6e`; exits successfully04:52:23.
8192 actions /32,752 frames /994 updates take52.742s; first248/residual0.
All finite reports/346 tensors,346 matched initial and unchanged frozen tensors,
CNN/reward/actor/value changes,4527 trusted actions (1036 hops),990 supervised
reports,33 pins, exact guided replay and guards pass. Frozen1024 /4096 frames
take.524s with zero updates and no natural completed episodes. Review SHA256
`9b03f5c7dc686b554b821d9a3fdbc6912dfa73c38d924c7a261730db575871ef`.
See [evidence and limits](../results/2026-10-25-cdp-cheap-qbert.md#higher-exposure-r16-bounded-check-passes).

## Selected fresh higher-exposure R16 comparison

After terminal bounded success and the paired continuation audits, select one
fresh200k life on reused4013/eval4000004013. Keep4M48/Auto/N8/B8/T16/H15,
coefficient1/actor0/full18/original rates, continuation/cap1920/expert1/
NOOPweight.05 and the same native binary/runtime/host. Change episode guidance
5%→20% and replay ratio32→16 together. More teacher experience and fewer
repeated updates are a joint cheap-recipe hypothesis, not isolated effects.

The52.742s bounded training rate extrapolates to21.46 minutes for200k; later
cost and guidance exposure may differ. Before launch, replace the unselected
20-minute draft with a **25-minute training cap**. The original draft/scripts
and preflight remain under`prepared-20min`; no launched allocation receives a
cap increase. Updated three-command/eight-script CPU preflight and retained
control/recovery readers pass in.080s without agent/environment construction.

- Train200,000 aggregate actions /24,970 updates, first248/residual0. Save100k
  recovery at12,470 updates and final200k. Restore no weights or experience.
- Evaluate fixed final and own actual initial on first3 natural episodes per8
  lanes=24 each, guidance off, zero updates/truncations. Keep the original
  mean15000/90%-pyramid gate and600k-action/300s caps per arm; retain excess
  completed episodes and partial tails.
- Require all finite reports/346 tensors,346 initial tensors matching primary
  continuation4013 and bounded check,346 invariant tensors per frozen arm,
  paired initial semantic traces, full guided input/action/outcome replay,
  frozen outcome replay/videos, host guards, recovery, curves, coverage,
  exposure and paired comparison. Ordinary frozen logs lack live pixel hashes.
- Record actual guided-action fraction, guided/unguided online outcomes and
  repeated replay-label exposure. CPU audit cap1200s; total service3360s.
  No retry, cap increase, extension, checkpoint selection or automatic successor.

Primary control is continuation4013/5%/R32:200k actions/49,939 updates in25m23s,
frozen4730.208/23 first pyramids. Same initial model and evaluation seed, but
guidance probability and replay ratio jointly change training trajectories.
Both reuse development data; no independent seed confirmation is claimed.
Full-game success and five-game reliability remain unproved.

Trial:`/x/Temp/kindle-qbert-exposure-20261027/qbert-4m-48-r16-p20-cap1920-expert1-200k-seed4013`.
Owner:`kindle-qbert-r16-p20-cap1920-200k-20261027`.
External`prepare_full.py`, `run_full.py`, `check_startup.py`, `finish_full.py`,
`compare_full.py`, `analyze_exposure.py` and the retained R16 recovery watcher
are pinned with the runtime and evidence (47 inputs). Only this fresh comparison
is selected; no automatic seed confirmation follows it.

## Higher-exposure R16 startup passes

Service starts04:56:02 UTC/source`d8cdc8a`, PID90234,
invocation`f880fba1fa344bd7b84467a0f5ceeb5e`. Declaration SHA256
`e5800c0bc04e6021c26c3fc9bc7f18d307c4599503f02e6d6e80d93ba722cb5c`.
All47 pins match. All346 initial tensors are finite and exactly match both the
primary continuation4013 and bounded check. First updates248:1/256:2/264:3/
272:4/280:5/288:6 match the declared R16 schedule. At3224 actions,372 reports
are finite,1612 actions are trusted and371 reports contain expert labels.
Read-only progress,100k R16 recovery and terminal CPU postprocessing observers
are attached. No frozen outcome yet; retain200k/25-minute cap and full audits.

## Higher-exposure R16 recovery passes

At05:07:46 UTC, the100k recovery contains12,470 updates and346 finite tensors.
The watcher verifies13,000 finite reports in the104,248-action complete prefix;
last progress102,400/12,770 at689.563s, zero debt. Receipt SHA256
`47bd8bef3f5e43330283138f04c1173ea78b2e2fd6efd5431f8f1f7d7c93bdfc`.
The original service remains active under PID90234/invocation
`f880fba1fa344bd7b84467a0f5ceeb5e`; no frozen result exists yet. Keep the
declared200k/25-minute cap. The recovery observer completes successfully.

CPU preparation of a [four-arm retained Pong greedy check](../results/2026-10-25-cdp-cheap-pong.md#prepared-greedy-comparison-of-both-retained12m-learners)
uses1.806s of measured preparation/source-validation work on one affined
low-priority core. No native module, agent or environment is constructed;
all47 active Qbert inputs remain unchanged. This is a co-host timing caveat,
not a wall-time correction. That separate frozen assessment remains unselected
and cannot start before Qbert and its postprocessing are terminal.

A separate.261s affined log-only [Breakout prefix review](../results/2026-10-26-cdp-breakout-pixel-reference.md#longer-saved-prefixes-expose-most-teacher-reward)
compares one1920-action prefix with retained640/full controls. It constructs no
agent/environment, makes no new decisions and leaves all47 Qbert inputs
unchanged. No Breakout allocation or runtime change is selected.
