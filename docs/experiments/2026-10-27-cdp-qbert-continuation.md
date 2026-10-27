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
