# Bounded native Qbert continuation-guidance check

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
