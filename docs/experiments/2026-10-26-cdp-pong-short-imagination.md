# Pong with shorter imagined rollouts at a fixed500k budget

**Selected after completed Qbert aided evaluation and analysis.** The aid lowers
Qbert's frozen score and coverage without a pyramid success; no extension is
selected. PongB16 saves22.37% wall time but lowers score14.958→11.875, and doubled
rates lower it again to9.125. This test keeps B8's original cadence and rates,
shortening only imagined actor/critic rollouts from15 to5 steps.

The completed BreakoutH5 run exercises this native path for121,873 updates and
passes ordinary frozen restore, despite missing its500k action target at the
45-minute cap. Its matched196,608-action prefix costs18.06% less wall time than
H15. That is cost motivation, not a Pong speed or learning prediction; Breakout
itself remains weak and its separate retained-model H5 warm start regressed.
No native/backend change, new shape qualification or automatic cap increase
follows. Keep the existing numerical checks and GPU host guard.

Fresh1009/eval4000001009, centered CDP4M/RGB48/Auto/N8/B8/T16/H5/R32,
3,578,225 parameters, full BPTT/microbatch8, replay100k, coefficient.01,
actor_unimix0, full18/sticky.25/repeat4/noops0,100000-frame episode limit and
original rates. Native config differs only in imagination_length15→5 from the
completed coefficient.01 B8 control. World replay sequence length and update
schedule stay fixed; shorter imagined rollouts and value bootstrapping can
change policy learning. Native max-pooled frames receive one GPU resize.
No external aid, RAM policy input, teacher or reward shaping is used.

Keep500k aggregate actions/124,939 updates, first248 and15,992,192 sampled replay
positions. Four100k recoveries have24,939/49,939/74,939/99,939 updates; fixed final
is500k. Set a45-minute training cap including construction, versus the control's
55-minute cap and measured49m55s. Retain any cutoff explicitly, without resume,
cap increase, extension or earlier-checkpoint substitution. This is a separate
fresh life with own actual initial, which must match all346 H15 control tensors;
reused learner and evaluation seeds are development data.

Evaluate fixed final and own actual initial for the first three natural sampled
matches per lane,24 per arm, under600k-action/300s caps. Keep mean15/90%-wins,
at least20 natural matches and zero updates/truncations. Retain excess episodes
and partial tails. Audit all124,939 numeric reports, full config/source/counters,
finite checkpoints,346 unchanged frozen tensors per arm, complete CPU replays/
videos and three guards. Compare action/frame/time curves and frozen results
against the completed H15 control, including semantic initial-trace equality.
Audit1200s/service4560s; no automatic next allocation.

Runtime: native`bfb8c31`, CLI`22bd61b`, replay`e853818`, unchanged binary.
Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-h5-r32-intrinsic001-500k`;
controller:`run_pong_h5_intrinsic001_500k.py`, SHA256
`e2e994d6adfd82a436d4cbff511976a94308d3277b52cd49617ad61a52b9c37c`;
prepared declaration`63d4d9666d1349ac22e7c9dd4d05abd5497250eab570d1b6f42f4de0031f3dbc`;
queue`a7e4951c7fd6c5fb091f133011af8ddddf19f465ef40317e1961029b2db2268f`.
Compilation, complete queue normalization and unselected refusal pass. The
completed-run reader validates the H15 control; comparison/curve automation is
prepared, with no process started from preparation. Comparison SHA256
`8381b57cd2996c5ac2cb72381b2cb63795927c35bc9c7b3bf223a1ef32915251`.

## Started on AMD

Started October26 at12:38:15 UTC, source`36500e7f969b86bb8e55b8fb2cc76e00114198b8`,
owner`kindle-cheap-pong-h5-intrinsic001-500k-20261026`, invocation
`2d611a7c16e94d1d92c382b33a4f1288`. Selection verifies the completed Qbert review,
postprocessing/comparison, terminal service, clean pushed source, host identity
and exact runtime/controller/queue hashes. Selected declaration SHA256
`538252e5e19818cfd4f24c8a60be4b5767a427a73e6931167314d6f7757bcb97`.

Startup passes full native config equality exceptH15→5,3,578,225 parameters,
fresh1009/no aid, all346 finite initial tensors matching the H15 control, and
numeric reports at248:1/256:2/256:3. Initial metadata SHA256
`687d4090d68632ec18fb8e4bb668f8ae62a2a23747a944670a6b2af6123e3ecd`;
startup review`1386dcf36abb32bea864f35a2faa5e0288952a9908389e6bbb44e5d5523dfcfd`.
Four recovery audits, fixed final/initial evaluation, complete replay/guards,
curves and completed comparison are running or waiting automatically. These
startup checks establish execution and initial equality; learning remains pending.

The100k recovery passes at24,939 updates with all346 tensors finite and25,383
strict reports through101,784 actions. Checkpoint metadata SHA256
`6819a61db9af700823cd3130fe8850a8f8eac54cf3ee051dc3ae6fd8a515bca0`.
The matched98,304-action/24,515-update prefix costs440.533s versus H15's584.186s,
24.59% less wall time, with393,138/393,167 emulator frames and zero debt.
The retained prefix digest identifies every complete log record through that
progress point. Cost-review SHA256
`0d6b8a1f6113179bd5e1ed329748dde6bd32b6d0da8f6438f96ba2086e3041dc`.
This is incomplete-run cost evidence; trajectories differ and frozen learning
remains pending. No budget or gate changes follow from the early speed.

The200k recovery also passes at49,939 updates,346 finite tensors and51,755
strict reports through207,272 actions. Captured204,800-action progress is926.108s
with zero debt. Checkpoint metadata SHA256
`50ec20e7a2541346b755690063138165142fa2cf4dec7ef9d9272f207b6918b2`.
The service continues toward the fixed500k endpoint; no frozen result yet.

The300k recovery passes at74,939 updates,346 finite tensors and76,307 strict
reports through305,480 actions. Captured303,104-action progress is1,377.361s
with zero debt. Checkpoint metadata SHA256
`85d9f29bb41da9da75289bf4018fc0beb837718f9b3708d0115c9aa238d9b9dc`.

All four recovery audits now pass. The400k checkpoint has99,939 updates and346
finite tensors;100,803 strict reports pass through403,464 actions. Captured
401,408-action progress is1,830.097s with zero debt. Checkpoint metadata SHA256
`6572e8436401aec668e905f153413c6dfc0210063ada594faf7e099092856dcf`.
The recovery watcher exits successfully while training continues to500k.
