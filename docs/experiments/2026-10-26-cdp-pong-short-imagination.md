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
