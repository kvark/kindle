# Fixed-recipe confirmation of Pong's smaller bonus

**Selected after the completed coefficient.01 primary and comparison.** Its
frozen14.9583 and24/24 wins improve over11.0/23 wins at the same500k budget, but
still miss mean15. Confirm this promising change on previously used4013 without
rounding the primary into a pass, relaxing the gate or extending its life.
The four-action Breakout candidate remains prepared for later review.

Only learner/evaluation seeds change from1009/4000001009 to4013/4000004013.
Keep centered CDP, intrinsic coefficient.01,4M/RGB48/Auto,3,578,225 parameters,
N8/B8/T16/H15/R32, full BPTT/microbatch8, replay100k, actor_unimix0, full18,
sticky.25/repeat4, zero no-op starts and100000-frame episode limit. Native
max-pooled frames receive one GPU resize. No external aid, RAM, reward shaping,
optimizer change, native rebuild or concurrent learner.

Fresh counters, replay, games, belief and RNG; save the own actual-initial
checkpoint. Verify all346 initial tensors against the completed coefficient1
seed4013 control. That same-seed control differs only in intrinsic coefficient;
the primary coefficient.01 recipe differs only in seed. These are two distinct
comparisons. Seed4013 and its evaluation seed are reused development data.

Fixed500k aggregate actions, roughly2M emulator frames,124,939 expected updates,
first update248 and a3300s/55-minute training cap including construction. Audit
100k/200k/300k/400k recovery checkpoints, then evaluate fixed500k final. No best
checkpoint, automatic resume, cap increase or unchanged extension. Retain any
cutoff explicitly and do not restart completed work.

Frozen final/own initial: first three natural sampled matches per lane,24 per
arm,600k-action/300s caps, zero updates/truncations. Retain excess matches/tails.
Keep at least20 natural matches, mean≥15, at least90% wins and zero truncations.
Audit every numeric report, full configuration/source/counters, finite
checkpoints, all346 unchanged frozen tensors, complete CPU replay/video and
host guards. Audit cap1200s; service cap5160s. Stop on real faults/numerical failure.

Native`bfb8c31`, runner`1ed07af` and CPU replay`e853818` stay fixed. Artifacts:
`/x/Temp/kindle-cheap-20261025/pong-4m-48-r32-intrinsic001-500k-seed4013`.
Controller:`run_pong_intrinsic001_500k_4013.py`; service:
`kindle-cheap-pong-intrinsic001-500k-4013-20261026`.
CPU preparation verifies the complete queue changes only seeds/output paths;
unselected execution refuses before GPU/log artifacts. Selection pins the
preceding completed review/comparison and requires its service to be terminal.
One focused confirmation, not a broad campaign; no automatic next allocation.

Started07:36:01 UTC, source`1ff41da`. Startup passes both full-config comparisons,
zero starting counters,3,578,225 parameters, all346 finite initial tensors
matching the same-seed coefficient1 control and updates248:1,256:2,256:3.
Initial metadata SHA256
`427af9e6b2308d5aebc1dcc6f9caafad56aac3c4cccd4975e8ae22c942abd2c6`;
selected declaration`2cc6d4a2fe7113d81960a8b3b38011110e60f01a7afab299f34466099bea48af`.
Read-only recovery monitoring covers60 minutes; completed-run plotting is
automated. The first watcher invocation omitted its milestone arguments and
exited before monitoring; the corrected invocation is live. Training was not
restarted and its limits did not change. Frozen results remain pending.
