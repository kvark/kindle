# Intermediate-capacity Pong development screen

Fresh seed1009, centered CDP/action-effects1, Size4M/RGB48/Auto,
N8/B8/T16/H15, full BPTT/microbatch8, replay100k/R16. Full18 actions,
sticky.25, repeat4, native max-pooled frames and one GPU resize. No action aid.
Actual trainable count3,578,225. Historical1M Pong was unreliable, motivating
intermediate capacity; the visibility sample retains the ball at48. The bounded
4096-action smoke and1024-action frozen restore pass, with all346 tensors
unchanged. These are execution checks, not learning evidence.

Fixed200k aggregate actions, expected24970 updates, first at248;1800-second
training cap including construction. Save the actual initial and fixed final.
Evaluate both at seed4000001009, first three natural sampled matches per lane,
24 selected per arm, zero updates,600k-action/300-second cap per arm. Use the
existing Pong gate: at least20 natural matches, mean at least15, win fraction
at least.9 and no truncations. Audit all counters, finite reports/checkpoints,
frozen tensor equality and full CPU action replays with stream0 video. Audit
cap1200 seconds; service cap3660 seconds. No automatic extension or seed queue.

Source/native identity, controller hashes and exact commands are declared at
`/x/Temp/kindle-cheap-20261025/pong-4m-48-200k`. Controller:
`run_pong_200k.py`; service owner `kindle-cheap-pong-200k-20261025`.
Use the already qualified resize runtime. The preceding Boxing trial is complete
and fails its gate. This tests a different game/capacity; it is not a matched
resolution ablation. One reused development seed does not establish reliability.
