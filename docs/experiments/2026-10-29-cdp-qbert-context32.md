# Qbert longer temporal training at fixed replay-position work

Select one bounded fresh **12M48/Auto/N8/B8/T32/H15/R16** check. The retained
12M/B16/T16 learner reaches11984.375 greedy or12389.583 at frozen action
temperature.5, below the original15000/90%-first-pyramid gate. Increasing
capacity to25M does not improve its primary greedy score. Later-game action
rankings are weaker than early rankings. Longer temporal credit assignment is
an untested candidate on this compact supervised Qbert recipe, not an established
cause of the gap. The old Breakout B2/T64 result cautions against assuming that
longer sequences help.

Change sequence/world-backprop length16→32 and batch/world-microbatch16→8.
Keep256 sampled positions per update, replay ratio16,12M capacity,48px native
GPU resize and all other learning settings. This changes temporal gradient
length, independent-sequence count and replay sampling together. Behavior and
expert targets use B*(T−1), increasing240→248 rows per update. Replay warmup
also changes: expected first update504 rather than376 actions, giving481
updates at8192 actions and12469 at200k, eight fewer than the control. Do not
claim an isolated memory effect or exactly matched optimizer counts.

Keep centered CDP/coefficient1/actor-unimix0/full18/original rates;seed4013 and
evaluation4000004013;20% episode continuation guidance/cap1920/expert100/all18
weights1/world gradients/all-state corrective labels. Use the same native447fe50,
runner and teacher as the retained control, from
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. The newer optional frozen
action-temperature mode is not part of this training check. No new native build,
checkpoint weights, replay, belief, games, RNG or prior check experience is reused.

Run8192 aggregate training actions, then1024 greedy frozen actions with zero
updates. Require exact config/source/counters,357 finite and control-matching
initial tensors,357 unchanged frozen tensors, changed CNN/reward/actor/value
weights, finite aligned actor/world expert losses, measured actual overrides,
complete training pixel/action/corrective-label replay and host guards. The
checkpoint geometry remains12M48; compare with the retained12M initial and
the existing independently checked schema. Preflight parses both commands and
checks the config boundary without constructing an environment or native agent.

Caps: training180s,frozen60s,CPU audit180s,whole service480s. Conservatively
project200k cost as measured training seconds*200000/8192*1.10; require≤2700s
(45min) before considering a separately selected full life. A passing short
check establishes execution and early cost only. It does not establish natural
episode competence or automatically authorize a full learner. No retry, cap
increase, extension, sweep or reuse of check experience follows. Stop on real
GPU faults/numerical failures; no driver recovery.

Root `/x/Temp/kindle-qbert-context32-20261029/native-check`, trial
`qbert-12m-48-b8-t32-r16-expert100-check`, service
`kindle-qbert-context32-check-20261029`. All preceding Kindle services are terminal
at selection. The declaration pins a fixed protocol copy and exact executables;
check live state again before launch. Existing completed lives remain terminal.
