# Intermediate-capacity Qbert development screen

Prepare one fresh seed1009 to advance the harder-game gap after reviewing the
active Boxing500k/R32 allocation. This is not an automatically launched queue.
Historical12M Qbert improves with experience but remains below its first-pyramid/
15k gate; historical1M is much weaker. Use intermediate capacity and48x48 input,
whose native execution and frozen restore already pass. Visibility samples retain
the pyramid and actor; actual learning remains unproved.

Centered CDP/action-effects1, Size4M/RGB48/Auto, N8/B8/T16/H15,
full BPTT/microbatch8, replay100k and ratio32. Keep full18 actions, sticky.25,
repeat4, native max-pooled frames and one GPU resize. No action aid, new reward,
pretrained encoder or RAM input. RAM is used only by the existing post-hoc CPU
first-pyramid observer during replay. Actual trainable count3,578,225.

Fixed200k aggregate actions, expected49939 updates and first update at248, with
an1800-second training cap including construction. Save actual initial and fixed
final weights. Evaluate at seed4000001009, first three natural sampled episodes
per lane,24 selected per arm, zero updates,600k-action/300-second cap per arm.
Keep all excess episodes, tails and cutoffs. Use the existing Qbert gate: at least
20 completed episodes, at least90% first-pyramid successes and mean at least15000.
A positive score alone is not success. No best-checkpoint selection or extension.

Audit every numeric report, update/frame counter, complete configuration and
source identity; verify finite checkpoints, all346 unchanged frozen tensors and
full CPU action replays with stream0 videos. Audit cap1200 seconds; service cap
3660 seconds. The earlier4M/RGB48 smoke supplies the exact architecture initial
control (same seed and18-action schema; replay ratio does not change weights).

Artifacts: `/x/Temp/kindle-cheap-20261025/qbert-4m-48-r32-200k`.
Controller: `run_qbert_200k.py`; intended service owner:
`kindle-cheap-qbert-r32-200k-20261025`. Exact commands, native/source hashes and
caps are declared before launch. Use the existing qualified runtime, with no
rebuild. Capacity, resolution, runtime and budget differ from historical controls;
this tests a useful cheap recipe, not an isolated ablation. Seed1009 is reused
development data; any promising result still needs confirmation across seeds.
