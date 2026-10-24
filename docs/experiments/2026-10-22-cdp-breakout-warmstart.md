# Breakout: retain learned weights and acquire more experience

The2M learner scores98.21 versus58.83 at500k. Its frozen15-step cumulative
reward forecasts improve and beat all fixed controls, with important timing/
false-positive limitations. Test more experience before changing architecture.
Published381.81 and the historical two-wall864 gate remain unchanged.

## Explicit warm start, not an exact lifetime resume

Restore the qualified2M source checkpoint, retaining parameters, both Adam
moments, slow critic, return normalization and counters. Replay, live beliefs,
games and RNG trajectory are not preserved. Native restore deliberately starts
between episodes with empty replay and a fresh scheduler; it earns no credit
until replay is ready. Label this a new learning life of the same lineage.

Keep config seed1009 and all12M/B8/T16/H15/R32/BF16/CDP/action-effects1 settings;
use new environment seed101009 with the existing stream offsets. No training
override, game hint, RAM input, reward shaping, new backend or loss change.
Full18/sticky.25/repeat4/native pixels/one GPU64 resize stay unchanged.

First restore the original source into a disposable4096-action smoke:
expected963 new updates, absolute learner counter500902, fresh replay warmup,
zero final debt. Then1024 frozen actions and exact346-tensor export. Verify
the actual executed-action ledger and absolute counters independently; a
fresh-only analysis helper must not relabel this as a new untrained model.
Training cap300s, service2400s. Review before the main allocation.

## Fixed main budget after the smoke

Start again from the original2M checkpoint, **not the smoke checkpoint**.
Add6M actions (8M total lineage exposure),1,499,939 updates, expected absolute
learner counter1,999,878 and zero debt. Estimated training7½ hours; cap10h,
whole service12½h. This is one finite learner, not a broad replication queue.

Save each1M additional actions. Predeclare evaluations at2M and6M additional
actions, plus the unchanged source checkpoint:24 natural rounds each, sampled,
evaluation seed4000001009,600k-action/1800s caps. Compare exact346 frozen tensors,
full replay/videos, reward scores and two-wall events. Keep all cutoffs/excess/
tails and do not choose the best saved snapshot. The old actual-initial1.42
control remains available, but is not the starting policy of this warm life.

Report additional and total exposure separately, including the discarded smoke
cost. Source-policy comparison does not isolate a pure uninterrupted-budget
effect because replay/games/RNG restart. One lineage is not seed replication.
Review final evidence before any extension or other learner. Five-game mastery
and matched RGB comparison remain open.

GPU workers12GiB/no swap/no CPU quota; CPU audit one CPU/2GiB/no swap.
Host-only guard, explicit current driver, no NVML polling/recovery. Allocation
warnings recorded; genuine GPU/API/numerical faults and deadline overruns stop.
Native/source hashes and the old evidence remain fixed. Fresh NFS artifacts.

Artifacts: `runs/cdp-breakout-warmstart-20261022.ec73sFZx`.
[Controller](../../runs/cdp-breakout-warmstart-20261022.ec73sFZx/run.py).

Smoke completes02:32:42UTC:4096 actions/963 new updates in25.25s, absolute
counter500902, environment counter2,004,096, zero debt;1024 frozen actions and
all346 tensors unchanged. Both guards and independent replay/counter audit
pass. The separate main service is `kindle-cdp-breakout-warmstart-20261022`;
its declaration retains the original source checkpoint, not the smoke output.
