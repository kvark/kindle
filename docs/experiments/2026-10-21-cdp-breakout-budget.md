# Breakout: fresh longer-budget learning

After the three Pong500k seeds pass the intermediate gate, move to a weaker
game. Historical12M Breakout500k scores48.17/15.88/50.58, mean38.21, versus
1M5.14. Existing capacity helped; the current fast BF16 implementation has not
yet been tested on Breakout. Old1M state/readout failures do not diagnose12M.

Run **one fresh seed1009 for2M actions**,499,939 updates, zero final debt.
Retain the current unassisted12M/B8/T16/H15/R32/BF16 centered CDP/action-effects1
recipe, microbatch8/replay100k, eight environments and batched inference. No
action overrides, game-specific hints, reward shaping, detector or RAM input.
Full18 actions/sticky.25/repeat4, native pixels with one GPU64 resize. No new
backend optimization, representation, precision or hyperparameter sweep.

Save actual-initial weights and500k/1M/1.5M/2M snapshots. Evaluate only the
predefined500k prefix,2M final and actual-initial models, sampled actions,
24 natural episodes each, seed4000001009,600k-action/1800s caps. Export frozen
models and compare all346 tensors exactly; audit zero updates, source/config,
finite checkpoints, complete action/reward/frame replay and whole videos.
Retain every cutoff/excess episode/unfinished tail and report two-wall864-point
events from the CPU replay observer. Observer state is never an actor input.

The fixed published381.81 target and historical two-wall gate remain unchanged;
neither is assumed achieved by a short-budget improvement. Published scores
come from much longer training and different protocols. Compare500k→2M within
this life; old-backend comparisons do not isolate experience or precision.
Review the result before any other seed, extension or recipe change. One seed
cannot establish reliability. Preserve failed work; do not resume a partial life.

Qualified native c8aeba28, no rebuild. Train deadline14400s, whole service24000s,
12GiB/no swap/no restart. CPU review one CPU/2GiB/no swap. Host-only guard records
allocation warnings; genuine API/numerical/host faults stop. No NVML polling.
Fresh NFS storage avoids the nearly full local disk. No automatic extension.

Service: `kindle-cdp-breakout-budget-1009-20261021`.
[Declaration](../../runs/cdp-breakout-budget-20261021.wqYChxy8/declaration.json).
[Controller](../../runs/cdp-breakout-budget-20261021.wqYChxy8/run.py).
