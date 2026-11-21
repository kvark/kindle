# Bounded 12M frozen actor posterior averaging check

Select two new frozen 4,096-action executions after the
[Tiny native qualification](../results/2026-11-06-cdp-actor-posterior-average.md#native-qualification-complete)
passes. Restore the retained Breakout 200k checkpoint, with 357 tensors and
12,477 learner updates, from
`/x/Temp/kindle-breakout-replay-value1-20261102/full/breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009/final`.
Original training remains 21m27.910s; no learning or new initial-model arm.

Use package fddf48ba/native SHA256
`960667d10fb031ec82963f5cd644802bd650c273c64db914ec6c46f6a6265680`
and its matching runner. First ordinary sampled-state/greedy-action inference;
then eight additional actor posterior samples with the same sampled live state
and greedy actions. Keep eight independent environments, evaluation seed
4000001009, full18/sticky.25/repeat4/noops0, native max-pooled RGB and one GPU64
resize. No aids, labels, temperature, modal posterior, learning or source changes.
Old metadata gains only the explicit existing `replay_value_bootstrap=imagined`
default; normalize that serialization difference in checkpoint/header checks.

Require all ordinary semantic action/episode/reset events through 4,096 actions
to equal the saved retained greedy evaluation. Validate environment/model/native/
runner identity, counters, finite values, both exported 357-tensor checkpoints
including optimizer state, and full recorded-action/reward/frame/reset replays.
Frozen logs lack live image hashes; no unrecorded pixel equality is claimed.
Record episodes and partial tails without using this short prefix as a score.

Cap each GPU child at 120s, with 2GiB minimum budget headroom and the existing
AMD host guard. No genuine device fault, nonfinite value or surviving child may
be ignored. Require candidate elapsed seconds per action ≤2× ordinary; report
construction separately. Cold compilation and changed trajectories limit this
short timing comparison. Also require matching candidate inference metadata and
zero updates. No cap increase, reset/reload, retry, completed-stage repeat or
automatic full comparison.

Before execution, exercise both actual CLI paths without native/emulator
construction, audit rejection cases, source/checkpoint pins and known ordinary
prefix. Root `/x/Temp/kindle-actor-posterior-average-20261106/bounded`; keep
exclusive logs, queues, declarations and terminal receipts. A separately fixed
trained-policy evaluation may be selected only after this check passes. The
original 864/90%-two-wall and five-game goal remain unchanged.
