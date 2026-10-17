# Complete the 12M capacity comparison on the other four Atari games

Goal: **DreamerV3 quality on Boxing, Pong, Freeway, Breakout and Qbert with CDP**,
retaining the budget comparison. None of the five long-run targets is achieved.
The [Breakout comparison](../results/2026-10-07-cdp-capacity.md) improves all three
seeds,5.14->38.21 frozen mean at500k actions, but costs3.58x the training time.
Test whether that whole-model capacity gain generalizes before another model,
loss, exploration or larger-budget decision.

## Reviewed runtime

The [projection rewrite](../results/2026-10-07-cdp-exploration-throughput.md)
fails its fixed numerical gates and is removed. No speedup claim or relaxed
tolerance. The separately qualified original graph uses Meganeuraf104f35,
Bladee349cddf, native83be73bf.20 guards pass:2,824 upstream comparisons,
independent primitive/grouped checks and all three production/frozen smokes.
One backend-only12M timing is107.85->108.00ms, effectively unchanged here;
all272 metric reports, first outputs and346 final tensors match exactly in
that fixture. This does not prove universal trajectory parity or erase the
failed stronger full-head gate. [Review](../../runs/meganeura-control-refresh-20261007.FZHHoQCK/review.json).

## Fixed learning allocation

- **Pong, Qbert, Boxing, Freeway**, in that order; seeds1009/2017/3019 per game.
  Twelve fresh12M learners,500,000 actual actions/124,939 updates each:
  **6M actions/1,499,268 updates** total. Retain all three completed12M Breakout
  learners; do not rerun them just to give the five games identical backend pins.
- Same centered CDP/action-effects coefficient1, N8/B8/T16/H15/R32,
  microbatch8/replay100000, cosine500, encoder6e-6/dynamics4e-4/base4e-5,
  warmup1000, AGC.3, ac_grads=false. Actual parameter count16.33M including
  CDP/exploration. Only the model preset changes versus retained1M controls;
  CNN, RSSM, policy and ensemble capacities change together.
- Native image input, one GPU RGB64 resize; full18/sticky.25/repeat4,
  no reset no-ops,100000-frame artificial cutoff. No game-specific settings,
  action hints, detector, RAM input, external reward shaping or video prior.
- Retain all three completed1M500k controls per game from the reconciled
  five-game report. Explicitly disclose their native6d38eea2/c637 backend;
  the new runtime qualification is not a retroactive repin. This is a capacity
  follow-up with a qualified backend refresh, not an exact same-binary ablation.
- Save actual initial weights before experience. After each learner: audit
  finite reports/checkpoints, exact configuration, actual124,939 update counter,
  zero training debt and every reported update; do not infer counts from wall time.

## Frozen controls and reporting

After each audited learner, evaluate final and actual-initial models before
the next learner. Use the same development-evaluation base4,000,000,000+seed
as the retained controls, eight independent stream offsets, sampled policy,
first3 **natural** episodes/stream and600k-action/30min caps. These reused seeds
are not an untouched final test. No model/optimizer updates; require exact346
saved tensors per frozen model. `cohort(..., natural_only=True)` is explicit.

Retain all cutoffs, excess episodes, unfinished tails and failed stages. An
incomplete natural cohort stops for review; it never deletes or substitutes
a seed. Replay every action/reward/reset/frame on CPU and save whole stream-zero
videos. For each game report all three online last50 curves against actions/time,
frozen final/initial/retained1M means and paired learner-seed bootstrap intervals.
Do not treat streams or episodes as independent learner seeds.

The unchanged long-run references remain Boxing99.6133, Pong20.4455,
Freeway33.3993, Breakout381.8114, Qbert193220.7665. Our500k actions are about2M
frames versus their200M; online and frozen protocols/windows differ. Do not
declare parity or a matched RGB compute-saving result from this capacity screen.

Expected**47–48hours** including evaluations at current measured cost, not a
guaranteed finish time. Each learner5h, each frozen worker30min; one72h-bounded
persistent service. Restart=no, KillMode=control-group, zero swap, serial host
guards, RTX5080/driver580.178.04 and>=2GiB sampled Vulkan estimated headroom.
Record standalone allocation warnings; stop/review API, numerical, hard-fault
and deadline failures. No separate NVML polling/recovery or blind retries.
GPU workers keep normal CPU allocation; CPU preparation/replay/audit uses one
CPU/2GiB. Inspect long training about every30min or on completion, not every
counter. No rebuild during the allocation and no automatic budget extension.

Artifacts:`runs/cdp-12m-four-20261007.iEvfk9Vm`. Source/native/ROM/job identities
are sealed before launch. This allocation advances the five-game quality goal;
finishing it does not complete that goal or its budget claim.
