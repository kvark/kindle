# Qbert: frozen greedy versus sampled actions, before another learner change

The [paired forecast check](../results/2026-10-11-cdp-qbert-forecasts.md) finds
useful h15 rewards in every2M model, but control reliability is unfinished.
Test one existing runtime choice: greedy rather than sampled policy actions.
No native rewrite, retraining, parameter transfer or world-model change.

## Fixed allocation

- Completed2M Qbert seeds1009/2017/3019, their actual initial checkpoints, same
  native7311547d/Meganeura31026833/Blade56f0565. Preserve the previous sampled
  candidate/initial cohorts as controls, with their exact runtime and seed pins.
- Six fresh frozen evaluations, final then actual-initial for each seed, using
  the existing `atari_vector.py --evaluate --greedy` path. Model configuration
  and all parameters stay unchanged. No exploration override or action hint.
- Same N8/eight host workers, full18/sticky.25/repeat4/no reset no-ops/native
  frames/one GPU RGB64 resize. Environment seed base4,000,000,000+learner seed.
  Select the first3 **natural** episodes per stream:24/model,144 total.
  Explicit `cohort(..., natural_only=True)`. Keep excess episodes and tails.
-600k-action/30min cap per frozen job. An incomplete natural cohort stops the
  allocation for review; do not count cutoffs as natural, change seeds, extend
  the cap or retry. Worst-case3.6M evaluation actions; zero learner updates.
- After every capture, require exact346 saved tensors, zero updates, source
  identity, full CPU action/reward/reset/frame replay and whole stream-zero
  video. Independently select task outcomes from the same natural episodes.

## Decision and limits

Report every seed's final/actual-initial score and first-pyramid fraction,
paired differences from the retained sampled cohorts, learner-seed bootstrap
intervals, actual actions/time and all failures. Do not average streams as
learner replicates. Historical>=90% first-pyramid/mean>=15,000 gates and the
193,220.77 long-run reference stay unchanged. No best-mode or best-seed score
may replace the existing standard sampled result.

A consistent gain would identify action sampling as one contributor and
motivate a separately reviewed policy-training change. No gain would leave
policy/value learning and longer-horizon survival/credit assignment open.
Either outcome is a bounded diagnostic, not five-game DreamerV3 parity, new
learning evidence or permission for a policy/entropy sweep. All five titles
and the matched RGB budget question remain in the goal.

GPU work is serialized under the existing host guard in a4h systemd service,
Restart=no/KillMode=control-group,30min/job. Keep GPU workers without a CPU
quota, expected RTX5080/580.178.04 and>=2GiB sampled Vulkan headroom. Record
allocation warnings; stop on API/numerical/hard faults or deadlines. No NVML
polling or recovery. Per-stage CPU replay/review uses one CPU/2GiB/zero swap,
10min deadlines and service lifetime ownership. Preparation verifies all
source/native/checkpoint/ROM pins before any new action.

Artifacts: `runs/cdp-qbert-greedy-20261011.5vPsrd3M`.

The first CPU preparation fails before any GPU work: the lazy package has not
loaded `_native`. Resolve its module path with `find_spec` for hashing instead.
The failed controller source remains; no actor or game was constructed.
