# Pong CDP: distinguish an early-learning floor from persistent failure

This is a separately reviewed follow-up to the completed
[five-game screen](../results/2026-10-05-cdp-five-game-screen.md), not an
automatic extension, checkpoint resume or restart of an old queue.

## Question and choice

At200k actions (about800k frames), Pong's online mean is−19.33 and frozen
seed scores are−20.83/−14.08/−20.67 against initial controls around−20.3.
Two seeds have weak reward-event separation and low raw KL. Those diagnostics
do not prove visual collapse or a backend bug. The released Atari57 DreamerV3
mean is also−20.50 at800k frames, improving to−7.16 at2M.

Change only the finite learning budget to **500,000 actions** (about2M
frames), with three fresh learners. This tests whether the early floor
persists at the reference's first substantial-learning region. Keep the small
model, representation, exploration coefficient, rates, replay and horizons
unchanged. No capacity/loss/precision tweak, new action aid, RGB/Tiny matrix
or duplicate backend qualification. Meganeura upstream6288f885 remains a
docs-only change over qualified592a2f5a; Bladee349cddf is unchanged.

## Fixed allocation

- ALE/Pong-v5, learner seeds1009/2017/3019, in that order.
- **Three fresh500,000-action /124,939-update runs**:1.5M new actions and
  374,817 updates total. No selection or restart of the weak seeds.
- Nativef4b6a5c7; CDP plus the unchanged action-effects bonus. Full recipe from
  the five-game declaration: Size1M/N8/B8/T16/H15/R32/microbatch8/replay100000,
  full18/sticky.25/repeat4/no reset no-ops/100000-frame episode cap. Native
  observations, one GPU resize toRGB64. Cosine500, encoder6e-6/dynamics4e-4/
  base4e-5, warmup1000, AGC.3, ac_grads=false, intrinsic coefficient1.
- No action assistance, externally shaped reward, video pretraining or new
  game-specific model settings. Repeating the fresh200k prefix is additional
  compute, not hidden experience or an uninterrupted lifetime.
- After each training: complete counter/debt/finite-checkpoint audit, then
  frozen final policy and actual saved zero-experience initial control.
- Evaluation: sampled policy, first3 completed episodes per each of8 streams,
  cap200k actions/30min per model. **New held-out base3,000,000,000 plus learner
  seed**, stream offset1,000,003 modulo2^32. Same seeds within each pair;
  no training on evaluation rewards. Retain cutoffs/excess episodes/tails.
- Exact unchanged saved tensors, full CPU trajectory replay and whole
  stream-zero videos before advancing. No best-checkpoint selection.
- Expected about3.5–4h including audits;6h service deadline,100min per training
  process,30min per evaluation. Persistent serialized GPU service, Restart=no,
  KillMode=control-group, host guard, expected RTX5080 and>=2GiB sampled Vulkan
  estimated budget headroom. Standalone allocation warnings recorded; API/
  numerical/hard faults or deadlines stop for review. No NVML polling/recovery
  or automatic retry. Heavy CPU analysis gets one CPU,2GiB and zero swap.

The unchanged orchestration is adapted only for budget, deadline and held-out
evaluation seeds, then CPU-rehearsed on an actual completed Pong trio.
No new production source or native rebuild is required.

## Readout and stop decision

Report all three score/action/time curves, final frozen versus initial-control
scores with seed-bootstrap uncertainty, actual frames/updates/elapsed time and
late reward-event separation/KL. Compare online curves at approximately2M
frames against **all six** released Atari57 Pong traces; retain the200M target
20.445 and the separate Atari100k reference−5.0. These remain different
protocols/windows, not a matched same-hardware compute comparison.

Useful progress is stable retained improvement across learner seeds, not just
one upward curve, an episode win or low latent loss. Do not call the full goal
complete if this short study improves Pong: Boxing, Freeway, Breakout and Qbert,
the long-run scores and the budget question all remain in scope. Failure or
persistent weak prediction leads to task-state/prior-forecast diagnosis before
more training. Even a positive result ends at the declared budget and requires
a reviewed next decision; there is no automatic longer or full-suite queue.

Artifacts: `runs/cdp-pong-budget-20261005.LTG3h04X`.
