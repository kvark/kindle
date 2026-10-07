# Centered CDP capacity: bounded existing12M preflight

The goal remains **DreamerV3 quality on Boxing, Pong, Freeway, Breakout and
Qbert**, including the budget comparison. All five remain open. The completed
[500k allocation](../results/2026-10-07-cdp-centered-five-game-budget.md) improves
every seed but leaves Breakout/Qbert weak and Pong unstable.

[Breakout diagnosis](../results/2026-10-07-cdp-breakout-world.md) finds weak
learned ball state/forecasts. Pixel whitening invalidated the first readout;
the fixed-range head still fails to learn the ball. A separate fixed color
detector shows the saved RGB64 input retains ball location on483/483 eligible
held-out open-playfield frames. No detector/RAM enters the actor.

## Decision

Test the already-supported **Size12M preset** before inventing another objective
or encoder. Relative to Size1M it raises CNN depth4->16, embedding256->1024,
deterministic state512->2048, categorical classes4->16 (32 groups), and
recurrent/policy hidden widths64->256. This is one **whole-model capacity**
hypothesis, not an isolated encoder ablation or a known fix. Keep centered
CDP, action-effects exploration, optimization and temporal recipe unchanged.

This document initially authorizes only the following finite preflight, not
three long learners. After cost/finite-state review, declare three fresh
Breakout seeds1009/2017/3019 with a fixed interaction budget, exact frozen
controls and action/time curves. Do not use an initialization smoke as learning
evidence or launch all five games automatically.

## Preflight, excluded from learning evidence

- Qualified native6d38eea2, Meganeurac6376542, Bladee349cddf. No native rebuild.
  Upstreamfc3a2fb/49ec60a was reviewed: f32 cooperative acceleration/checked
  geometry and presentation-damage hints, not an identified small-CDP learning
  fix. Retain this runtime to compare capacity; historical evidence is not
  repinned. Larger-shape smokes are not a fresh full upstream gradient-parity
  claim. No new kernel, backend tuning or numerical tolerance changes.
- One fresh Breakout Size12M learner, seed1009:1,024 actual actions and the
  existing schedule's195 updates, N8/B8/T16/H15/R32, microbatch8, replay100000.
  `--cdp --cdp-centered --disagreement-scale 1`, cosine500, encoder6e-6,
  dynamics4e-4/base4e-5, warmup1000, AGC.3 and ac_grads=false.
- Native input, one GPU RGB64 resize; full18/sticky.25/repeat4/no reset no-ops,
 100000-frame artificial cutoff. Save exact initial and final weights.
- Review finite metrics/checkpoint tensors, encoder weight movement, counters,
  zero debt and sampled device/headroom before the next job.
- One frozen restore,1,024 actions, seed3,500,001,009; zero updates and exact
  saved tensor equality. These action-limited smoke tails are not a natural
  episode competence cohort. Retain all actions, rewards and boundaries.
- Production stage timings and memory give only a feasibility estimate. This
  short warmup-stage run cannot establish a sustained speedup or GPU utilization.

Each GPU process has a300-second bound; persistent user services use Restart=no,
KillMode=control-group, host guards, RTX5080 and>=2GiB sampled Vulkan estimated
headroom. GPU workers keep their normal CPU allocation; preparation/audit uses
one CPU,2GiB,zero swap. Record allocation warnings; stop/review actual API,
numerical,hard-fault or deadline failures. No NVML polling, recovery or retry.
The frozen check launches only after reviewing the training smoke. Additional
cost:1,024 training plus1,024 frozen actions/195 learner updates, not part of a
future learning comparison. Review before extending any budget.
