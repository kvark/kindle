# Centered CDP capacity: existing12M versus retained1M Breakout

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

The initial declaration covered only the finite preflight below. Its reviewed
success now permits the separately specified three-seed allocation at the end
of this document. Do not use the smoke as learning evidence or launch all five
games automatically. The preflight's original plan is retained in its run root
as `preflight-plan.md` and in commit6d63d55.

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

## Preflight outcome, October7

Both guards and independent CPU audits pass without new warnings. Training
finishes1,024 actions/195 finite updates in22.23s, zero debt. All12 CNN parameter
arrays move. Frozen restore executes1,024 actions in.604s with zero updates
and exact346 tensor equality (world269/behavior66/slow11). Eight host workers
are retained. Native and all actor settings except model preset match the
retained1M control. [Compact evidence](../results/2026-10-07-cdp-12m-preflight.json).

Last128 updates average108.98ms: imagination60.63ms, world training32.45ms,
posterior10.00ms. This projects about3h50m per500k learner; it is a short
warmup-stage estimate, not sustained performance or a matched speed claim.
Minimum sampled Vulkan budget-minus-usage is12,552,241,152bytes, not physical
free/peak VRAM. The preset has16,334,353 actual trainable parameters here
(world14,688,256 including CDP/exploration, behavior1,646,097), not exactly12M.
Raw artifacts:`runs/cdp-12m-preflight-20261007.WrHwG8YN`.

## Reviewed learning allocation

- **Three fresh Size12M Breakout learners**, seeds1009/2017/3019,500,000 actual
  actions/124,939 updates each:1.5M new actions/374,817 updates total. Do not
  resume the smoke or reuse it as a fourth learning replicate.
- Keep the preflight's qualified native and entire centered-CDP recipe above.
  Only the model preset changes from the completed1M allocation. This is whole
  model capacity, not isolated CNN capacity, loss, context or exploration.
- Retain all three completed1M500k seeds as controls, using the reconciled
  natural-cohort evidence—not the invalid old Breakout3019 summary. Compare
  configurations exactly apart from model size. Acting-source changes since
  that campaign only repaired natural-episode accounting; disclose v4/v5 and
  the repaired control's8->1 host workers. No retroactive source repinning or
  claim that models share initial tensors across incompatible shapes.
- Save actual initial12M checkpoints before any experience. Audit each
  completed learner, then final and actual-initial frozen controls before the
  next learner. Same evaluation base4,000,000,000+seed as the retained1M
  controls; same per-stream offsets, sampled policy, first3 **natural**
  episodes/stream,600k-action/30min cap. These are reused development-evaluation
  seeds for a paired capacity comparison, not a new untouched test suite.
- Require explicit `cohort(..., natural_only=True)`, zero evaluation updates,
  exact346 frozen tensors, all CPU action/reward/reset replays and full
  stream-zero videos. Retain cutoffs, excess episodes and unfinished tails;
  incomplete natural cohorts stop the controller for review, not seed removal.
- Report all online last50 curves versus actual actions/time and the frozen
  final/initial/control means with three-learner-seed bootstrap intervals.
  Preserve every seed and failed/extra compute. Compare online windows near2M
  frames separately from the unchanged381.811 long-run reference; no parity
  or compute-saving claim from different budgets/protocols.

Expected roughly**12hours** including evaluation. Each training process has a
5h deadline; frozen processes30min. One24h-bounded persistent service with the
same guards/ownership/warning policy and no retries; CPU audits/replay remain
one CPU,2GiB,zero swap. This is three learners plus six frozen controls, not a
new representation matrix. No other game, capacity, seed or budget is added
automatically. At the finite end review score improvement and its additional
cost before any extension; all five original quality targets remain open.

Learning artifacts:`runs/cdp-12m-breakout-20261007.3msA0ITD`.
