# Centered CDP: one longer allocation across all five Atari games

The [three-seed Pong comparison](../results/2026-10-06-cdp-centered-learning.md)
and [paired world probes](../results/2026-10-06-cdp-centered-world.md) justify
keeping centered prediction. The [bounded throughput trial](../results/2026-10-06-cdp-imagination-throughput.md)
passes:8.12% less full network-update time with qualified grouped B128 products.
Return to learning, not another optimization or representation matrix.

The objective remains **DreamerV3 quality on Boxing, Pong, Freeway, Breakout
and Qbert**, including the budget question. This allocation tests longer useful
learning and whether the centered recipe transfers to the other four games.
Completing it, beating initial controls or solving a subset does not finish
the objective. The unchanged published long-run targets remain99.613,20.445,
33.399,381.811 and193,220.767 respectively.

## Fixed allocation

- **Fifteen fresh learners:**500,000 aggregate actions/124,939 updates each;
  7.5M new actions/1,874,085 updates total. No checkpoint lifetime resume.
- Seeds1009/2017/3019. Seed-major order, with Pong/Boxing/Freeway/Breakout/Qbert
  within each seed, gives early breadth without dropping weaker seeds.
- Native`6d38eea2`, Meganeura`c6376542`, Blade`e349cddf`. The qualified grouped
  path changes execution, not capacity, initialization, rates or objectives.
- Common `--cdp --cdp-centered`, action-effects coefficient1, Size1M/N8/B8/T16/
  H15/R32, microbatch8, full BPTT, replay100000. Full18/sticky.25/repeat4,
  no reset no-ops,100000-frame episode cap, native observations and one GPU
  RGB64 resize. Cosine500, encoder6e-6/dynamics4e-4/base4e-5, warmup1000,
  AGC.3 and ac_grads=false. No per-game settings, action hints, labels,
  externally shaped rewards or video pretraining.
- Save each actual zero-action/zero-update initial checkpoint before learning.
  Audit training counters, zero debt and finite final tensors before evaluation.
- Frozen final and actual-initial control: sampled policy, first3 natural
  episodes in each of8 streams, held-out base**4,000,000,000 plus learner seed**,
  stream offset1,000,003 modulo2^32. Same seeds within each pair.
  Ceiling600k actions/30min per model, increased from the earlier200k ceiling
  to allow longer-surviving actors; the selected first24-episode cohort is
  unchanged. Retain excess episodes, artificial cutoffs and unfinished tails.
- Zero evaluation updates, exact346 frozen tensors/model, full CPU trajectory
  replay and whole stream-zero videos. A capped cohort is retained and stops
  the controller for review; no complete-cohort claim or automatic retry.
- Same exercised job/audit helpers as the prior study, explicitly hash-bound.
  Rehearse training audit on the actual changed-native smoke and validate all
  45 commands before launch. These are15 training runs plus30 frozen controls,
  not a revived45-learner matrix.

Expected roughly**16–18 hours** including evaluation/audits. One persistent
24h-bounded service,100min per training process,30min per frozen process,
Restart=no, KillMode=control-group, serialized host-guarded GPU jobs. Require
RTX5080 and>=2GiB sampled Vulkan estimated headroom. Record allocation warnings;
API/numerical/hard faults/deadlines stop for review. No NVML polling, recovery
or blind retries. CPU replay/analysis gets one CPU,2GiB and zero swap.

## Readout and next decision

Retain every seed's online action/time curve, actual frames/updates/elapsed
time, final frozen-versus-initial return and world/reward diagnostics. Report
learner-seed bootstrap uncertainty, not episode/stream pseudo-replication.
Compare online curves to all released DreamerV3 seeds near the actual frame
count (500k actions are approximately2M frames) separately from the unchanged
long-run targets and frozen controls. Report prior experiments and repeated
fresh prefixes as additional compute.

This is not a fresh raw/centered or RGB comparison across five games: centering
has controlled evidence on Pong only, and the optimized execution/backend
differs from older runs. No five-game causal centering or same-hardware RGB
compute-saving claim. Preserve earlier weak results and historical assistance.

At the finite end, review every game before allocating more. Persistently weak
reward/state prediction calls for diagnosis; useful but incomplete learning
calls for a reviewed budget/capacity/context decision, not an automatic extension.
Keep all five and their original quality targets in scope.

Artifacts:`runs/cdp-centered-five-20261006.ODmbym7M`.
