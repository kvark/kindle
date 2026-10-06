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

## Prepared follow-up: frozen Breakout diagnosis

October6 review, **not launched**: the first two completed Breakout learners
score4.0/3.5 versus initial1.5/1.625, with late online means4.24/3.26.
Keep the entire current allocation unchanged. After its final audit and review,
diagnose Breakout before allocating more training to it. This does not close or
replace the other four games' quality targets.

Use the existing `probe_cdp.py --game Breakout` extension (commit7e5df42),
current native6d38eea2 and the **actual initial/final checkpoints of all three
500k learner seeds**, including3019 regardless of its eventual score. Compare
same-seed pairs; do not select favorable learners or checkpoints.

- First, one new-environment smoke on final1009:384 forced random actions,
  five GPU readouts with32 updates each. Review its guard, numerical outputs,
  sample alignment and exact346 frozen tensors before proceeding.
- Then initial/final1009, initial/final2017, initial/final3019. Each model gets
  the existing eight fixed development trajectories,4096 actions each:
  train9101/9109/9127, validation10103/10111, test11113/11117/11131.
  Full18/sticky.25/repeat4, no reset no-ops, native input/one GPU RGB64 resize.
- Read ball x/y and paddle x only for diagnostic labels. Mask absent balls;
  RAM never enters policy, replay or actor losses. Require identical action /
  reward/reset traces and sampled actual GPU-resized pixels across models.
- Fit the same five GPU readouts: resized pixels, CNN, posterior, prior h1 and
  h15. Existing hidden128/batch64,2048 updates/readout; train-only normalization
  and validation-only checkpoint selection, never test-set selection. Pixel,
  CNN and posterior comparisons share arrival indices. Report masked counts.
- Retain persistence, training mean, unrelated-action, zero-reward and matched
  privileged-position controls. Count sparse reward/terminal events, split
  posterior estimates from forecasts, and retain every trajectory. Raw cosine
  diagnostic distances are not the centered minibatch training objective.
- Total additional diagnostic collection:196,992 actions including smoke;
  61,600 readout updates, **zero actor updates**. Initial actors remain at
  step0 and trained actors at124,939. Exact346 tensors/model, hashes, finite
  outputs, per-model guards and independent CPU consolidation are required.

This probes random-action development trajectories, not held-out gameplay
competence or policy-optimality. Readable pixels but weak CNN implicate learned
features; readable CNN but weak belief implicates state learning. Useful state
with poor prior forecasts points toward dynamics; useful forecasts with weak
play calls for policy/credit analysis. These are diagnostic hypotheses, not
automatic architecture changes or proof that any one mechanism is responsible.

No native rebuild or learning extension. One serial GPU service,45min bound,
120s smoke/900s per full model, Restart=no, KillMode=control-group; the usual
device/headroom checks and record-only allocation warnings apply. Stop and
review actual failures; no automatic retries. Heavy CPU work remains one CPU,
2GiB and zero swap. Prepare exact runtime declarations only once all source
checkpoints exist; do not start a follower behind the current campaign.

Upstream rechecked October6 at19:50 UTC: Meganeurafc3a2fb adds f32 cooperative
matmul/attention-backward acceleration and checked dispatch/padding; Blade
49ec60a adds/fixes presentation-damage hints. No fix for the observed small-CDP
learning weakness was identified. The active allocation and its frozen probes
retain their qualified runtime. Review refresh/qualification before the next
learning change; do not modify neighboring user work or silently relabel runs.
