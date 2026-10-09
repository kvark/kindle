# Qbert: test experience before extending the five-game budget

The goal remains **DreamerV3 quality on Boxing, Pong, Freeway, Breakout and
Qbert with CDP**, retaining the RGB budget comparison. This allocation tests
one remaining uncertainty, not a narrower replacement for that goal.

At500k actions (~2M emulator frames), the completed12M Qbert cohort scores
1,790.63/1,328.13/4,855.21 frozen; only3019 completes a first pyramid reliably
(0/0/24 of24 natural episodes per seed). Its online mean2,600.33 is close to
the published2M-frame bin2,665.04, with differing protocols and averaging
windows. [Retained results](../results/2026-10-08-cdp-qbert-capacity.md).
More experience is a plausible next test, not evidence that it will fix the
weak seeds. The larger model also regresses on Freeway; keep that result.

## Fixed allocation and controls

- Three **fresh** Qbert learners, seeds1009/2017/3019 in order, each2,000,000
  actual actions/499,939 updates. Total6M actions/~24M emulator frames and
  1,499,817 updates. No restore/resume of the completed500k lifetimes.
- Unchanged12M centered CDP/action-effects coefficient1, N8/B8/T16/H15/R32,
  microbatch8/replay100000, cosine500, encoder6e-6/dynamics4e-4/base4e-5,
  warmup1000, AGC.3, ac_grads=false.16.33M actual parameters including
  exploration. No change to capacity, precision, exploration, loss or BPTT.
- Native pixels and one GPU RGB64 resize, full18/sticky.25/repeat4,
  no reset no-ops and100000-frame artificial cutoff. No action hint, RAM input,
  detector, external shaping or video prior. Task observers run only in CPU
  replay, never in the policy or training reward.
- Qualified Meganeura31026833/Blade56f0565/native7311547d. The
  [backend refresh](../results/2026-10-09-meganeura-refresh.md) passes without a
  speedup. [T64](../results/2026-10-09-cdp-sequence-preflight.md) misses its
  efficiency gate; the [cost capture](../results/2026-10-09-cdp-imagination-cost.md)
  adopts no optimization. Do not retry the rejected projection rewrite.
- Retain all three12M500k controls and all1M controls with their original
  pins. Current versus retained12M learning has a qualified backend refresh,
  so this is not an exact same-binary budget ablation. No retrospective repin.

Audit each completed learner's finite metrics/checkpoints, exact recipe,
499,939 updates, zero final training debt and all per-stream replay credit.
Save actual zero-experience weights before acting. Final checkpoints do not
preserve live replay/RNG/belief; interruption cannot be called an equivalent
mid-life resume. Keep every failed/interrupted stage and its compute.

## Frozen evaluation and decision

After each learner, before the next, evaluate final and actual-initial models
with sampled policy, eight independent streams, development seed base
4,000,000,000+learner seed, and first3 **natural** episodes per stream. These
are reused development seeds, not an untouched final test. Each frozen worker
has600k actions/30min maximum; incomplete cohorts stop for review. Explicit
`cohort(..., natural_only=True)`, zero updates and exact346 saved tensors are
required. Keep cutoffs, excess episodes and unfinished tails, but never select
cutoffs into the natural cohort.

CPU replay checks every action/reward/reset/frame and writes whole stream-zero
videos. A second CPU review independently reselects natural episodes and reads
first-pyramid completion from their replay outcomes, not positive-return counts.
Automate these checks and the final three-seed summary within the service.

Primary outputs are all three frozen final/initial scores, paired gains over
retained12M500k, first-pyramid success counts, and score-versus-actions/time
curves with learner-seed bootstrap intervals. Retain the500k prefix and all
subsequent online reports; do not select favorable checkpoints or average
streams as independent learners. Report measured training/evaluation/extra
diagnostic compute separately.

The historical intermediate gate stays>=90% first-pyramid completions and
mean>=15,000 **per learner seed**; it is not the full quality target. The
published8M-frame online mean14,607.50 is descriptive context, not a new target
or frozen comparator. Qbert's long-run193,220.77 and the other four long-run
targets stay unchanged. A gain without reliable task completion is partial
progress. If weak seeds remain weak, review their learning/forecast evidence
before another allocation; no automatic budget extension or mechanism sweep.
This allocation cannot establish five-game parity or RGB compute savings.

## Operation

Expected47–48h including evaluations at the measured current cost. Each learner
has20h maximum; one72h persistent systemd user service, Restart=no,
KillMode=control-group, zero swap and serial `gpu_host_guard.py` workers.
Require RTX5080/driver580.178.04 and>=2GiB sampled Vulkan estimated headroom;
that is not physical free/peak memory or utilization. Record standalone
allocation warnings. API/numerical/hard-fault/deadline failures stop for review;
no blind retries, separate NVML polling or recovery operations.

GPU workers keep normal CPU allocation. Preparation, replay and audits use
one CPU/2GiB/zero swap; no rebuild during the allocation. Inspect training
about every30min or at a stage boundary, not every counter. Preserve the
qualified source/native/ROM/job identities before launch.

Artifacts:`runs/cdp-qbert-2m-20261009.992tJMEX`. The service stops after these
three train/evaluation pairs and their CPU reviews. The other four games stay
in scope for the goal but have no new allocation in this declaration.
