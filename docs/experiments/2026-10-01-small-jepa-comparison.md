# Compact causal Tiny learning comparison

Declared before new JEPA gameplay, after the [small RGB replication](2026-09-30-small-replication.md).
Start only after its three upstream/native seed pairs pass their completion audits.
This is the separate Phase 2c study, not more replication or a restart of the
cancelled 45-run matrix. The replication cap remains ten attempts, including
its retained interruption. This study adds **six training runs on one game**;
it reuses all three unchanged native RGB controls, not a favorable subset.

## Fixed comparison

- **Seaquest**, held out from Tiny's pretraining titles; seeds **1009/2017/3019**.
  Fresh pretrained Tiny and its own initial-weight Tiny per seed. Both frontends
  stay frozen. No Large arm, new pretraining, intrinsic/shaped reward,
  privileged policy observation, action aid, restore or curriculum.
- Same Size1M/F32 learner, **N8/B8/T16/full BPTT/M8/H15/R32**, 200,000 aggregate
  actual actions, replay capacity 100,000 arrivals/context1, LR4e-5/warmup1000,
  AGC.3, actor unimix0/RSSM unimix.01. Resets earn no action/update credit.
- Full18 actions, sticky.25, repeat4, no reset no-ops, cutoff100,000 executed
  frames. Stream seeds are learner seed + stream×1,000,003 modulo2³².
  Separate causal histories, recurrent states and resets for all eight streams.
  The encoder's 16-arrival chunk boundary does not reset the RSSM/environment.
- Native max-pooled RGB210×160 enters the existing GPU causal Tiny frontend;
  preserve detail through its 224-pixel preprocessing, never RGB64→224.
  Frozen observations are 7×7×64 (existing projection/pooling). Reconstruction
  weight0 and causal future-prediction weight.25; every other learner setting
  matches the RGB control. RGB uses its explicit single RGB64 resize and
  reconstruction weight1/future-prediction0. Thus RGB versus Tiny compares
  **whole observation/objective packages**, not just pretraining. Pretrained
  versus initial Tiny changes only the frozen weights.
- Tiny is ~5.5M parameters, not the 303M Large model. Pretrained SHA256
  `7fe9b25287b2bff6e0555a74ad2fc7ee9f0e5513fc465475040b082ad2907c5b`;
  its own initial weights `7bc344f316d2258bfd429728da26cd26cc36814aedca620c72719ba6bd8cceda`.
  Both have149 matching tensor shapes. Tiny previously saw **250k random-play
  RGB64 frames** from Boxing/Pong/Freeway/Breakout/Qbert (45k train +5k validation
  each). This is additional experience, not online-only learning.
- Reuse the qualified native package, SHA256
  `65fc89696b06ecdaf49c072c4cd5dd7c6459352e353803bf7eaf37303c3a2c22`,
  Meganeura `75d08173` / Blade `7cca6377`. No simultaneous learning/backend
  changes or local compilation during timing.

Run order alternates treatment order across seeds: pretrained1009, initial1009,
initial2017, pretrained2017, pretrained3019, initial3019. Launch **one at a time**
only after reviewing the previous guard, complete trajectory/counter ledger,
checkpoint finiteness and optimizer steps. No automatic retry/successor.
Retain failures and review them before any amended declaration.

RTX5080/driver580.178.04; persistent guarded systemd user services, two
CPU-equivalents/12GiB host memory/no swap, **one-hour hard deadline per run**.
Require the native device and ≥2GiB sampled Vulkan estimated budget headroom.
Headroom is not physical free or peak memory. Separate NVML polling remains
off; GPU utilization is unmeasured. Keep live logs local.

## Evidence and decision

Publish all episodes/tails, initial/final online windows, score versus actions
and wall time, three-seed bootstrap intervals and paired final-score differences.
The existing auditor requires all three paired seeds; no interval from episodes
or incomplete method pairs. Report construction separately, whole-run actions/s,
actual aggregate/per-stream real time, and measured learner stage times. World
training, posterior and imagined rollout times stay distinct; stage wall times
are not hardware utilization. Include sampled memory headroom and parameter
counts, with the pretrained encoder's extra capacity disclosed.

A frozen frontend must earn its whole-agent cost through useful probes and
learning versus learned/random controls. Read the learning/cost tradeoff at
matched interactions and within common measured time support; no mastery gate
or automatic claim from one favorable seed. If a useful benefit is not
established, prefer learned RGB for this 2D screen and retain causal Tiny as an
explicit 3D/video hypothesis. Inconclusive small results are not proof that JEPA
cannot work. Do not tune another variant on these six results before recording
the decision and its limits.

Phase 2 closes only after these results are published and the evidence-backed
frontend decision is implemented and verified. Results alone are not closure.

Artifacts: `runs/small-jepa-comparison-20261001.oD7gNW/`.
Status stays in [PR31](https://github.com/kvark/kindle/pull/31).
