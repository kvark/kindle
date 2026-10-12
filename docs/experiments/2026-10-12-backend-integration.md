# Final pinned backend performance check

Declared October12,2026 before final GPU execution. Integrate only useful
changes: BF16 dense/convolution storage, repeated-row addressingPR249 and
hardware-row convolution selectionPR252. PR248's unsuccessful weight-gradient
kernel stays out. CDP uses the qualified fixed `mixed` policy; RGB uses `bf16`.
Auto/default and live acting are unchanged. Remove the experimental dense-only
CLI mode; its completed evidence/source/binaries remain, without checkpoint
compatibility scaffolding.

Upstream advanced during the round: Meganeura maineb8c90f adds Qwen/RoPE and
attention-mask/cache fixes, Blade main7909fc1 serializes compilation only on
Qualcomm Android. Rebase our BF16 and row-addressing PRs without merge commits;
preserve pre-refresh branch tips. BF16 cache27 rejects pre-refresh branch
formats. Final Meganeurad68f1a5 stacks PR250/fd9608f, row addressing and hardware-row
selection on that main; Blade663afd2 retains BF16 on7909fc1, Naga forkc9fd6a85a
is unchanged. Do not repin historical evidence or attribute old failures to
these unrelated upstream changes. Only the user merges.

## Qualification then timing

1. CPU precision/strict-region, convolution selection, cache and shader tests;
   row-copy compilation test, Rust formatting/Clippy, both Kindle locks/revision
   labels and Python CLI tests. Use git pins, no Cargo path overrides.
2. Four serialized120-second native processes: all nine BF16 checks, the16
   independent BF16 convolution/gradient cases, the scoped F16 hardware-row
   forward/input checks, and independent row-addressing values/gradients. The
   historical unchanged scalar weight-oracle failure stays explicit, not rerun
   or erased. Require5080,>=2GiB sampled headroom and positive test markers.
3. Ordinary native full updates, each16 warmup+256 measured and600 seconds:
   retained pinned679ea87/b52965b F16 control versus integrated CDP12M/B32/mixed,
   and the same control versus integrated RGB12M/B8/bf16. Check exact initial
   346/292 tensors, finite reports/targets/initial-first-final checkpoints,
   actual precision selection and comparison with the preceding candidates.
   This measures the combined stack, not additive isolated speedups or identical
   stochastic trajectories. Same workload/model/replay/checkpoint/fixture RNG.
4. Repeat the existing JAX RGB12M/B8 FP32/highest and BF16 controls, same upstream
   pin/environment and16+256 work. Matched shapes, not identical weights/RNG/graphs.
   Report the remaining gap and world-stage tradeoff; neither timestamps nor
   ordinary latency are SM utilization. No learning or time-to-score claim.

No compilation or other GPU jobs during matched timings. Bounded non-restarting
systemd and the existing host guard serialize all work; record allocation warnings,
stop/review other failures. No NVML polling, recovery, game actions or old queue
resume. Artifacts `runs/backend-integrated-20261012.xWfA9jYb`.

Useful mixed precision need not mean BF16 everywhere: the measured CDP world
gradient path retains F16 inputs/F32 storage. Broader CDP world/backward reduced
storage and the JAX speed target remain future work, not achievements to infer
from this round. Finish the integration review before considering fresh learning.
