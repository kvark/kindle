# Move learner random draws onto the GPU without changing the sequence

The scalar-region trial barely changes complete RGB updates. The remaining
host pipeline generates and uploads large uniform arrays for every posterior
and imagined step. Test that boundary directly, keeping the compiled learner
and qualified wider-control backend unchanged.

Artifacts `runs/device-rng-20261014.EZ1UNxiK`. Mechanically replay the
audited wider source snapshots, not its later barrier diagnostics or the
unpromoted scalar-region trial. Upstream Meganeura2214412/Bladefe5f5b5 remain
unchanged. No production pins/defaults, neighboring work or Atari allocations
change. This is one mechanism, not a new compiler/precision/batch sweep.

Use the pinned rand0.9.4 ChaCha12 stream: explicit rand_chacha0.9 state exposes
its seed, stream number and word position without generating the full host
array. A Blade compute shader fills the existing F32 session inputs directly.
Keep independent replay/live-policy/posterior/imagination RNG streams and
their consumption order. Check byte extents, tails, resource lifetime and
shared-queue ordering. No CPU learner workaround or alternate stochastic
algorithm. Native graphs, targets, optimizer and matrix precision stay fixed.

CPU checks compare explicit ChaCha12 with the existing StdRng at the same
seeds. GPU checks require exact random words and F32 conversion versus the
pinned CPU implementation across zero/irregular/large lengths, word offsets,
counter carry/wrap, nonzero streams and consecutive split fills. Preserve
canaries and demonstrate consumption by a following session without a host
wait. No statistical test replaces sequence equality.

After numerical review, two24-update RGB/CDP smokes, then one ordinary
16-warmup+256-measured wider-control/candidate pair per workload. Require
exact initial/first/final tensors, every metric and first categorical/target
output. Record whole-update and stage latency; no isolated RNG benchmark
speedup or utilization claim. Any mismatch/failure stops for source review
before follow-up. Keep failed binaries/receipts; no blind retries.

Serialize bounded GPU services through the host guard, Restart=no and
KillMode=control-group. Expected5080,≥2GiB sampled Vulkan headroom; allocation
warnings retained, separate NVML polling off. CPU builds/analysis use one
CPU/2GiB/no swap, not during matched timing. GPU workers have no CPU quota.
The broader root-cause/performance goal remains active; targets/EMA and
row/layout-level compilation are separate subsequent mechanisms.
