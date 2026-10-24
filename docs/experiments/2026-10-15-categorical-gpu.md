# Device-side categorical means

Declared October 15, 2026, before GPU execution. Root:
`runs/categorical-gpu-20261015.lkZAlUAS`.

Question: how much complete-update latency comes from transferring three
255-bin distributions to the CPU and decoding them there? The retained B32
profile assigns 3.752 ms to decoding and 1.632 ms to subsequent target
construction. This tests the first boundary, not the entire remaining JAX gap.

Use the unchanged isolated general-ordering backend and native `28864210` as
control. Candidate changes only Kindle's distribution decoding/readback:
one native Blade reduction per reward/value/slow-value head, reading existing
F32 logits and returning one mean per row. Keep categorical support exactly
mirrored and identical to the CPU's F32 bins. Use stable mirrored exponential
differences to preserve exact symmetric zero. Device arithmetic is explicitly
F32 instead of the CPU decoder's F64 exponential/accumulation. Target building,
normalizer, reward/terminal indexing, RNG, old-weight consumers and all network
graphs stay unchanged. No precision/geometry sweep or production promotion.

Qualification before timing:

- Independent CPU F64 means, both direct probability sum and paired sum;
  synthetic odd bin counts 3, 7, 255 and 511, varied/tail row counts, mirrored
  logits, extreme concentrated distributions, near-symmetric and random rows.
- Every GPU output finite; exact zero for mirrored rows; pointwise error at
  most `1e-4 + 2e-5 * abs(reference)` and relative L2 at most `2e-5` per fixture.
  Keep every failure, not just aggregate norms. No gate relaxation.
- Inputs/outputs canaries, repeated dispatch with changed and restored inputs,
  queue/readback ordering, no stale values or mutation of the input.
- One unchanged-CPU synthetic learner update captures actual reward/value/
  slow-value logits from the retained 272-update control weights (not just
  initially zero heads). Qualify all 24,576 means at B32/T16/H15 against the same
  reference/gates. This capture is not a timing sample.

Only after numerical review: one complete B32/T16 RGB12M control/candidate
pair, each 16 warmup + 256 measured synthetic updates. Retain all 272 reports,
first targets and initial/first/final checkpoints. Require identical initial
model tensors and finite outputs/updates, report first-target errors and final
tensor drift. Precision-induced later stochastic differences are not numerical
equivalence or learning evidence. A useful result needs at least 3% lower
complete-update mean; no microbenchmark-only speed claim. Existing JAX timing
is retained, not rerun. No game actions or learning campaign.

Each numerical/capture process is bounded to 180 seconds, each complete timing
to 300 seconds; sequential guarded jobs in a persistent systemd service with
Restart=no, KillMode=control-group and a 30-minute overall bound after CPU build.
Expected RTX 5080/driver580.178.04/current boot, at least 2 GiB sampled Vulkan
budget headroom, standalone allocation warnings retained. API/numerical/hard
failures stop the affected stage for review, no blind retry or driver recovery.
CPU build/review: one CPU, 2 GiB, no swap; no build during timing. Upstream
Meganeura `99e3de27` and Blade `e7c3f7b` rechecked unchanged. Production and
Atari stay stopped; the broader performance goal stays open.
