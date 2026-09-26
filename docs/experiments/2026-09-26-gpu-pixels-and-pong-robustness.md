# GPU pixels and a failed Pong robustness check

## Pong: fixed-protocol success is not robust competence

The historical root1009 final policy was evaluated with25% sticky actions,
retaining its original Large encoder, native886bae68 package, RGB64 observations,
N6/full18, sampled policy and no reset no-ops. No learning or input-resolution
change was allowed. Four matches per stream were required; all extra completed
matches and partial tails remain in the record.

| Cohort | Wins | Mean return | Cutoffs |
| --- | --- | --- | --- |
| First four per stream | 2/24 | −7.1667 | 0 |
| All completed | 3/31 | −8.3871 | 0 |

Both cohorts fail the existing90%-wins/mean≥15 thresholds. The guard completes,
76,464 actions have zero updates, all241 state entries/146 optimizer moments
are unchanged, and full ALE replay/video verifies. All38,266 native memory
samples pass, with minimum estimated Vulkan headroom7,804,157,952 bytes.

[Complete raw audit](../../runs/pong-sticky-evaluation-20260926.SxeHCw/seed1009-result.json),
[whole stream-zero video](../../runs/pong-sticky-evaluation-20260926.SxeHCw/seed1009.mp4),
[declaration and reader instructions](../../runs/pong-sticky-evaluation-20260926.SxeHCw/README.md).

This is one learner root and no new untrained-control pair. It establishes a
serious generalization failure for that policy, not its cause or a population
estimate. Preserve the earlier non-sticky results but stop describing them as
general Pong mastery. Further unchanged long campaigns are not justified by
the old wins alone. Roots2017/3019 remain undeclared.

## GPU preprocessing

[One Blade kernel](../../kindle/src/vision/preprocess.wgsl) now letterboxes,
bilinearly resizes, normalizes and packs channel-major patches directly into
the Meganeura encoder input. CPU Atari frames upload original RGB bytes.
[Resident input](../../kindle/src/vision/preprocess_gpu.rs) accepts borrowed
RGB8/RGBA8/BGRA8 buffers, byte offsets and padded rows on the same context.
Alpha is ignored. Submission ordering and buffer lifetime are explicit.

For one160×210 Atari frame, raw upload is100,800 bytes versus602,112 bytes for
the previous224²×3 F32 patches—83.3% fewer uploaded bytes at this boundary.
That is arithmetic, not an83.3% runtime saving or a measured whole-agent speedup.

The first candidate preserved intermediate RGB8 rounding. Native160×210 tests
passed, but64×64 exceeded the declared1% changed-value gate:9,912/451,584 values
differed. There was no device loss or recorded kernel fault. Preserve
[that failed invocation](../../runs/gpu-pixels-test-20260926.5xk0z7/README.md),
the preceding borrow-check/Clippy failures and their isolated sources.

The revised implementation removes the unnecessary intermediate RGB8
quantization, interpolates directly in F32 and uses exact normalized-zero
padding. Its **encoding revision is v2**: old v1 policies must use their pinned
old package, not silently restore with changed visual semantics. Encoder
weights/projection/pooling and world-learning equations are unchanged.

The [independent F64 reference test](../../runs/gpu-pixels-float-20260926.F6sEvQ/result.json)
passes40 comparisons across ten image shapes, including native Atari, RGB64,
224-square, one-pixel axes, portrait/landscape and640×480. Worst normalized error
is0.00016474724, under0.01 RGB8 levels. Resident/uploaded GPU results match
exactly; inactive streams are exactly zero. Odd offsets, row padding, alpha,
channel order and same-queue GPU producer/consumer handoff are exercised.

The [full synthetic Tiny encoder test](../../runs/gpu-pixels-tiny-20260926.ODQsqs/result.json)
passes37 exact-input dense comparisons (relativeL2≤9.1e-7, max absolute≤7.63e-6)
and36 ticks of independent-stream reset/gap/chunk tests. Batched/serial output
matches exactly. Dense numerical parity uses the exact normalized reference
input, separately from the independently checked preprocessing semantics.
All device/memory/guard assertions pass. CPU preparation passes95 Rust tests,
formatting and release Clippy. CI now also exercises resident preprocessing on
lavapipe and Metal; its outcome is separate from these local hardware results.

This is not yet a complete GPU-capture-to-action pipeline. Projected features
still cross the host boundary, capture import/semaphore ownership is not wired
to mind-games, and the new package still needs gameplay/restore qualification.
The old b00ce7be package remains the qualified historical gameplay runtime.
Measure end-to-end latency/throughput before claiming a speedup.
