# Direct native counters confirm excessive convolution instruction work

October 16, 2026. [Readiness declaration](../experiments/2026-10-16-native-perf-sdk.md)
· [Capture declaration](../experiments/2026-10-16-native-counters.md)
· [Compact data](2026-10-16-native-counters.json).

**The supplied Perf SDK works on the RTX 5080 and driver 580.178.04.** Direct
in-application Vulkan capture now measures actual Meganeura kernels, without
Graphics injection, a CUDA translation, clock changes or a driver upgrade.
All 20 contracts covering 23 convolution occurrences pass capture and audit.

The main CNN1 derivative kernels execute **15.94×/9.04× more warp instructions
than the retained full-JAX library kernels**, despite fewer tensor instructions.
This replaces the previous CUDA-proxy uncertainty with native evidence of a
large instruction/data-feeding overhead. It is not a 9–16× latency prediction.

| CNN1 B512/CI32/CO48/HW32/K5 | Native dX | JAX dX | Native dW | JAX dW |
| --- | ---: | ---: | ---: | ---: |
| Warp instructions, millions | 1,554.07 | 97.49 | 1,207.88 | 133.57 |
| Tensor instructions, millions | 9.96 | 13.11 | 13.11 | 16.93 |
| Warp instructions / tensor instruction | 156.01 | 7.44 | 92.15 | 7.89 |

Native values are range-profiler counters; JAX values come from its retained
[actual full-program Nsight Compute capture](2026-10-15-library-convolution-feeding.md).
Warp totals use the same named metric, independently checked against the raw
JAX export. Native tensor-pipe counts and JAX SASS tensor-opcode counts use
different collection interfaces. These are matching operation shapes, not
identical input fixtures, layouts, output precision or ordinary timing contexts.
The dW comparison excludes its separately captured final reduction.

## What the whole family shows

| Existing native implementation | Occurrences | Warp instructions | Instructions / tensor instruction |
| --- | ---: | ---: | ---: |
| Packed convolution | 14 | 1.596 billion | 21.17 |
| Scalar-staged BF16 forward | 3 | 3.347 billion | 85.12 |
| Scalar-staged typed derivatives | 6 | 6.867 billion | 104.79 |

These are occurrence-weighted heavy-kernel counters, not full-update costs.
The nine scalar-staged occurrences account for most instruction work; simply
putting more matrix arithmetic on tensor cores does not remove their surrounding
addressing, staging, conversion and fragment-handling work. Pipeline counters
do not individually identify which source expression produced an instruction.

Native CNN1 dX/dW measure **5.63%/5.67% of peak DRAM throughput**, and
29.51%/15.29% of peak L2 throughput, in these captures. They also report
122.08M/157.98M shared-bank conflicts. External-memory bandwidth saturation
does not explain these two captured kernels; memory latency and on-chip feeding
remain relevant. Empty ranges retain 0–1,024 DRAM bytes, so even measured traffic
is not assumed to contain no profiling overhead or automatically subtracted.

ALU-pipe instructions are 47.71%/42.02% of their totals. FMA-pipe counts are
not a count of floating-point math alone. Stall counters show several substantial
causes, including long-scoreboard, fixed-latency wait and, for dW, short-scoreboard
waits. Shared conflicts are now observed, but their removable latency is not
isolated. Neither stall fractions nor instruction counts are additive time savings.

## Decision and remaining work

Keep isolated `67559ce5`; no production change or Atari restart. Complete
RGB12M/B32/T16 remains **51.945ms / inclusive JAX27.223ms, 1.91×**. Counter
captures are not complete learner timings, and convolution alone is not yet
proven to explain the whole residual. Host targets, connected intermediates and
dependent non-convolution execution remain separate costs.

The next implementation contract must reduce scalar work per matrix product
and improve cooperative operand staging together. Start with a CPU proof of
type-aware word/vector transfers and shared-bank layout, preserving existing
tiles, K order, compensation, rounding and external interfaces. Use these
native counters as the baseline; check the resulting instruction/transaction
change as well as complete-operation latency. Padding alone is not a promised
fix. Qualify the complete family before any integration/full-update claim.
No new tile/geometry search, revival of rejected policies, favorable-shape
selector, relaxed numerical gate or speculative Naga feature project.

## Qualification and accounting

- 21 passing guards, zero new allocation warnings or faults: one empty-session
  readiness process and 20 complete-operation capture processes. All workers exit.
- 680 counter replay passes, 1,680 total standalone operation calls, zero game
  actions/learner updates. Capture queue finishes **04:39:04 UTC in 3m09.109s**.
- 58 ranges and 2,378 metric values reevaluate exactly in separate CPU-only
  processes from saved counter images. This checks persistence/evaluation, not
  an independent implementation of NVIDIA's hardware metric formulas.
- Every range has its exact expected workgroup count; empty ranges have zero
  launches/instructions. No dropped ranges/trace bytes; all 34 passes per process
  collect. Full retained plans/pipeline keys and **258,287,296 output elements**
  match exactly, including changed-input/restoration checks and final outputs.
- All 1,794 capture-source seals and 29 readiness seals pass. Only isolated
  Blade handle accessors and the corresponding dependency path differ; shader
  code, compiler options, selections and numerical rules remain unchanged.
- Minimum sampled Vulkan estimated headroom is 15,828,123,648 bytes, not peak
  physical free memory. CPU audit finishes04:39:52 in5.320s/989.3MiB; no GPU retry.

Artifacts are under `runs/native-perf-sdk-20261016.ZVAi5Ubl`: package manifest,
readiness receipt, sealed reviewed queue, per-contract raw counter images,
offline evaluations, full outputs, `audit.json` and `interpretation.json`.
The vendor SDK/license remain untracked. SDK initialization recognizes GB203;
the device API-version support flag is false, with the SDK's documented
unknown-version fallback; every called API succeeds. No claim of official
support for that Vulkan version is made.

Pre-launch review corrected asynchronous decode waiting, retaining the first
bridge/build manifest before any workload executed. No GPU failure or retry
occurred in this study. The first CPU publication check used `sha256` instead of
the package manifest's `archive_sha256`; its source/failure remain, with only
that field-name correction and no GPU repeat. Earlier Graphics-injection failures, unavailable standard
counters, numerical cancellation failures and rejected kernels remain unchanged.
[NVIDIA's SDK limitations](https://developer.nvidia.com/nsight-perfsdk/getting-started/release-note-v2026.3)
explain why markers stay on primary command buffers and clock-setting APIs are
never invoked. Separate NVML polling remains off. CI440 passes preceding887f499;
this diagnostic does not promote a new learning runtime.
