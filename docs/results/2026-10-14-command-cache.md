# Reusing commands removes a real host-side bottleneck

October 14, 2026. [Declaration](../experiments/2026-10-14-command-cache.md)
· [Data](2026-10-14-command-cache.json).

**Complete RGB updates improve 28.416→25.530 ms: 10.16% less time.**
All 272 reports, first outputs, compiled plans/pipeline keys and 292 tensors
at each initial/first/final checkpoint match exactly. Keep this isolated
candidate. It is still **1.99×** retained complete JAX's 12.836 ms, not parity.
Production remains on its earlier qualified ~53 ms runtime; no Atari resumes.

## The causal difference

The compiled graph was fixed, but Meganeura reconstructed its bindings and
commands every update. The prototype retains graph chunks in Vulkan secondary
command buffers. Dynamic optimizer parameters, step counters and clipping are
still recorded normally. Inputs and parameter contents change as before.

Nsight Systems confirms the removed work in every one of eight captured
updates per arm:

| Recorded CPU API calls per update | Ordinary | Cached |
| --- | ---: | ---: |
| Descriptor-template updates | 5,351 | 65 |
| Pipeline binds | 5,297 | 11 |
| Queue submissions | 15 | 15 |

Exactly 5,286 graph dispatches cease being re-recorded in both first rows.
**Those GPU dispatches still execute.** Eight chunks are recorded once:
five world-prefix/suffix chunks, one imagination, one behavior and one slow
value. Across 272 updates they are reused 2,168 times. No batch, precision,
operator, dependency or ordinary submission-chunk change explains the gain.

Selected CPU API duration falls 1.562→0.116 ms/update in this capture. This
does not account for all Rust/driver work or establish a device-side speedup.
Nsight perturbs control smoke timing by +7.85%, versus −0.17% for cached;
its 30.455/25.238 ms NVTX intervals are not the ordinary speed result.
Vulkan GPU tracing and GPU metrics are explicitly disabled. Some Vulkan
calls are not represented in this CPU-API trace; missing events are not zero
work. No SM-utilization or CUDA-graph-equivalent execution claim is made.

## Whole-update result

| Stage | Ordinary ms | Cached ms |
| --- | ---: | ---: |
| Posterior | 4.330 | 3.808 |
| Imagination/targets | 7.400 | 6.511 |
| World training/sync | 15.064 | 13.808 |
| Behavior training/sync | 1.622 | 1.403 |
| Total | 28.416 | 25.530 |

One ordinary 16-warmup + 256-measured pair, RGB12M/B8/T16/H15/BF16, four
world chunks, same binary disabled/enabled; no profiler, concurrent build or
timing repeat. Both shorter smokes and the separate CPU captures reproduce
all 24 reports, first outputs and saved tensors. The disabled control also
reproduces the retained control. JAX was not rerun.

## Lifetime and numerical checks

Sixteen cases cover F32/BF16, aliases/poison, Adam/LaProp, ordinary/staged
execution, changing inputs/late targets, parameter rebinding, chunk-count
changes and changing optimizer parameters. Each set compares 278,528 exact
values across 256 small updates. Rebinding uses separate donor allocations
for control and candidate; neither can hide the other's parameter errors.
No tolerance relaxation or alternative arithmetic is involved.

The first set requested Vulkan validation but omitted a Rust logger. It
remains numerical evidence, not a clean API-log receipt. A separately declared
test-only logging repair repeats the set; messages are retained, missing
layers/unreviewed errors are fatal, and the logged set passes. The unchanged
full-model binary is not rebuilt for that repair. Four Blade CPU command
tests and release Clippy for all three libraries pass.

Secondary encoders retain descriptor/scratch storage. Existing session waits
precede reuse/invalidation/destruction; changes to bindings, scheduling or
tuning invalidate the cache. All original barriers remain, plus conservative
chunk boundaries. Profiling keeps its original recording path. This Vulkan
prototype is not a portable API/production integration or CDP qualification.

Eight guards pass, four standalone allocation warnings remain. Total 1,152
synthetic updates, zero game actions: 512 small qualification updates, 48
smokes, 544 ordinary timing updates and 48 CPU-capture updates. No worker or
recovery remains. Retain initial visibility/Clippy failures, the zero-test
filter corrected to four tests, pre-build lock snapshot/declaration failure
and failed map-order audit. The repaired plan audit preserves every path/value
and array index, ignores map serialization order, and only normalizes the
retained compiler's omitted empty optional `extra_outputs` field.

## Decision and remaining gap

Artifacts: `runs/command-cache-20261014.rATLoIBZ`, including source patches,
locks, original/logged binaries, full state/plan audits and both Nsight reports.
Optimized 056562b-based backend, Blade96c1dc7/Naga a97559b; upstream
7574ad1/e7c3f7b was rechecked without mixing a full-main migration into this test.
Neighboring user work stays untouched.

Keep command reuse as a real architectural improvement. Do not turn it into
a cache/geometry sweep or infer the residual GPU cost from CPU timings.
Next match the remaining expensive GPU algorithms to JAX's actual executable,
starting with the five checked boundary convolutions already measured at
3.785 ms. The rejected scalar alternative does not settle their cause.
Library-kernel efficiency and the measured dependent-dispatch/CUDA-graph gap
remain separate problems; this result closes neither the whole performance
goal nor learning-quality qualification.
