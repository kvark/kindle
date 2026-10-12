# Split-convolution tile fix qualifies

October 12, 2026. [Declaration](../experiments/2026-10-12-split-convolution-tiles.md)
· [Data](2026-10-12-split-convolution-tiles.json)
· [Meganeura PR255](https://github.com/kvark/meganeura/pull/255).

The [Nsight diagnosis](2026-10-12-nsight-root-cause.md) now has a small upstream
fix and a refreshed, pinned integration. **Complete updates take 8.14% less time
for RGB and 6.17% less for CDP**, with exact numerical controls. Keep the change.
No game training, new profiler capture or learning-quality claim.

| Ordinary synthetic update | Refreshed control | Tile fix | Less time |
| --- | ---: | ---: | ---: |
| RGB12M / B8 / BF16 | 61.027 ms | 56.058 ms | 8.14% |
| CDP12M / B32 / mixed | 135.070 ms | 126.739 ms | 6.17% |

Each arm performs 16 warmup and 256 measured updates on the same fixture,
checkpoint and configuration, with no profiler, NVTX annotations or concurrent
compilation. RGB world training/sync drops 36.62→31.63 ms; CDP 54.34→45.94 ms.
Other stages are essentially unchanged. These are different RGB/CDP workloads,
not an equal-work comparison between representations. The prior JAX BF16
control is still about 4.5× faster than native RGB, with the already-disclosed
graph, weight, RNG and input-residency differences.

## What changed

An unsplit convolution gradient chooses a small scalar register tile to expose
enough workgroups. Split-K then adds parallelism but kept that small tile.
Reuse the existing occupancy rule with the partition count: widen a 16-wide
producer to 32 only when that grid fills the existing 64-workgroup target.
Keep low-occupancy cases and existing 32/64-wide choices unchanged.

No new shader, precision policy, partition count, temporary-buffer capacity
or summation order. The standalone main-based runtime patch `9d7caff` and
integrated `5a49332` have the same patch ID; later standalone `8a9baad` corrects
only an oracle test expectation and adds its local device/headroom assertion.
Existing cached plans remain safe but retain
their old selection until rebuilt. This is not proof that 32 is universally
optimal or an implementation of Tensor Core weight gradients.

## Qualification and provenance

- Fourteen standalone CPU selection/legality checks, formatting and Clippy pass.
  The GPU F64 derivative test now exercises 16-wide inputs and verifies that
  widening actually occurs; every split partial keeps its original numerical
  gate. Optimizer/alias checks pass on standalone and integrated builds.
- Refreshed BF16, convolution and row-addressing checks pass. All **17 GPU
  guards** pass, including the later oracle correction; four standalone
  allocation warnings remain (three original plus one follow-up). Expected RTX 5080,
  driver 580.178.04 and sampled Vulkan headroom ≥2 GiB pass. This is not peak
  physical free memory or an SM-utilization reading.
- Both pairs have **272 exact numerical reports**, exact first targets and
  all **292 RGB / 346 CDP initial, first and final tensors**. The refreshed
  no-tile controls also reproduce the earlier pinned fixtures exactly; this
  does not establish universal equivalence of the upstream refresh.
- Ten full-plan semantic comparisons permit only four RGB/two CDP shader/grid
  changes. Buffers, partition counts and other sessions are unchanged. The
  independent CPU audit passes in 96.65 seconds with 1.3 GiB peak memory.
- The git-pinned native build, configuration/revision checks, both Clippy targets
  and refreshed Python extension pass; all 193 runner tests pass. Both dependency
  locks agree. CI status is tracked on the PR dashboard, not inferred from local
  success. No GPU or preparation worker remains after publication.

Meganeura main `20af998` includes newer parameter encoding, measured-search /
layout support and GGUF mapping. Refreshed control `62fd7dc` preserves the
qualified BF16/row-addressing changes; pinned candidate `5a49332` adds only this
fix. The already-upstream hardware-row patch is dropped. Combined cache28
rejects the older BF16 cache27 parameter encoding. Blade `663afd2` remains based
on published main `7909fc1`; Naga stays `c9fd6a85a`. Neighboring user work and
historical pins are untouched. Only the user merges upstream PRs.

## Retained preparation and CI failures

A reused packaging script tried to copy an executing backend binary. The OS
refused with `Text file busy`; qualification completed using the unchanged
declared executable. The replacement builder uses a new, write-once output
and records the correct revision labels.

The first CPU audit expected full plans that ordinary mode does not export.
Four short annotation-only exports on the same binaries supplied them: **96
additional synthetic updates**, no injected profiler or replacement timing.
All 24-report prefixes and initial/first tensors match ordinary runs; short
final tensors match between control/candidate. Both failed logs/sources remain.
Neither failure is a GPU fault or grounds to repeat the completed timing pairs.

[Upstream CI974](https://github.com/kvark/meganeura/actions/runs/38265565595)
fails one stale oracle shader-selection expectation on both Metal and lavapipe;
the other163 oracle tests pass. The large selected-split fixture now correctly
uses32, while the smaller fixture keeps16. Test-only `8a9baad` corrects those
exact expectations without relaxing the F64 gates. The guarded5080 rerun
passes in2.68 seconds, with one allocation warning and the already-reviewed
VUID10684 retained. Both failed CI logs and the correction patch remain.
Production pins and complete-update measurements are unchanged; fresh CI
status is kept on the PR dashboard.

## Next

Continue with efficient reduced-precision convolution derivatives and
matrix/grouped-RSSM costs, then posterior reuse and device-resident targets/EMA.
CDP imagination now costs 54.22 ms versus 45.94 ms world training: optimize
whole-agent cost, not only the world model. The JAX gap, clean Nsight teardown
and learning/time-to-score remain open. Training stays stopped; no old
allocation resumes.

Artifacts: `runs/split-conv-tiles-20261012.exkYhrAF`. Source, binary hashes,
all guarded runs, failed preparation and exact audits are retained there.
