# Spatial caching is correct but makes the convolution family slower

October15,2026. **Reject the halo cache.** The complete forward/input-gradient
family takes **13.706→18.598ms GPU,35.69% slower**, including original packing
and output handling. Queued wall time is13.744→18.638ms. All13 distinct shapes,
15 actual occurrences and all128 samples/shape remain; the declared10% gain
gate fails. No full-update trial, selector/geometry sweep or integration follows.

[Declaration and coverage repair](../experiments/2026-10-15-convolution-halo.md),
[compact result](2026-10-15-convolution-halo.json). Raw sources, binaries,
per-call samples, plans, guards and failures are under
`runs/convolution-halo-20261015.kK1iZn4z`; qualified family data is in its
`contract-repair/` subdirectory. Production and the isolated076f3135 candidate
stay unchanged; no game actions or learner updates.

## What this isolates

The [preceding address proof](2026-10-15-convolution-matrix-feasibility.md)
covers20.070M addresses and reduces logical image requests9.461→1.642GB.
This trial preloads each tile's surrounding pixels into typed shared memory,
then redirects the original image reads there. It keeps original32/64 tile
geometry, K order, matrix arithmetic, A/B staging, output scratch/scatter and
existing barriers, adding one initial preload barrier. All eight weight-gradient
products and compensation are unchanged. Shared storage fits47,104/49,152B.

One generic shape/resource admission check handles F32/F16/BF16; unsupported
geometry, dW and excessive shared storage retain the original kernel. Two CPU
tests, typed shader validation/SPIR-V generation and Clippy pass. Source review
confirms that the original generator supplies the matrix/output program; only
image reads, declarations and initial loading are changed. The three backend
files live in an isolated copy, not the production dependency.

Numerical qualification passes503,808 direct F64 outputs, with zero pointwise
failures and bit-exact original/cache pairs at unchanged gates. Changed-input
and restored-input checks pass. At B512, all257,949,696 final output pairs match
exactly. Same-binary original/cache, actual shader/grid/packing contracts and
unchanged plans pass. These scoped checks are not whole-model or learning parity.

## Complete-family result

Times include transforms/output handling, not just the heavy shader. The two
64-channel8×8 cases each occur twice; all others once.

| Operation (forward Ci→Co, image size) | Original ms | Cache ms | Change |
| --- | ---: | ---: | ---: |
| Forward3→32,64 | .589 | .720 | +22.1% |
| Forward32→48,32 | .842 | 1.071 | +27.2% |
| Forward48→64,16 | .373 | .478 | +28.3% |
| Forward64→64,8, twice | .117 | .147 | +25.6% |
| Forward64→48,16 | .372 | .481 | +29.3% |
| Forward48→32,32 | 2.359 | 3.172 | +34.4% |
| Forward32→3,64 | 3.882 | 5.437 | +40.0% |
| dX32→48,32 | 2.578 | 3.840 | +49.0% |
| dX48→64,16 | .369 | .476 | +29.0% |
| dX64→64,8, twice | .118 | .148 | +24.8% |
| dX64→48,16 | .381 | .486 | +27.7% |
| dX48→32,32 | .747 | 1.128 | +51.0% |
| dX32→3,64 | .742 | .718 | −3.2% |
| **Weighted15 occurrences** | **13.706** | **18.598** | **+35.7%** |

Only the last case improves slightly. No favorable-subset promotion. Fewer
logical image requests do not by themselves reproduce the library's advantage:
extra shared reads/address conversion, preload/barrier work and increased
resource pressure can offset them. Their individual contributions, actual
DRAM traffic and occupancy are **not measured** here. This result rejects this
cache, not spatial reuse in every convolution implementation or Vulkan parity.

## Timing and provenance limitations

The original B1 harness compiled different kernels because runtime channel
packing has workload thresholds. Both GPU guards pass, but CPU coverage review
rejects that allocation after the first131,072 F64 checks. Keep the original
source, declaration snapshot, outputs and failed review. The separately declared
repair loads the actual retained B512 schedules, shortens only image batch and
matching pack/grid counts, and locks that schedule. It preserves allocated
buffer capacity, weights, types and output handling; no backend or tolerance
change. All actual contracts then pass before ordinary B512 timing begins.
Two earlier CPU build-receipt path failures also remain. A distinct build cache,
compiled-source receipt and actual old/new pipeline assertions establish which
code ran; a binary hash alone would not.

Two isolated checked kernels again change substantially within the retained
samples. Original forward48→32 first/last16 means4.050/1.502ms; its dX32→48
counterpart5.583/2.151ms. Candidate counterparts4.887/2.715 and5.719/3.345ms
also shift. No samples are removed or relabeled steady state, and the original
acceptance gate remains failed. The cause is not established. NVIDIA documents
[background shader optimization for DX12](https://developer.nvidia.com/blog/?p=67605),
which is not evidence that it explains these Vulkan results.

A CPU-only audit of the latest retained complete updates finds no comparable
shift: RGB first/last32 measured means54.260/54.198ms, CDP47.738/47.731ms.
All256 measured samples remain. Thus this observation does not explain away
the retained54.245/26.632ms RGB/JAX gap or replace either ordinary timing.

Six GPU guards pass, two new allocation warnings retained, no recovery/NVML
polling.3,952 operation calls/8,720 operation shader dispatches include the
rejected-coverage allocation; initialization/housekeeping are not counted as
convolution operations. Minimum sampled estimated headroom15,828,123,648B,
not peak/physical free VRAM. Timing service finishes08:42:42UTC in226.440s;
no worker remains. Independent publication seals and warning-schema audit pass.

## Next

Refresh the **latest unmodified native whole-update trace** against the retained
actual JAX trace before another kernel change. Earlier native captures predate
several whole-program improvements; family timings and bypass interventions
cannot be added into a full-gap budget. Reconcile execution spans, host work,
dependency waits and convolution/matrix/other device work at the same scope.
Source review also finds that CategoricalGpu submits Blade work outside
Meganeura's coarse GPU trace. Uncovered intervals therefore include that GPU
work and cannot all be labeled CPU/idle; any refresh must disclose this gap.
Keep asynchronous library staging/address-generation differences as hypotheses
to test, not a promise that copying one kernel feature closes the gap. No
geometry/precision sweep, compiler feature project or Atari restart is declared.
