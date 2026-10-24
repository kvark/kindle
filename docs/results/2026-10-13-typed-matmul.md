# Shared typed cooperative matmul qualifies

October 13, 2026. **Adopt the consolidation, without a speedup claim.**
[Declaration](../experiments/2026-10-13-typed-matmul.md),
[data](2026-10-13-typed-matmul.json),
[Meganeura PR257](https://github.com/kvark/meganeura/pull/257).
Atari learning stays stopped.

## Implementation

Replace the BF16-only dense entry/shader with explicit operand storage,
multiplication and output types. F32 accumulation is fixed. One checked
cooperative template supplies indexing, transposes, staging, partition bounds
and final stores for F32/F16/BF16. Ordinary native16 F32/F16 cooperative
products and the existing BF16 convolution loop also use that template.
There is no per-element runtime type switch or implicit precision change.

Typed F16/BF16 products use the same initial partition rule. They now enter
bounded partition tuning: precision is in the search/pipeline key, scratch
uses the correct storage, references round operands appropriately, and the
following F32 reduction changes with the selected partition count. Output
rounding stays after that reduction. Live bindings are not enlarged.

This does not merge genuinely different algorithms. Fused/prologue/horizontal,
native8 and large-tile implementations remain; legacy F16 cooperative entries
outside the typed family are still excluded from measured search. This is not
universal tuning coverage, a tile sweep or a new precision policy.

## Complete updates

One ordinary pair per unchanged recipe, 16 warmup plus256 measured updates.
Same checkpoints, fixture RNG, model, batch and learning settings. No profiler
or compilation during timings; all learner stages and synchronization count.

| Recipe | Control | Shared typed kernel | More time |
| --- | ---: | ---: | ---: |
| RGB12M/B8/BF16 |52.803 ms|53.047 ms|0.46%|
| CDP12M/B32/mixed |124.997 ms|126.182 ms|0.95%|

Both pass the declared no-greater-than2% regression gate. One pair does not
establish statistical equivalence; neither improves speed. CDP world training
and synchronization45.889→46.915ms accounts for most of its measured difference.
RGB world training30.193→30.249ms; CDP imagination54.208→54.298ms.
The remaining JAX gap is not closed by this refactor.

**All272 numerical metric reports, first outputs/targets and initial/first/final
saved tensors match exactly:**292 RGB and346 CDP tensors per stage. Controls
also reproduce the retained split-K fixtures exactly. This is fixture parity,
not universal bitwise equivalence or new evidence of game-learning quality.
Sampled estimated Vulkan budget headroom remains at least10.51GiB; not a peak
or physical-free-memory claim. SM utilization remains unmeasured.

## Qualification and failures

Thirteen guarded jobs pass. Six F16/BF16 transpose/edge cases and six typed
tuning probes pass, including unchanged live outputs after scratch tuning.
The existing full F64 oracle checks32 matrix/partial cases and859,216 exact
values;12 long-gradient,18 master-gradient,18 reduced-gradient,32 connected
Adam updates and16 convolution cases also pass. Existing F16 stored-weight
and prologue checks pass. Native F32 cooperative matrices are unavailable on
the5080: F32 shader/SPIR-V validation is not native-F32 device qualification.

CPU shader, type/schedule, cache, binding, actual compiled-plan collection and
reference checks plus all-target/all-feature Clippy pass. Preserve the first
obsolete `vec4<f16>` source assertion failure; the replacement checks actual
F16 buffer storage. Two failed GPU guards remain:

1. The new test requested one timing pair; the existing option validator
   requires at least four. Correct the test, not the validator.
2. Tuning incorrectly rounded odd halfword logical extents to four bytes and
   rejected valid bindings as too small. Keep logical extents exact and pad
   only private transfers; add an actual compiled-plan CPU regression.

Neither failure was a numerical mismatch or wedge. No tolerance relaxation,
ordinary timing repeat, NVML polling or driver recovery. Two standalone NVIDIA
allocation warnings remain recorded in successful guards.

## Adoption

Meganeura056562b, unchanged Blade663afd2/Naga c9fd6a85a; PR257 is stacked on
PR256/9e05731, not an independent main patch. The immutable timed control87a3236
has the same runtime as qualified9e05731. Cache30 and kernel-memo3 invalidate
older matrix representations/search decisions. Both Kindle locks and the
reported backend revision move together; installation/CI stay on
[PR31](https://github.com/kvark/kindle/pull/31). Only the user merges.
Installation finishes05:29:47 UTC:14 native configuration/revision checks,
workspace/Python Clippy and193 installed-binding tests pass. Installed extension
SHA256: `98d1664ff02735b74aa1b59307f9af49a9f5dcff47357060772ac9d193a70ade`.

Artifacts: `runs/typed-matmul-20261013.9OmuXOPr`, including source archives,
all failed/passing guards, immutable binaries, checkpoints and the independent
CPU audit (7.99s CPU; it finishes05:26:41 UTC).1,088 synthetic learner updates,
32 backend Adam updates, plus all development checks; zero game actions.
