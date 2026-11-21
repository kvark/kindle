# RGB96 qualification after correcting the inactive-stream fixture

The [original native declaration](2026-11-06-cdp-rgb96-native.md) fails its first
resize fixture at14:16:47.521 UTC November6: maximum absolute error.125. Child
1504247 exits101, with no unfinished children or reported host fault. Encoder
and restore stages never start. Original binaries, source, logs and262 checked
pins are retained under `/x/Temp/kindle-rgb96-20261106` and remain failed.

The fixture initialized the observation input to.125 and incorrectly expected
that value in the inactive lane. `GpuPreprocessor::submit` explicitly clears the
whole target before a partial-stream write. The expected inactive value is zero,
not the previous input. Correct only the test's inactive expected values and
rename it to describe clearing. Keep the nonzero initial sentinel, all active
pixel references, exact resident-layout comparisons and numeric tolerances.
No production code, runtime, shader, precision or host changes.

Original fixture SHA256:
`d45be8f4034fbbbc0809e42e1621ebafdae8b57b129c6af2227c512ed34a33c5`.
The separate CPU check passes three RGB96 tests with three native tests ignored;
strict Kindle test-target Clippy and release fixture rebuild pass. The repaired
receipt verifies only this test source differs among256 files. Production native
remains `7704d25c9df5d37c48d5223f2d12c1d5d3c0a812c1f68cf393c87a637d010ab0`.
New test binary SHA256:
`c2225f098e08cbe1da14c17ee9d1410a076fb9ca3122b71ddf155d6232798df2`.

Select a separate queue at `fixture-repair/native-guards`, with:

- Corrected `vision::preprocess_gpu::rgb96_tests::gpu_rgb96_resize_matches_scalar_and_clears_inactive_streams`.
- Previously unstarted `dreamer::networks::rgb::tests::tiny_rgb96_encoder_matches_independent_reference`.
- Previously unstarted `dreamer::agent::rgb96_tests::rgb96_learns_and_restores_parameters_moments_and_geometry`.

All other original native criteria, Auto/AMD host guards,120s child caps and420s
controller cap remain. Use a fresh `fixture-repair/native-checkpoint`; zero Atari
actions and exactly two Tiny learner updates. A failure stops this new queue,
without automatic retries or tolerance changes. Preserve both original failures.
Separate success may qualify the repaired fixture and unstarted checks; it must
not relabel the original native attempt as passed. No bounded/full learner is
selected. Original864/90%-two-wall and five-game goal remain unfinished.
