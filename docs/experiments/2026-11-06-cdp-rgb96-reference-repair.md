# RGB96 scalar-layout repair and remaining native checks

The [first repair queue](2026-11-06-cdp-rgb96-native-repair.md) passes resize:
child1504944 exits0, all six2×4 objects remain visible, and scalar/layout/cleared-
inactive checks pass. Reuse that completed stage; do not rerun it.

Encoder child1504955 exits101 before derivative checks at14:20:27.027 UTC
November6. The scalar reference's final channel-to-token conversion still uses
16 spatial cells and16-cell channel stride, producing512 values; RGB96 requires
36 cells and1,152 values for Tiny/B2. This explains its incorrectly compared
layout. Restore never starts. Original failed bytes, binary, guards and262 source
pins remain under `fixture-repair`; no reported host fault or recovery.

Correct only the reference to use its computed final side and channels. Add a
CPU constant-output case checking all1,152 values against independently derived
channel RMS/SiLU and an explicit native-reference output-length assertion.
Four RGB96 CPU checks, strict test-target Clippy and release fixture build pass.
Production native7704d25c is unchanged. The new receipt verifies only the two
documented test files differ from the original256-source build.

Select only the repaired encoder and previously unstarted restore in the fresh
`/x/Temp/kindle-rgb96-20261106/reference-repair/native-guards` queue:

- `dreamer::networks::rgb::tests::tiny_rgb96_encoder_matches_spatial_reference`
- `dreamer::agent::rgb96_tests::rgb96_learns_and_restores_parameters_moments_and_geometry`

Test binary SHA256:
`0bb0d62a50b1b63b2c3b4148eb1d99cee21c74289fc96dd638f5a41afcf4d8e5`.
Original value/derivative tolerances, finite-difference step, synthetic inputs,
seed, Auto/AMD guard and120s child/420s controller caps remain. The restore path
is fresh `reference-repair/native-checkpoint`. Zero Atari actions andtwo Tiny
learner updates. Stop on failure; no automatic retry or tolerance changes.
Preserve both failed native queues and the original Python failure. Separate
success may complete the missing qualification, not relabel any failed attempt.
No bounded/full learner is selected; retain the original goal and gates.
