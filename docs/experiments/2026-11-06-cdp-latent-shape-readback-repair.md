# Separate categorical fixture readback repair

The original [native qualification](2026-11-06-cdp-latent-shape-native.md) fails
on its first categorical fixture after the first scalar-loss comparison. At
`latent_shape_tests.rs:287`, `Session::read_output(1)` reads one value from the
first output, not output index1. The raw-KL assertion therefore compares the
clamped loss with raw KL. The later `read_output(2)` likewise would read two
values from the first output instead of the1024-entry categorical sample. Source
inspection of pinned Meganeura `runtime/transfer.rs` confirms the API contract.

Original sourcebefcbf0c, binary73a18277 and all queue/log/guard/execution artifacts
remain under `/x/Temp/kindle-latent-shape-20261106`; failed child1462683 exits101,
native test.71s/controller2.066s/wrapper2.169s, completion11:00:22.953 UTC November6.
No derivative assertions or synthetic learner updates were reached. The guard
reports the assertion exit, no recovery, and no unfinished children. This is
not evidence of a GPU fault. The original qualification stays failed.

Select a test-only repair of those two reads to `read_output_by_index`, with
appropriately sized buffers and diagnostic actual/expected scalar messages.
Keep graph, F64 oracle, draws, tolerances, route cases, production sources,
packaged native and learning fixture unchanged. Preserve original fixture bytes
under `readback-repair/original-latent_shape_tests.rs`. Run the affected CPU tests,
strict workspace Clippy and compile a separate release test executable; do not
rebuild the wheel or repeat installed-package checks for this test-only change.
Require that only this fixture differs among the original246 source pins.

Then run the repaired categorical fixture and the previously unexecuted learning/
restore fixture serially under the existing host guard, with the original120s
per-job/300s controller caps, cases and thresholds. Use separate declarations,
binary, queue, guards and checkpoint under `readback-repair`. Pin every current
source, original package and failure receipt. Do not execute the original queue
again. Stop on any assertion, real fault or numerical error; no automatic repeat.

A successful repair completes only missing qualification evidence; it does not
relabel the original failed run. No Atari learner or bounded check is selected.
Retained143.667/70.750 and the unfinished864/90%-two-wall/five-game gates remain.
