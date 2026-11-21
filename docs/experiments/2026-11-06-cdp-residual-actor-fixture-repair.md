# Separate residual learning fixture repair

The original native derivative fixture passes once in.73s. The learning/restore
fixture then fails in.29s while constructing its first synthetic observation:
`Observation::from_vec` requires3136 feature values, while RGB32 supplies3072.
This happens after native construction but before any transition or learner
update. Its guard fails on child exit101; no GPU reset or fault repair occurs.
Keep both original results, source, binary, declaration and guard artifacts.

Repair only the test observation factory to call `Observation::rgb(values,32)`.
Its deterministic values, sizes, seeds, architecture, loss, two-update budget and
all assertions remain unchanged. Add a CPU test that constructs all34 synthetic
observations without a GPU. Recheck the affected CPU tests and strict Clippy,
then package a separate test executable. Production Rust/Python files and the
installed native3003151f/wheel remain unchanged; compare their original pins.

Root `/x/Temp/kindle-residual-actor-20261106/learning-fixture-repair` retains the
original fixture and new receipts. After passing CPU preparation, separately
pin and execute only the corrected learning/restore fixture,120s guard cap and
180s controller cap, on idle AMD/Auto. Use a new checkpoint/output path. Reuse
the already completed derivative result; do not execute that fixture again.
This is the first valid synthetic learning/restore execution, not an Atari
learner or an unchanged retry. No allocation is permitted before the new binary
and source checks pass. Preserve any further failure without automatic repeats.

The original native qualification remains failed. Successful repaired learning
can complete the missing evidence alongside the original derivative result;
report both. No bounded Atari run or full learner is selected by this repair.
The retained control and unfinished five-game goal remain unchanged.
