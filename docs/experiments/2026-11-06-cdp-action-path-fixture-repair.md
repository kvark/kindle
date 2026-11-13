# Shape repair for the action-path numerical fixture

The original [action-path qualification](2026-11-06-cdp-breakout-action-path-derivatives.md)
at source ceb2be51 fails during toy graph construction: elementwise multiply
does not accept [1,1] by [1,3]. The device context was created, but no native
forward/backward execution or production checkpoint probe occurred. Preserve
the original queue, failed guard, stderr, binary and source identity under
`/x/Temp/kindle-breakout-action-path-20261106`. Do not rerun that stage.

Replace the fixture's scalar-by-row operation with [1,1] × [1,3] matrix
multiplication. This expresses the same intended three logits and F64 reference.
Make the existing CPU reference test construct and compile this toy graph, so
its shape is checked before GPU execution. Production graph, objectives,
estimators, manifests and thresholds remain byte-identical.

Use `/x/Temp/kindle-breakout-action-path-20261106/shape-repair` for the new
qualification, with a separately built and pinned executable. Reuse the
disposable parent build cache only for compilation; retain the original
packaged binary. Recheck the changed CPU test and strict Clippy, then declare
the repaired native fixture with the unchanged 180s cap and host guard. It must
pass both F64 objectives/18 derivatives and Tiny forward parity before production
can be separately selected. No cap increase, optimizer, learner, emulator work
or GPU recovery is authorized by this repair. Original failures remain failures.
