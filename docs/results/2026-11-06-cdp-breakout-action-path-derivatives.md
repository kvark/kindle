# Frozen action-path derivatives at saved rescue origins

The [action-path probe](../experiments/2026-11-06-cdp-breakout-action-path-derivatives.md)
passes its separately declared [fixture repair](../experiments/2026-11-06-cdp-action-path-fixture-repair.md).
Production is selected for the two frozen checkpoints and 120 origins each,
with eight paired draw streams per origin, two fixed plans and two objectives.
There is no learner, optimizer step or Atari action. No production result is
established by qualification alone.

Original qualification at ceb2be51 fails before forward/backward execution:
the toy fixture attempts elementwise [1,1] × [1,3]. The repaired fixture uses
matrix multiplication and its CPU test now constructs and compiles the graph.
Production graph, objective, estimator, inputs and thresholds are unchanged.
The original failed guard, stderr, binary and fixture source remain under
`/x/Temp/kindle-breakout-action-path-20261106`; do not repeat that stage.
Earlier missing historical-receipt and closure index-type failures remain.

Before repair, 190 Kindle/5 gym CPU tests, four analysis tests and strict Clippy
pass. After the fixture change, both affected CPU checks, strict Clippy and
the release build pass. Native qualification checks all 18 F64 surrogate
derivatives, both scalar objectives, frozen inputs and zero optimizer steps.
Tiny forward values match separate existing transition/head/value inference
graphs, maximum relative error 1.180303e-6. Native test wall time is .81s;
guarded controller/audit is 3.193s, with 774 unchanged pins and a passing host
guard, completing **07:04:52.759 UTC November 6**. This repaired qualification
passes; the original qualification remains failed.

Repaired source `b7d45c32eda52bd400b29142ff5b3911469c7219`, test executable SHA256
`f564ec9027cd095a676e2a280c51faae8954641e40afea1177f73587baf7928b`.
Root `/x/Temp/kindle-breakout-action-path-20261106/shape-repair`; exact declaration,
source pins, inputs, binary and guard receipts are retained there. There is no
new Python package or production runtime change. The parent build cache is
disposable after packaging and validation; both packaged binaries are retained.

The two production jobs remain bounded at 180s each, serial, under the existing
host guard. Their first batches must match the existing native inference path;
all frozen inputs/checkpoint files must remain unchanged. Numerical success
does not establish useful derivatives or gameplay gains. Keep the retained
Breakout control and actor dynamics off; original 864/90%-two-wall and five-game
gates remain unfinished.
