# CDP prediction scale1000: bounded execution and cost check

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-cdp1000.md) selects
one fresh8,192-action/489-update check of prediction scale500→1000. Only
`loss_scales.future_prediction` changes. The retained native447fe50, original
convolution/max-pool encoder, teacher, geometry and learning schedule remain
fixed. No native build or full learner is selected.

Earlier gradient attribution found imitation already a major representation
objective, partly aligned with CDP. This does not establish that stronger CDP
will improve control; the check tests an untried coefficient without adding
parameters, rollout steps or updates. The completed stride-two comparison
remains closed and its option stays off.

## CPU validation and preserved import failure

The first canonical-runner check fails before pytest collection in0.095063s:
`ModuleNotFoundError: No module named 'kindle._return_boundaries'`. It combines
the current runner with the older retained Python package. Preserve the exact
failure in `canonical-tests.json`; no GPU/emulator construction is reached.

A separate CPU-only package combines current Python sources with the unchanged
native binary and blocks native construction. Canonical checks pass39 tests
in0.220011s including process overhead. The private retained runner passes38
tests, with one explicit skip for the unsupported newer open-loop option,
in0.239808s. Both report zero native construction attempts. Production code and
tests are unchanged by the package repair; its source and binary receipt is
`canonical-package-repair.json`. The GPU queue uses the original retained
runtime unchanged, not this CPU fixture.

Tests cover omitted/explicit500/1000, centered and uncentered config identity,
invalid F32 weights, encoder/evaluation/restore rejection, checkpoint value
preservation and effective open-loop overflow in the canonical runner.
The exact two-command preflight passes in0.118725s, with zero native/emulator
construction,357 tensor/schema checks and47 initial-trace fixture checks.
It rejects a500-weight config and verifies1000-weight loss arithmetic against
a saved report. Those fixture results are not new learner measurements.

## Allocation and evidence

Root `/x/Temp/kindle-breakout-cdp1000-20261106` retains the private runner,
pre-edit canonical bytes, runner diff, all CPU declarations/results,
`cpu-review.json`, `runtime-reference.json`, fixed protocol and bounded helpers.
Native SHA256:
`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`.
The source-copy receipt and preflight pin the exact commands and audit sources.

The bounded GPU queue is prepared, not yet run. It declares8,192 training
actions followed by1,024 frozen restore actions, full action/label replay,
initial/frozen tensor checks, host guards and fixed cost/headroom gates.
No natural-episode competence evaluation or fresh200k allocation is selected.
Retain143.667 greedy/70.750 sampled; original864/90% two-wall and the five-game
goal remain unfinished. No automatic successor or repeated completed stage.
