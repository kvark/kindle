# CDP prediction scale1000: bounded check and fresh learning comparison

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-cdp1000.md) selects
one fresh8,192-action/489-update check of prediction scale500→1000. Only
`loss_scales.future_prediction` changes. The retained native447fe50, original
convolution/max-pool encoder, teacher, geometry and learning schedule remain
fixed. No native build is needed. After the completed bounded check, a separate
[fresh200k comparison](../experiments/2026-11-06-cdp-breakout-cdp1000-200k.md)
is selected below; it has not started yet.

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

## Completed bounded check

The queue completes once at source957bed96. Training takes46.680678s for
8,192 aggregate actions/32,728 emulator frames and489 updates, versus the saved
47.689684s control (2.116% less observed time). This is not a paired speed result.
Construction takes2.352502s separately; the padded200k projection is1253.631498s
against the1800s gate. Conservative full-replay headroom is11.221905GiB against
the2GiB gate. Both gates pass.

All489 reports are finite and satisfy the1000-weight world loss and behavior
loss arithmetic. First update376 and residual credit.5 match the declaration.
All357 actual-initial tensors and47 pre-update events match retained. Encoder,
reward, continuation, actor and value weights change. Full independent replay
verifies all8,192 actions, pixels, frames, rewards, boundaries and expert labels,
including3,804 overrides (46.436% of actions;20% is the episode-selection rate).

Frozen restore completes1,024 actions/4,096 frames in0.922426s, with zero updates,
no guidance, preserved1000-weight config and all357 tensors unchanged. Its128
ticks per lane complete no natural episodes; this is not competence evidence.

| Training stage | Seconds |
|---|---:|
| Learning |40.404433|
| Acting |2.466933|
| Environment |2.506276|
| Observe |0.500698|
| Reset |0.094614|
| Checkpoint |0.262006|

Reported learner timing totals40.348870s, including posterior6.156417,
world training13.647480, imagination16.570146 and behavior training2.508965s.
These are wrapper/session timings, not isolated kernel costs.

The controller completes in65.909064s at18:14:27.545 UTC November6. Both host
guards pass, both GPU children exit0, and all49 declared pins plus four experiment
source files remain exact. Independent0.337578s CPU publication closure at
18:15:32.218 UTC verifies53 completion artifacts and terminal controller/child
identities1531541/1531549/1531722. No active Kindle service or render holder
remains. The completed GPU stages, audits and replays are not repeated.

Evidence is in `bounded/check/completion.json`, `review.json`, guidance/frozen
replays and guards, with `bounded/publication-closure.json` sealing source,
artifact and terminal checks. The original CPU import failure remains preserved.
No full learner was selected by that bounded declaration. Retain143.667
greedy/70.750 sampled; original864/90% two-wall and the five-game goal remain
unfinished. No automatic successor follows from the bounded result.

## Fresh200k comparison selected

The separate fixed-budget declaration selects one fresh200k/12,477-update
learner, seed1009, with100k recovery and actual-initial/final sampled/greedy
24-natural-episode assessments. Only prediction loss500→1000 changes. Favor
the candidate only if greedy raw score and own-initial gain/min improve, with
a positive lower paired eight-lane95% gain/min interval. This is a reused
development seed/cohort; even a passing result would need fresh confirmation.

The first full CPU helper preflight fails after2.567353s at the plotting fixture:
its copied path points to `check/train.jsonl` instead of `bounded/check/train.jsonl`.
The original failed invocation, plot receipt and source bytes remain retained.
Earlier helper assertions reached this plot assertion, including exact bounded/
retained expert accounting, header/config rejection,1000-weight loss arithmetic
and rejection of a500-weight total. The comparison entry fixture had already
passed11 negative-header cases.

A separate1.187419s CPU repair fixes that path and the fixture's expected
coefficient, completes only the missing plot/source checks and imports no
native module. It does not rerun the original failed helper invocation or any
completed GPU stage. `helpers-review.json` explicitly records
`passed_with_plot_fixture_repair`, linked to the preserved failure and separate
repair. The original canonical-package import failure also remains preserved.

Root `/x/Temp/kindle-breakout-cdp1000-20261106/full`; the five actual queue
commands and launch declaration are being prepared. No new GPU allocation is
active yet. No extension, second learner or automatic successor is selected.
