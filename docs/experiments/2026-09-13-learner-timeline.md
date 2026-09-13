# Opt-in learner timeline: CPU preparation

Status: **CPU-prepared, not GPU-qualified**. Candidate
[`ac8b52d`](https://github.com/kvark/kindle/commit/ac8b52d3d7a9f811f0b74af48fa207e27adf2100)
on `exp/learner-timeline-20260913` forks dependency-only 58f328a. Both use
Meganeura **75dfe901** and Blade **f6f2729e**; the preparation freshly checks
upstream main. Main, the active Pong pair and all three throughput waiters
remain unchanged. No additional GPU declaration or follower is created.

## What changes

The optional `profiler` feature enables Meganeura's existing CPU/GPU trace
writer. It labels the six core learner stages and the outer update, copy and
readback calls. Kindle's two direct transfer paths now harvest their completed
Blade timestamps after the existing waits, including the last pending device
copy during drop. It adds no submit, wait, dispatch, sampling or learning change.
The independent source check removes only the optional annotations/harvesting
lines and reproduces all three original production source bodies exactly.

`dreamer_canary --trace-dir` requires the feature, explicit GPU timing, one
learning repetition and a fresh directory. It rejects simultaneous
`--profile-dir`: that separate profiler changes dispatch grouping. The trace
is saved after sessions/context drop so final pending queries are harvested.
Its contract identifies **synthetic core only**, normal dispatch grouping and
unqualified coverage. Default builds enable no instrumentation or GPU timing.
No Python tracing API or source-matched Python package is added.

This is not yet an Atari/perception timeline. CPU trace spans and GPU pass
intervals are different measurements; a pass extends to the next pass start or
submission completion, not an instruction-level kernel boundary. Clock-error
bounds, actual imported coverage and device execution remain unverified. An
uncovered interval must not be labeled GPU idle merely because it is blank.

## Completed CPU evidence

The [result](../../runs/learner-timeline-cpu-v2-20260913.8diroE/result.json) and
[independent post-commit audit](../../runs/learner-timeline-cpu-v2-20260913.8diroE/independent-audit.json)
reverify **158 source/output pins and all 15 command lifecycles**. Both default
and profiler-enabled builds pass **98 CPU tests each**; each leaves 22 GPU tests
ignored. Formatting, both feature-mode workspace Clippy checks and default
Python Clippy pass. Three actual debug-canary invocations refuse missing
features, missing timing or mixed dispatch profiling before GPU construction
and output creation. These are negative entrypoint checks, not a trace capture.

The two debug executables are retained separately. No release fixture, Python
package, GPU trace, state-parity result or speedup is claimed. The enforced
one-core / 2 GiB / zero-swap scope reaches its 2 GiB memory cap; CPU-only work
is not assumed free of host contention.

Preserve the initial [optional-edge guard failure](../../runs/learner-timeline-cpu-20260913.gzzymq/preparation.json)
and [standalone-lock comparison failure](../../runs/learner-timeline-cpu-continuation-20260913.hAg8LO/execution.json).
Both stop before compilation. The corrected checker verifies the seven optional
tracing packages against upstream versions/checksums and registry requirements.
It explicitly retains Kindle's existing log/smallvec versions and compatible
Windows-only dependency edge; no old dependency is replaced. The default Python
lock remains byte-identical. These tooling corrections change no learning code.

## Before diagnostic use

### CPU-only trace reader and writer fixtures

The [reader check](../../runs/learner-timeline-reader-cpu-20260913.1qssZS/result.json)
passes **55 CPU tests** against the pinned 75dfe trace format. A small Rust
executable links the actual completed profiler libraries; it creates no GPU
context. Its serial fixture records real CPU spans and **fabricated** GPU pass
intervals, deliberately harvesting the latter after the CPU spans. The reader
recovers two synthetic update windows, six stages each and 36 fabricated pass
intervals. These are parser fixtures, not learner updates or GPU measurements.

The two other actual-writer fixtures are correctly rejected: one carries a
before-epoch timestamp rejection; the other forces two CPU threads to emit
`A begin → B begin → A end → B end`. Upstream puts all four boundaries on the
same CPU track. The reader refuses this crossed nesting rather than assigning
the wrong durations. The format supplies no CPU thread identity, so even a
structurally valid trace is not proof of correct arbitrary-thread attribution.
This controlled fixture is not an observation of crossing spans in Kindle's
gameplay, a training failure or a reason to change the current GPU queue.

The [read-only continuation audit](../../runs/learner-timeline-reader-cpu-20260913.1qssZS/independent-audit.json)
reverifies all ten command histories, actual writer PIDs, three raw files,
55 tests and **994 pins**, plus its two supplementary pins. Preserve the
original final audit's [exit 1](../../runs/learner-timeline-reader-cpu-20260913.1qssZS/execution.json):
all commands completed, but tuple rows compared unequal to their serialized
JSON list rows. The separate `audit.py` normalizes only those already-equal
boundary rows, retaining all original checks. Never rerun the exclusive writer;
the original `run.py --audit` retains that known comparison failure.

This is a strict reader for the pinned writer subset, not an independently
validated general Perfetto importer. Use bounded, declared canary update counts.
It rejects malformed/incomplete fields, unordered or rejected timestamps,
missing stages, crossed CPU slices and overlapping/duplicate GPU intervals.
Equal adjacent GPU endpoints do not fail just because harvest order differs.
It reports pass-interval coverage and **uncovered**, not idle, nanoseconds;
execution, calibration, workload-coverage, idle-time and speedup qualification
flags remain false. No hardware, state-parity or whole-Atari gate is added.

### Remaining hardware and runtime checks

First finish the [already declared dependency and block qualification](2026-09-13-throughput-priority.md).
Do not insert this preparation into those pinned queues. A separate diagnostic
must then bind fresh release binaries and the selected runtime, verify the
executing adapter and qualify:

- A small real transfer/compute fixture: complete, nonduplicated query harvest,
  ordered timestamps and host-bracket/calibration checks. Import the raw trace;
  reject missing workload coverage or rejected timestamps.
- Update-1 and eight-update full state, all 146 optimizer moments and non-timing
  reports against the unchanged control. Trace annotations are not parity proof.
- Actual six-stage coverage throughout the declared learner window, separately
  from construction/teardown, and directly free memory of at least 2,048 MiB.
- Instrumented versus uninstrumented overhead with retained untraced timing.
  A traced duration is diagnostic, never the throughput benchmark.

Only after those checks should this trace rank recurrent handoffs and producer
work. A full N6 pixel trace, including perception and actual-frame clocks, is
still needed before whole-Atari idle-gap claims. This work changes no game budget,
seed, competence gate or five-game result.
