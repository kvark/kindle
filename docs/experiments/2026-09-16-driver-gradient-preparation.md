# CPU preparation for the next driver-580 correctness check

The two approved [initialization-only runs](2026-09-16-host-only-initialization.md)
are complete. Approval for a separate production-gradient control/candidate pair
was pending during this preparation. The user's later direction resumes GPU work,
and the [separately declared pair now passes](2026-09-16-driver580-gradients.md).
This preparation itself executes **no GPU test or NVML query**, creates no
GPU launcher/declaration, and does not restart training or the held Pong queue.

## Matched test-only changes

Two isolated, pushed branches preserve the historical runtimes:

| Arm | Source / branch | Backend / Blade | New native SHA256 prefix |
| --- | --- | --- | --- |
| Control | `afb973a`, `exp/driver-gradient-control-20260916` | `9b9e7ee7` / `c96a9a87` | `b069df58` |
| Candidate | `0d0a831`, `exp/driver-gradient-candidate-20260916` | `070f4b51` / `100bb813` | `7b4b9a8c` |

Only worktree instructions and `world.rs` test code change from f7b1914/eebbf7c.
Both new `world.rs` files are byte-identical, and all production code, dependencies,
locks, graph construction, test inputs, initialization and arithmetic are unchanged.
The source check rejects any removal/replacement of the original test body except
its ignore-description string. These are new test executables, not the binaries
that passed combined initialization.

The existing ignored temporal-batching test now requires explicit selection
`production-world-driver-20260916`, `KINDLE_FULL_WORLD_PARITY=1` and a listed
expected driver before GPU context creation. It checks the actual RTX 5080,
NVIDIA driver/name/version, non-software status and selector 0x2c02 before either
large session is built. Missing selection, tiny-mode fallback and an unlisted
driver are rejected by the two new CPU contract tests.

The intended GPU fixture remains the original serial-versus-temporally-batched
**12M/B16/T64/full-BPTT/F32** world test, with nine loss comparisons, all parameter
gradient comparisons and frozen replay-critic checks. Its loss tolerance stays
`3e-4 * max(abs(reference), 1)`; gradient tolerance stays `3e-3 * norm + 1e-7`.
The prior outer worst-relative-L2 gate of .003 must also remain in a future
declaration. These are checks of one synthetic step per session, not gameplay,
a learning campaign or a throughput benchmark.

The fixture retains `DreamerConfig::new(18)` defaults, including seed 0 and
train-ratio metadata 32, with full BPTT, reconstruction zero and future prediction
.25. The ratio is not used to schedule this standalone test. Do not relabel it
as the N6/R256 pixel/learner memory gate; no video frontend is loaded here.

New flushed records cover context, graph/session construction, parameter/input
preparation, step submission/return, every loss and parameter-gradient summary,
and completion counts. A new assertion rejects all-zero gradients without changing
any original numerical tolerance. This improves diagnostic evidence, not the model.

**Remaining limitation:** these historical `Session::wait` implementations discard
errors. The breadcrumb is deliberately named `step.wait_returned`, not successful
GPU completion. The instrumentation does not repair that API. Numerical checks,
the nonzero-gradient assertion and a separately declared host guard remain
necessary; none is a proof of driver safety or first-wedge prevention.

## Completed CPU evidence

[`dPzrO2`](../../runs/driver-gradient-cpu-20260916.dPzrO2/declaration.md)
builds into new private copies of ZkxGRu/ITVpsF caches, under one CPU / 2 GiB
maximum memory / zero swap. Both original writers and caches remain unchanged.
Each arm passes **86 Kindle CPU tests**, formatting and release Clippy, with
**23 GPU tests ignored**. Only enumeration selects the ignored gradient test;
it is never executed. Meganeura's unfiltered test suite is not run.

The independent read-only audit verifies all **14 command lifecycles**,
**18,108 input pins**, **ten compiled artifact pins**, both fresh native build
records and actual shared Meganeura/Blade dependency edges. The retained parent
CPU results are re-audited, not recompiled. No Python package, hardware result,
native gradient value, speedup, memory reserve or adoption is established.

Fresh upstream checks still find Meganeura **5a570099** and Blade **bbf5bf5**.
The new Blade optional-timing API fix remains work for a separate compatible
timing candidate. Timing is disabled in this fixed-runtime driver comparison;
do not call either arm latest-Blade qualification.

```bash
/usr/bin/python3 -B runs/driver-gradient-cpu-20260916.dPzrO2/prepare.py --audit
```

Preserve the completed preparation and private targets; never rerun its writer.
Before any hardware execution, obtain approval and separately declare the exact
two native commands/environments, fresh host/upstream identity, host-only guard,
complete initialization plans/observations and strict numerical-result reader.
That new reader/launcher still needs its own CPU refusal checks. Inspect control
before candidate, stop on any failure, and start no automatic successor.
GPU recovery action, utilization and directly free/reserved memory stay unmeasured.
All original qualification/throughput gates, quarantines and the five-game goal
remain; this preparation is not another Atari win.
