# Meganeura and Blade calibrated-timing update

Status: isolated dependency/identity candidate; Rust/backend and Python CPU
qualification complete. Candidate branch a7fc16b is committed and pushed.
No GPU work, runtime qualification, speedup or adoption is claimed. The user's
September 13 [throughput priority](2026-09-13-throughput-priority.md) now places
this qualification before unstarted Pong roots. Active root 1009 and every
historical package stay fixed; earlier queue statements below are historical.
The newer [September 13 convolution-runtime update](2026-09-13-meganeura-conv.md)
supersedes this source candidate before GPU work. Preserve this completed CPU
evidence and the retired, unrun hardware declaration; never restart its follower.

## Why this update comes before more backend diagnosis

A direct remote check at **2026-09-12 19:12:16 UTC** finds Meganeura main at
[`45991be1`](https://github.com/kvark/meganeura/commit/45991be11880e3b9da33fa75bab66098b6329c31).
Unlike the preserved morning documentation-only check at `3622e06f`, this tip
has runtime changes. The six new commits cover calibrated GPU/transfer timing,
context reuse, self-describing tracing and model loading, the landed Blade API,
and package-assembly compatibility. Relative to qualified ce80e9cd, 23 tracked
paths differ. Compiler, autodiff, shader and LeVJEPA cache code remain unchanged;
that inspection does not replace numerical tests.

The [landed timing API](https://github.com/kvark/meganeura/commit/a10dc737)
requires shared **git Blade `f6f2729e`**, not registry Blade 0.9.0. Both direct
Kindle and transitive Meganeura dependencies must resolve to that same package
to keep GPU context types compatible. Rust 1.92 and Naga 30 remain unchanged.
Upstream's `cargo package --no-verify` change accommodates the unpublished API;
it is not a runtime validation result.

## Scope and identities

Candidate `exp/meganeura-timings-20260912`, Kindle **`a7fc16b`**, is based on
qualified control `1e00e818`. Its worktree is
`/x/Code/.kindle-meganeura-timings-20260912`. Only AGENTS.md, the two Cargo locks,
`kindle/Cargo.toml` and backend identity/checking in `dreamer/mod.rs` change.
The separate backend and Blade worktrees contain the exact upstream commits;
the shared developer checkouts remain unchanged.

The [completed Kindle lock resolution](../../runs/meganeura-timings-cpu-20260912.mlddbv/locks-result.json)
binds 15 inputs/outputs. Both production locks change only the Meganeura and
Blade package records; unrelated dependencies are exactly retained. Backend
and shared-Blade checkpoint identities remain strict. No model arithmetic,
Python runner, observation protocol, replay ratio, BPTT, exploration, block
matmul, low-priority queue, skipped parameter initialization or tracing option
is enabled by this candidate.

Main stays on qualified ce80e9cd. The running learning package stays native
f6a2b6ad / backend 4d45ba3a, with its exact source-matched Python. The prepared
block-matmul and Breakout packages remain on ce80e9cd; their completed evidence
is preserved, not silently relabeled as evidence for this update.

## CPU qualification and preserved preflight failure

The first [preflight declaration](../../runs/meganeura-timings-cpu-20260912.mlddbv/cpu-declaration.json)
copies the old standalone backend lock and updates Blade. Cargo completes both
commands, but the strict lock comparison stops before cache copying, compilation
or testing: seven Windows-target dependency references also resolve differently.
Preserve that [command ledger](../../runs/meganeura-timings-cpu-20260912.mlddbv/cpu-events.jsonl)
and original writer; do not rerun it or report its stop as a numerical failure.

The separate [continuation](../../runs/meganeura-timings-cpu-continuation-20260912.eNLRUw/declaration.json)
accounts for exactly those seven `windows-sys` edges. All package versions,
checksums and other fields remain unchanged apart from Blade's required git
source. Checksum-matched registry metadata verifies each changed edge is
`cfg(windows)`, including the renamed dependency in nu-ansi-term. Its snapshot
is retained independently of the mutable registry cache. An initial read-only
metadata probe missed that alias; no native code or experiment changed.

CPU checks use a private copy of the previous build cache, one CPU core,
2 GiB host memory, zero swap and one Cargo job. They cover formatting, workspace
and Python-binding Clippy, all Kindle CPU tests, focused backend compile/cache/
codegen/block tests, new profiler tests and Blade timestamp/barrier tests.
The [completed result](../../runs/meganeura-timings-cpu-continuation-20260912.eNLRUw/result.json)
passes **95 Kindle tests**, **80 focused backend tests**, **9 profiler tests**
and **4 Blade timestamp/barrier tests**, plus formatting and both Clippy checks.
The 22 Kindle GPU tests stay ignored; backend GPU tests remain filtered.
All 12 command lifecycles and **18,358 input/output pins** independently
reverify using the read-only `continue_cpu.py --audit`. Result SHA-256:
`4fc2a2d7613a57022b5f5927528732d38e6a2483793864354bfc3ca4aa1ccd3e`.
The host cgroup reaches its 2 GiB cap and completes normally; this is not GPU
memory evidence. Preserve both the failed writer and completed continuation.

### Completed Python package

The separate build in `runs/meganeura-timings-package-20260912.NBiJHP` uses a
private target directory and the same resource caps. Its launch freshly checks
that remote main still resolves to 45991be1 and reverifies the completed CPU
proof before compilation. The wheel builds successfully from the actual source.
Its first import audit stops before pytest because the wrapper expected Kindle's
Rust crate version to equal the Python package's version. They are 0.1.0 and
0.3.0, respectively. Preserve that [audit-only failure](../../runs/meganeura-timings-package-20260912.NBiJHP/package-tests.log)
and writer; it is not a native compilation or learning failure.

The [completed check](../../runs/meganeura-timings-package-check-20260912.hCiUlb/result.json)
reads each crate's name/version from its Cargo manifest and tests the same
already-built package, without rebuilding or modifying it. All **547 Python
tests pass**. Source files, compiled library, wheel and actual import agree;
all original command outcomes, inherited CPU evidence and historical/active
controls remain bound by **23,920 pins**. Its read-only `check_package.py --audit`
reverifies the result; neither completed writer may be rerun.

The CPU-qualified bundle is
`runs/meganeura-timings-package-20260912.NBiJHP/package`, source `a7fc16b`, native
`29774c090d0bc812a3b1ddd6176f96f8d4d60176ed8f67aff432505e49497ad7`.
Wheel SHA-256:
`9775a78565c9a63e5c9d898565d31606b4d34e4bf9c239e67f33627f20b2a2f7`.
Completed package-check SHA-256:
`2965b45d084053a3388eb57a306e6f7958b3287295c0f19ada901a1eaa359b1a`.
This package has not constructed a GPU agent or passed a hardware gate. Do not
use it for a new learning run or pair it with a different Python runner/auditor.

### Release hardware fixtures are prepared, not executed

The [compilation-only preparation](../../runs/meganeura-timings-fixtures-cpu-20260912.Xk29rR/result.json)
completes on **September 13**. Its entrypoint freshly confirms upstream
45991be1 and reverifies the completed Rust/backend and Python-package evidence.
It copies only the release cache into a private target directory; the original
cache, source locks, package and every active/historical control remain unchanged.

All five source-matched release executables are built: Kindle's test binary,
the backend regression/smoke/Gemma test binaries, and `dreamer_canary`. Cargo
artifact records bind their exact manifests, release profiles and output paths.
The four test binaries only run `--list`: all **19 required hardware-test names**
are present, but **none is executed**. The canary is not run.

All **eight command lifecycles and 25,489 input/output pins** independently
[reverify](../../runs/meganeura-timings-fixtures-cpu-20260912.Xk29rR/independent-audit.json).
Result SHA-256:
`0f4b4c1a1e6fa3bfaa6e1f2ec3cb8824c021da5863b6ca22578dd5fa9f6f566c`.
The one-CPU, 2-GiB/zero-swap preparation completes normally at its memory cap.
This is host build evidence, not combined GPU-memory or runtime qualification.
Preserve the completed writer; its `prepare.py --audit` is read-only.
No GPU declaration, follower, queue change, numerical parity, speedup or adoption
is introduced. The required runtime and tracing checks below remain outstanding.

## What the timing changes can and cannot establish

The September 13 [separate nineteen-test hardware stage](2026-09-13-throughput-priority.md#superseded-first-native-stage)
was declared with 42 CPU scheduling/proof checks. Its once-only follower was
retired while idle after upstream advanced to 75dfe901; no GPU test ran. Preserve
the compiled fixtures, declaration and terminal record. A separately declared
latest-source handoff is required, followed by full-state and pixel/timing gates.

Blade now exposes calibrated CPU-domain pass-start and completion timestamps.
Meganeura harvests them after completion rather than placing passes using host
submission times; tracing also carries more stage and buffer metadata. This
addresses a limitation of the earlier profiler, but accuracy, coverage and
overhead on Kindle's actual device are still untested. A pass interval can
include time until the next pass starts; it is not instruction-level kernel time.

Kindle already supplies the shared GPU context to its sessions, so the new
context helper does not call for an actor/learner redesign. Its own device-copy
and readback paths still wait on Blade directly, without harvesting their
transfer timings. Updating dependencies alone therefore does not establish a
complete transfer trace or calibrated idle-gap accounting for Kindle.
The later [isolated learner-timeline candidate](2026-09-13-learner-timeline.md)
prepares optional harvesting and core-stage labels on 75dfe901. Its CPU tests
do not establish an actual GPU trace, calibration or whole-Atari coverage;
the default and pinned packages remain unchanged.

Before adoption, qualify the source-matched package above with the unchanged
full production-gradient/cache/reset tests, complete logical state and optimizer
moments from updates 1 and 8, restore/pixel traces, at least 2,048 MiB directly
free throughout covered GPU windows, and untraced AB/BA timing. Separately test
trace coverage/calibration and overhead before using it to diagnose idle gaps.
No new GPU declaration or follower is created here. Keep the dependency update
separate from block-matmul or transfer instrumentation, and recheck upstream
again before a new backend investigation.
