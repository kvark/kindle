# Latest Meganeura: dependency-only convolution-runtime refresh

Status: **CPU-qualified only**. Kindle `58f328a`, Meganeura `75dfe901`, shared
Blade `f6f2729e`; isolated native `fa6bdd2a`. Main remains qualified ce80e9cd,
and the active Pong pair remains on its original f6a2b6ad package. The user
explicitly confirmed finishing that pair, then qualifying throughput before the
remaining Pong roots. The hardware and full-state/pixel stages are separately
declared and waiting. No GPU test result, speedup, adoption or new learning is claimed.

## Why another source update

The September 13 direct remote recheck finds
[`75dfe901`](https://github.com/kvark/meganeura/commit/75dfe901deb87ca0054c438437efd3aa388b7188),
three commits after the previously prepared 45991be1 runtime. The diff covers
18 files: shape-specialized convolution selection in bounded autotuning,
convolution code generation and tuning metadata, qualification-example adapter
selection, tests and documentation. It is not a documentation-only update.

Compiler/runtime defaults still select uniform-parameter convolution kernels;
specialized variants require opt-in tuning. Kindle's LeVJEPA frontend uses
patch matmuls and its session constructor leaves tuning disabled. This update
does not enable convolution tuning, block matmul, tracing, skipped parameter
initialization, low-priority queues or new learning settings. No automatic
speedup is expected. The calibrated timing API from 45991be1 is retained, but
API availability is not verified Kindle idle-gap coverage.

The isolated `exp/meganeura-conv-20260913` worktree changes only AGENTS.md,
`kindle/Cargo.toml`, both locks and the backend identity constant relative to
`a7fc16b`. Both locks change only Meganeura's git revision. The standalone
backend lock is byte-identical to the already checked 45991 lock. All historical
packages, source controls, cache corrections and strict restore rules remain.

## Completed CPU evidence

The [CPU preparation](../../runs/meganeura-conv-update-20260913.E9P9ai/result.json)
passes **95 Kindle tests and 122 backend/Blade tests**, formatting and both
Clippy checks. The backend coverage includes compiler, caches, shader generation,
block CPU oracles, profiler, tuning configuration/selection and Blade command
timing tests. The 22 Kindle and three tuning GPU tests remain ignored.
All **17 command lifecycles and 24,475 input/output pins** reverify. Result:
`93cb5f62ecb248b3bff483ebe5746ab68765075f2242f2190537643da9616828`.

The [isolated package](../../runs/meganeura-conv-package-20260913.yBx5nt/result.json)
passes all **547 Python tests**, source/build/wheel/import identity and
**30,056 input/output pins**. It derives Rust crate versions from their manifests,
preserving the earlier hardcoded-version failure without repeating it.
The bundle is `runs/meganeura-conv-package-20260913.yBx5nt/package`.

- Native: `fa6bdd2a00343e53ad1c56dbc831eb434ceb8ebe4a765077a7fead52790c97c8`.
- Wheel: `e6a39e18885ae34b5d1c42bc8ddb09c6eabaed81c5cbf8e953443c8b1b2ae3f3`.
- Package result: `c88699d16facac6d884d123a9cb53b277ad0bf83cf8617875c6b90703133ab81`.

Both preparations use one CPU core, one Cargo job, 2 GiB host memory, zero swap
and private cache copies. They complete normally; host build memory is not GPU
memory evidence. Preserve these completed writers. Their `--audit` modes are
read-only.

The [release-fixture preparation](../../runs/meganeura-conv-fixtures-20260913.IwwfUr/result.json)
also completes: five source-matched executables, eight command lifecycles and
**31,640 pins**. The four test binaries only run `--list`, confirming all nineteen
required GPU tests; the compiled canary is not executed. Its result hash is
`e2fec7c9608583f88eac482fe4d5a3946affac4191a0b108bbcbb29e1ba4c972`.
No GPU test or canary has run. Preserve the completed writer and private target.

## Queue preservation and remaining qualification

The old 45991 hardware declaration remains intact. Its idle follower
260782/14012130 was [retired by bound pidfd](../../runs/meganeura-conv-update-20260913.E9P9ai/retirement.json)
at **05:57:57 UTC**, with no child or GPU-stage output and all four active Pong
processes unchanged. Preserve its `SystemExit(130)` terminal event, scripts,
fixtures and declaration; never restart it. The tested hold before Pong 2017
remains installed.

The [new first-stage declaration](../../runs/meganeura-conv-runtime-20260913.ZVxDeV/manifest.json)
binds **31,664 inputs and 46 CPU checks**, the actual live-parent refusal,
retirement proof and nineteen source-matched native tests. Follower
**269283/start ticks 14332226** starts at **06:09:35 UTC** and waits on original
scheduler 42730/1021056. Its [independent handoff audit](../../runs/meganeura-conv-runtime-20260913.ZVxDeV/handoff-audit.json)
reverifies all pins, actual declaration/launch stdout, both retired followers,
unchanged active process identities and absence of any hardware child or GPU
outputs. Manifest hash:
`4cee74d69d2e748048c41036fa3dbd89a3cd8dcaf33922cef7e0a73f40b762a5`.
Do not edit these live inputs or start a competing worker. A complete paired
competence failure is retained; incomplete raw evidence, another terminal error,
changed upstream or a runtime/memory failure stops the gate without retries.

The [throughput order](2026-09-13-throughput-priority.md) is unchanged: finish
the full active pair, reverify its complete raw evidence and exact no-spawn
boundary, then qualify this dependency against ce80e9cd. Require full production
gradients/cache/reset checks, complete state and all optimizer moments at updates
1 and 8, N6 pixel/restore traces, at least 2,048 MiB directly free and untraced
AB/BA timing. The first hardware stage is unchanged and cannot adopt the runtime
alone. A separate [full-state/pixel continuation](2026-09-13-meganeura-runtime.md)
is now declared after it: 55 passing CPU tests and 31,711 verified pins, with
follower 273381/14615337 waiting on 269283/14332226. The independent live audit
confirms the original pair is unchanged and neither GPU stage has started.
Only then test block matmul on the same qualified backend, holding
N6/R256/B16/T64/full BPTT/F32 and all scientific gates fixed. CPU tests and a
dependency refresh do not establish useful throughput or another Atari win.
The [identical block carry](2026-09-10-block-matmul.md#september-13-identical-carry-onto-latest-runtime)
now passes 98 Rust and 547 Python CPU tests on this same backend, with native
5e4ea9e1 and all 21 GPU fixtures prepared. Its five scalar CPU oracles pass;
the separate [block qualification](2026-09-13-block-matmul-runtime.md) is now
declared and waiting after the full dependency gate. No block GPU test, timing
result or adoption is claimed.
