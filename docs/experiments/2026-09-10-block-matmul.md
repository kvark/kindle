# Small-batch block matrix products: CPU-only candidate

September 13 scheduling update: the user has prioritized throughput qualification
before unstarted Pong work. The [boundary hold](2026-09-13-throughput-priority.md)
preserves active root 1009 and holds later roots. Qualify latest Meganeura/Blade
separately before testing block-matmul on the same backend. Earlier queue-order
statements below describe the preserved declarations, not the new priority.
The [latest same-backend carry](#september-13-identical-carry-onto-latest-runtime)
is now CPU-qualified on 75dfe901; dependency GPU qualification still comes first.

Prepared September 10, with a September 11 carry onto the qualified upstream
backend and a September 12 source-matched Python package below. Not adopted,
timed or GPU-qualified. The pinned learning
campaigns keep their original packages and inputs. Neither candidate starts a
GPU worker or follower.

## Change and rationale

World training and posterior inference dominate much of the remaining learner
cost. Each RSSM block-linear layer currently slices eight independent input,
weight and bias blocks, multiplies them separately, then concatenates outputs.
Meganeura's ordinary matmul batches rows against one matrix; it does not batch
independent weight matrices. The similarly named Gemma `batched_matmul_bt` test
still uses two-dimensional operands.

The new F32 block operator uses the existing register-tiled accumulation loop,
with block-aware storage indices and one block per grid-z index. It includes
both transposed products needed by autodiff:

| Product | Input layout | Output layout |
| --- | --- | --- |
| Forward | `[M, G*K]`, `[G, K, N]` | `[M, G*N]` |
| Input-transposed | `[K, G*M]`, `[K, G*N]` | `[G, M, N]` |
| Weight-transposed | `[M, G*K]`, `[G, N, K]` | `[M, G*N]` |

Kindle uses it only for batches 2–16. Batch-one GEMV and larger cooperative
imagination batches retain their original serial graph. The original GRU gates,
parameter names/shapes, initialization and optimizer ownership are unchanged.
This is separate from the
[rejected grouped-GRU rewrite](2026-09-08-grouped-rssm-gates.md), and does not
resolve that rewrite's numerical failure.

For each isolated production block layer—eight blocks, input/output widths
1024/256 or 256/768, batches 2/4/6/8/16—raw and optimized graph compilation
reduces **65 dispatches to 2**. These are block-only CPU graph counts, not a
whole-world plan, a GPU occupancy measurement, a memory result or a speedup.
Large imagination batches are deliberately outside this change. No full-sized
world/BPTT memory-plan probe was run alongside training.

## Source and CPU evidence

Both isolated branches are named `exp/block-matmul` and are committed/pushed:

- Meganeura `70803c22dd23ee73150febb69aae77818c568ab4`, based on qualified
  `4d45ba3a`; worktree `/x/Code/.meganeura-block-matmul-20260910`.
- Kindle `4ae539ae72aec5915697f26eda468a1b9554e3da`, based on qualified
  `90b4763`; worktree `/x/Code/.kindle-block-matmul-20260910`.

The 33-pin summary is
[`cpu-evidence.json`](../../runs/block-matmul-cpu-20260910.Vqu6hq/cpu-evidence.json),
SHA-256 `a001555a263fb91953492253b82db1e7f5b1ed6849cd90d4789bf9504a0843b1`.
Its read-only `inspect.py` rechecks source bodies and the preserved test logs.

- 94 focused Meganeura CPU checks pass: all three composed-loss chain rules
  against f64 finite differences; shape/storage rejection; production dispatch
  geometry; shader validation/SPIR-V generation; graph, autodiff and compiler
  regressions. The ordinary matmul template arithmetic is source-identical
  after expanding the new layout hooks. Blade assigns bindings later; the
  shader-name/runtime-binding check passes too.
- All 98 ordinary Kindle workspace/target tests pass; 23 GPU tests are ignored.
  The new CPU tests preserve parameter layouts, derivative precision markers
  and exact batch-one/large-batch graph serialization. These are not native
  weight, optimizer-state or output parity checks.
- Formatting and Clippy pass for Meganeura, the Kindle workspace and Python
  bindings. No Python source changed, no new Python native package was built,
  and no Python runtime-test result is claimed for this candidate.

Initial setup failures are retained: shader validation initially requested
bindings before Blade assigns them, and a library test's exhaustive shader-entry
match needed the new variants. Both are repaired and the final suites pass.
The scratch backend's initial locked build also required copying the unchanged
qualified lock; subsequent backend builds were locked/offline. Kindle builds
use its own unchanged dependency versions, with only the Meganeura source pin
changed. Builds used one low-priority CPU job and fresh candidate targets.

The active handoff's 756 input pins also reverified unchanged. These CPU builds
overlapped ordinary Boxing training, not a matched timing experiment. No control
target directory, package, checkpoint or runner was replaced.

## Required hardware comparison, not yet declared or scheduled

Retain the current serial learning queue. Afterwards, predeclare a separate
comparison using source-matched candidate and control packages on the qualified
current backend. Do not combine this with world-sync fan-out, a
backend refresh, exploration, replay-ratio or BPTT changes.

1. Run the new ignored native f64 composed-loss/all-gradient test and Kindle's
   production block output/all-gradient exact comparison. Include the existing
   production world-loss/gradient, reset-causality and serial/vector checks.
2. Require exact complete full-learning canary state and non-timing reports in
   both orders, including logical weights, every optimizer moment, normalizers
   and initial state. Preserve snapshots starting at update 1: zero learning
   rate does not mean unchanged optimizer moments. Any mismatch rejects adoption;
   diagnose it without weakening the comparator or editing historical runs.
3. Require matched N6/B16/T64/full-BPTT/F32/R256 pixel AB/BA evidence: exact
   state and action/reward/reset traces, complete GPU workload coverage, at
   least 2,048 MiB directly reported free, and repeatable untraced end-to-end
   timing gains. Report learner substages and aggregate/per-stream game clocks.

Only passing all gates would support runtime adoption. Fewer dispatches and
passing component tests alone do not establish learning parity, super-real-time
training, Pong reliability or any additional Atari win.

## September 11: unchanged block candidate on qualified upstream

The isolated `exp/block-matmul-upstream-20260911` branch, commit
`7b190f881e07ffc39d01396da6dda98d3d90490f`, carries the candidate onto
**qualified control source `1e00e818` / Meganeura `ce80e9cd`**. Its only native
diff from that control is `networks.rs`, byte-identical to the original `4ae539a`
candidate, including its tests. The other change is an isolated-worktree
AGENTS.md notice. Both lockfiles, backend identity, manifests and all Python
sources remain unchanged; upstream already contains the block operator.

This removes the need to compare a new block graph against a different backend.
It does not adopt the graph, transfer old performance claims or overwrite either
original candidate. Future controls must use the same qualified upstream backend,
without world-sync, action-vocabulary, reward, replay-ratio or BPTT changes.

The [completed CPU preparation](../../runs/block-matmul-upstream-cpu-20260911.PZnUq0/result.json)
passes **98 Rust workspace/all-target tests**, formatting, and both workspace
and Python-binding Clippy checks with warnings denied. All **23 GPU tests remain
ignored**. The release library and canary are compiled and pinned; the production
block all-gradient test is listed but unrun, and the canary is not executed.
Compiler output independently confirms the library was rebuilt from the actual
candidate source, not reused from the copied control cache.

The twenty raw/optimized component cases still show **65 → 2 dispatches** for
batches 2/4/6/8/16 and the two production block shapes. Batch-one and large-batch
graph serialization stays exact; parameter layouts and full-precision derivative
markers pass. These are CPU graph properties, not GPU numeric parity, memory
headroom or useful end-to-end speed. No production-sized world memory-plan probe
ran alongside Qbert.

All ten preparation command exits/output hashes and **5,683 evidence pins**
independently reverify, including 3,699 copied-cache inputs and all 1,870 active
hardware pins. Builds used a fresh target and one low-priority job under an
enforced one-core / 2 GiB / zero-swap scope, peaking at **1,932.9 MiB host memory**.
The original cache, controls and source are unchanged. This is not a GPU-memory
measurement. No new Python native package was built or runtime-tested.

Keep the current Qbert → Freeway → Pong → Breakout hardware queue fixed. This
carry declares no GPU comparison or follower. The full hardware, complete-state
from update 1, source-matched Python-package, pixel AB/BA, direct-free memory and
timing gates above remain necessary before adoption for any later learning run.
Preserve the completed preparation; no speedup or additional game is claimed.

## September 12: isolated Python package, CPU-qualified only

The [completed package](../../runs/block-matmul-package-continuation-20260912.QUT8zv/result.json)
builds the unchanged `7b190f8` candidate into an isolated CPython 3.14 wheel and
package, native **`f4742ac7`**. Its actual import, compiled library and wheel
bytes agree; all six Python modules match the candidate source. The build log
confirms both Kindle and its Python binding compiled from that worktree.
All **547 Python CPU tests pass** with this extension. The existing **98 Rust
CPU tests / 23 ignored GPU tests** are rechecked from their pinned raw logs,
not rerun or represented as GPU evidence.

The [read-only rechecker](../../runs/block-matmul-package-continuation-20260912.QUT8zv/audit.py)
independently verifies all **9,574 pins**, three command lifecycles and raw logs,
source/wheel/import identity, test counts and the unchanged historical/default
and qualified upstream controls. Result SHA-256:
`b0feb3251cb1c8e41549c71cebebecbfc7e129fb86fa30830b94bbc6f7e736f7`.
That upstream preflight found `3622e06f`; only documentation/paper files differed
from the package's qualified `ce80e9cd` runtime. The later **19:12 UTC** check
finds runtime changes at `45991be1`; see the separate
[timing dependency update](2026-09-12-meganeura-timings.md). This package remains
the preserved ce80e9cd candidate, not the newest upstream runtime.

The first attempt in `runs/block-matmul-package-20260912.Ipk6gU` stops before
compilation because the capped service cannot find Cargo. Preserve its
[original failure](../../runs/block-matmul-package-20260912.Ipk6gU/wheel-build.log)
and copied cache. The separate continuation only exposes the installed Rust
toolchain in the subprocess PATH; it changes no source or dependency. It finishes
normally under one core / 2 GiB / zero swap, with a **2,048 MiB cgroup memory
peak**. That is host memory, not GPU usage or extra headroom. No production world
graph, GPU fixture or native learning agent is constructed.

The package is ready for a future matched comparison against qualified
`1e00e818` / native `abf4ae5d`, not for learning use yet. Keep the complete-state,
all-gradient, cache/reset, N6 pixel AB/BA, direct-free memory and timing gates
above. No GPU declaration or follower is added; the existing Freeway → Pong →
Breakout hardware order remains unchanged. Preserve both exclusive packaging
writers; the read-only rechecker may be reused.

### Backend oracle executable

The upstream oracle is the separate `block_matmul` test target; it is not in
the already compiled `regression` executable. The [isolated preparation](../../runs/block-matmul-backend-fixture-20260912.TVdIQX/result.json)
builds that target from unchanged ce80e9cd, using a private copy of the qualified
build cache. All five CPU tests pass; `gpu_composed_losses_and_all_gradients_match_f64`
is listed but not executed. No new test implementation or backend change was needed.

The fresh release executable is `target/release/deps/block_matmul-4e7513bce8c1178f`
under that run, SHA-256
`331b32d6f9e0d94d13f45c56ba96600d1e8be7ed6e677a2c14130687c9be7389`.
Its [read-only audit](../../runs/block-matmul-backend-fixture-20260912.TVdIQX/prepare.py)
(`--audit`) independently reverifies all 12,287 input/output pins, the four
command lifecycles, actual compiled source, raw CPU results and preserved controls.
Result SHA-256: `a1ec7bc340f7a63ce5b65611c5f3a935bec45513f5f2386e355d20fd85095485`.
The build uses one core / 2 GiB / zero swap and starts no GPU work. Its preceding
remote check found documentation-only tip 3622e06f; the newer timing update above
requires separate qualification before carrying this optimization forward. Preserve the completed
exclusive writer; this supplies a missing executable for the future comparison,
not GPU qualification, a speedup, adoption or a new follower.

## September 13: identical carry onto latest runtime

The isolated `exp/block-matmul-conv-20260913` branch, **`20b9b8a`**, carries the
same `networks.rs` byte-for-byte onto dependency-only control **58f328a /
Meganeura 75dfe901 / Blade f6f2729e**. Only that file and the worktree's AGENTS.md
notice differ from the control. Both locks, backend identity, Python sources and
learning settings remain unchanged. Older branches and packages are preserved.

The [completed CPU check](../../runs/block-matmul-conv-cpu-20260913.4MgN62/result.json)
passes **98 Rust tests**, formatting and both Clippy checks. All **23 GPU tests
remain ignored**. The twenty raw/optimized block-only cases still give 65-to-2
dispatches; the production output/all-gradient GPU fixture is listed, not run.
All eight command lifecycles and **30,255 pins** reverify. Result hash:
`da5bf33dacb5f16a07cb6c3fc919f9355e81c52cbfe6301c42b044dc8476e240`.

The run uses one CPU core, one Cargo job, 2 GiB RAM, zero swap and a private cache
copy. No production world graph, new Python package, GPU comparison or follower
is created. Preserve the completed writer; `prepare_cpu.py --audit` is read-only.
Qualify the [latest dependency](2026-09-13-meganeura-conv.md) first, then the full
same-backend component/state/trace/memory/AB/BA comparison before the remaining
Pong roots. These CPU results do not establish exact native learning or a speedup.
