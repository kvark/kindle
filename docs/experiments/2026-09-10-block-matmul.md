# Small-batch block matrix products: CPU-only candidate

Prepared September 10. Not adopted, timed or GPU-qualified. The active Boxing
confirmation and its serial successors keep their original packages and inputs.
This candidate starts no GPU worker or follower.

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

Retain the current serial learning queue. Afterwards, build source-matched
candidate and control packages and predeclare a separate comparison on the
qualified current backend. Do not combine this with world-sync fan-out, a
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
