# Narrow products: less padded arithmetic, slower learner

October14,2026. [Declaration](../experiments/2026-10-14-narrow-products.md)
· [Data](2026-10-14-narrow-products.json).

**Reject this candidate: complete RGB updates slow28.488→30.492ms (7.03%).**
World training/sync increases15.063→17.032ms; other stages barely change.
One ordinary16-warmup+256-measured pair, RGB12M/B8/T16/H15/BF16, four world
chunks, same binary disabled/enabled. No timing repeat, geometry sweep, CDP
follow-up or production promotion. The best retained experimental control
remains about2.20× complete JAX; installed production is unchanged.

The shared typed checked-product generator now has an isolated scalar path
for1–8 rows: each lane owns a column and reuses its loaded operand across
the real row accumulators. F32 arithmetic,256-position compensated partials,
existing indexing/K partitions and explicit BF16 rounding remain. This removes
padded tensor products and shared staging, but also gives up tensor-core
execution. It is not a controlled measurement of padding alone. The actual
learner changes exactly two checked decoder kernels, three-output forward
and dW; all other plans/pipelines match. Staged posterior export is the same
world graph, not two additional changed kernels.

All90 numerical cases/4,705,658 values pass unchanged independent F64 gates,
maximum relative L2=6.161e-7. Dense/transposed/ragged and convolution forward/
dX/dW tests include long partials, aliases/poison and explicit rounding. F32
scalar execution is qualified, not F32 cooperative hardware. Grouped indexing
is shared but no grouped GPU fixture was run. CPU checks pass90 shader variants,
eight selection assertions,45 existing shader tests and both release Clippy
checks. No numerical tolerances changed.

Disabled control reproduces all272 retained reports/first outputs and292
initial/first/final tensors exactly. Candidate initials/first outputs match;
192 first tensors are exact (aggregate relative L2=.03197), only one final
tensor is exact (L2=.08646), and0/272 reports match. Finite trajectories are
not learning equivalence. Exact first outputs precede the changed backward
execution and therefore do not prove identical gradients.

Five guards pass; three standalone allocation warnings remain. Total592
synthetic updates and90 optimizer-free numerical executions, zero game actions.
Source patches, locks, both binaries, raw plans/checkpoints and complete audits
remain in `runs/narrow-products-20261014.YS5lxYKB`. No worker/recovery remains.
Retained optimized056562b-based backend/Blade96c1dc7/Naga a97559b; upstream
7574ad1/e7c3f7b was rechecked, not combined with a full-main migration.

The [five expensive boundary convolutions](2026-10-14-heavy-issued-work.md)
remain a real limitation; this alternative does not fix them. Move next to
the separately measured CPU command-recording/submission cost, while retaining
the GPU dependency and library-algorithm gap. Vulkan recording reuse is not
evidence of CUDA-graph-equivalent device execution.
