# Prior-gradient discrepancy: finite-precision propagation

October16,2026. **The captured first-variant failures are explained; prior
batching is not yet qualified for whole updates.** Keep native834241b6 and
RGB48.068/JAX27.223ms,1.77×. No production/Atari change or new speed claim.
[Declaration](../experiments/2026-10-16-prior-gradient.md),
[summary](2026-10-16-prior-gradient.json),
[original failed qualification](2026-10-16-prior-batching.md),
[forward diagnosis](2026-10-16-prior-tape.md).

## What was wrong with the comparison

The original reference rounds matrix operands to BF16 but carries normalization,
activation and backward arithmetic in F64. This is not the executed F32 program.
The forward capture already explained the second variant's logit differences.
Using that captured forward tape and source-ordered F32 derivatives now clears
every gradient gate in the second variant, in both serial and batched arms.
The same CPU hypothesis still fails3/130 first-variant weight values, so that
variant received one diagnostic backward capture, not a qualification rerun.

The capture preserves the original168 dispatches, plan, pipelines and buffer
groups. All retained outputs,40 raw-gradient arrays and immutable parameters
match exactly. Its39 selected intermediates show the remaining mechanism:

| Matrix | BF16 derivative inputs different from the original reference | Original weight-gradient failures | Failures with observed rounded operands |
| --- | ---: | ---: | ---: |
| Prior logits | 50 | 0 | 0 |
| Prior layer1 | 115 | 3 | 0 |
| Prior layer0 | 403 | 66 | 0 |

The first difference is already in the KL derivative: maximum absolute
F32-versus-reference difference1.93127e-10 changes50 BF16 values. Subsequent
derivatives and rounding propagate this difference. All three forward matrix
inputs remain identical after BF16 rounding in this variant; the discrepancy
is on the derivative side, not a changed activation or missing gradient.

Independent F64 local softmax/KL, SiLU, RMSNorm and matrix calculations pass the
unchanged pointwise2e-5+.02|reference| and relativeL2<=.01 gates. BF16 casts and
free-nat masks match exactly; all512 states are above the threshold, with minimum
margin.34709. The36 checks cover7,686,144 comparisons, including exact output
handoffs and repeated reconciliation checks, not that many unique oracle values.
There are49 finite-difference checks and seven rejecting negative controls.

For every weight matrix, the observed operand differences exactly reconcile
the reference change using `(X−Xref)ᵀDref + Xᵀ(D−Dref)`. Remaining matrix
accumulation errors are at most5.402e-7. A separate CPU audit checks the same
recorded operands against both retained arms: all six matrix checks pass;
the serial maximum is1.676e-7. No tolerance is fitted or enlarged.

## Limits and next decision

This is a local, compositional diagnosis, **not an end-to-end F64 pass or a
full-world independent oracle**. The original14 failed array checks remain
failed. The second-variant F32 calculation is a source-arithmetic hypothesis,
not a bit-exact Vulkan simulator. All original16 cases/eight pairs remain;
Tiny cases still pass their independent checks, and the four clamped12M cases
retain identical forward inputs/logits and exactly zero gradients.

The evidence supports a separately declared actual-world staging comparison
using the unchanged serial/batched raw-gradient gates. It does not automatically
authorize a timing result or promoteaca80566. Whole/staged equality, changed late
targets, poisoned-prefix isolation and unchanged state still precede timing.

One GPU guard/no warnings, one raw backward, zero optimizer updates/actions;
sampled estimated budget headroom16,480,206,848bytes. The guard finishes09:00:44UTC;
no worker remains. CPU review verifies2,669 unique source seals. Preserve the
reader's39-element-array serialization compile failure and the audit's mistaken
35-check count; slice serialization and explicit36-check membership repair
both pass without a GPU repeat. Only the standalone reader was built; backend
sources/libraries are unchanged. Upstream f648a453/4845d652 remains unchanged.

Raw artifacts: `runs/prior-gradient-20261016.4efbUy2T`.
