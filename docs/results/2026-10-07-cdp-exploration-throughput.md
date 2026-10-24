# Projection reuse rejected at numerical qualification

The [declared rewrite](../experiments/2026-10-07-cdp-exploration-throughput.md)
is **not adopted**. Two bounded GPU checks fail their fixed limits. No timing,
game actions or actor updates run, and no tolerance is relaxed. Production
returns to the original expanded-action graph; candidate source, binaries and
failed logs remain under `runs/cdp-action-throughput-20261007.dAJvnek`.
[Compact evidence](2026-10-07-cdp-exploration-throughput.json),
[independent review](../../runs/cdp-action-throughput-20261007.dAJvnek/failure-review.json).

| Check | Observation | Decision |
| --- | --- | --- |
| Complete-head F64, B3 | Tiny/1M pass.12M head0 has max absolute2.946e-6, relative L2 9.10e-7, but exceeds the pointwise2e-6×(1+abs(reference)) limit. Expanded/factored outputs are identical for this case. | Failed; this F64 excess is shared by the control. |
| Previously unstarted production1M B128 | First head differs by max absolute1.880e-3, relative L2 4.909e-4 versus the2e-5 gate. | Failed; remaining shapes/jobs stay unrun. |

The first test stops after3.46s. After reviewing it, a new bounded queue declares
only the previously unstarted production-shape checks; its first check fails.
This is not a retry of the first test or an extended learning campaign. Neither
the1920-row nor12M production-shape checks complete. No GPU speedup is measured.

Meganeura's Auto policy can use reduced-input cooperative products on f16-only
devices, and the rewrite changes matrix shapes. Different selection is a
possible explanation for the larger discrepancy, **not proven causality**.
The equal B3 outputs also show why a new full-head F64 failure cannot simply be
attributed to this algebraic rewrite. The CPU identity remains true; it did not
prove preservation of GPU floating-point behavior.

Both jobs exit cleanly with test code101, no unfinished child, recovery or
separate NVML queries. Their sealed receipts remain; bounded post-failure
kernel snapshots exceeded the output limit, so this is not a complete incident
capture or proof about driver health. This is a numerical job failure, not
evidence that the GPU is wedged.

CPU parameter/gradient-detachment checks, formatting, Clippy and1,189 Python
tests pass. Earlier stale-revision and test-lint build failures remain in the
artifact directory. Removed production/test code is recoverable there, without
adding an unqualified compatibility path to Kindle.

Proceed with the separately declared [latest-backend control refresh](../experiments/2026-10-07-meganeura-control-refresh.md),
using the unchanged production graph and established scoped numerical checks.
That does not make the failed stronger test pass. No precision-policy change,
CPU learner or optimization sweep. Review its bounded timing before the next
learning allocation. All five quality targets and the budget question remain open.
