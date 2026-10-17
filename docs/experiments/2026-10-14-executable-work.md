# Compare executed heavy work before another optimization

October14,2026. **Complete:** [result](../results/2026-10-14-executable-work.md).
Only1.40% extra nominal heavy work, despite2.20× retained complete latency;
actual four-head wide-GEMM dataflow is proved. Initial CPU tuple-boundary
assertion remains. CPU analysis of retained immutable artifacts; no new GPU,
learner, shader or timing allocation.

The current native RGB control is about2.2x complete JAX. Compensated dW is
not a large cost (.89% fixed-world effect); existing convolution prefetch
regresses the complete matched pair by.98%. Stop those tuning paths.

Audit whether the optimized programs perform comparable mathematical work:

- Parse the retained optimized BF16 JAX HLO from its entry, through reachable
  calls/fusions and declared loop trip counts. Do not count every textual
  computation once or ignore loops. Fail on unknown heavy call/control flow.
- Count dense/grouped dot and convolution work from actual operand/result
  shapes and contraction/window metadata. Keep padding and output precision
  explicit. These are nominal MACs, not hardware instructions or tensor activity.
- Inventory the current staged native plans without double-counting the
  posterior prefix, including horizontally batched products and all sessions.
- Reconcile layers/families/dynamic repetition and disclose differences.
  No new parity claim from equal parameters or a similar total alone.
- Reinspect all eight retained JAX capture windows for actual algorithm families,
  preserving transforms and auxiliary kernels. Kernel names alone do not
  prove their arithmetic instruction count or removable native cost.

Run one-CPU/2GiB/zero-swap bounded analysis; preserve parser failures. Reuse
existing artifacts, not a capture sweep. The result determines whether further
work should remove surplus mathematical work or improve equivalent execution.
