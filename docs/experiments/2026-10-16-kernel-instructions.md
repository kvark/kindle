# Remaining kernel instruction contracts

October 16, 2026. CPU-only diagnosis after qualified native `58aff452`:
RGB12M/B32/T16 46.199ms versus inclusive JAX27.223ms. No production or Atari
change and no new GPU allocation in this declaration.

Questions:

1. Which current matrix/convolution shaders still use runtime dimensions, and
   does the rejected packed-specialization trial actually cover those paths?
2. Are automatic bounds checks a possible source of the excess instructions?
3. Do retained native counters distinguish shader address work from local-memory
   traffic? If not, which supported additional metrics would distinguish them?

Reconcile all heavy products in the actual complete RGB/CDP plans, pipeline
keys and dumped shader sources. Keep the training-prior batching failure closed
to timing. Do not assume cuBLAS/cuDNN kernels specialize every dimension merely
because XLA has static tensor shapes. Static source operations, logical bytes
and SPIR-V are not executed machine instructions or latency estimates.

Use the existing SDK host evaluator, saved counter availability and counter
configuration. Enumerate supported metrics and their raw dependencies on CPU;
do not interpret an uncollected counter as zero. Local memory can represent
compiler-managed arrays as well as spills; unavailable compiler statistics are
not repaired by masking their bits. No native assembly or Graphics-injection
retry, barrier/robustness weakening, geometry/precision sweep or compiler project.

Preparation runs in a bounded user service: one CPU, 2GiB RAM, zero swap,
12-minute deadline, no automatic restart. Preserve original failures and source
receipts. A new counter capture, if warranted by this review, needs a separate
bounded declaration and unchanged-kernel/output checks, not another tuning sweep.

Artifacts: `runs/kernel-instructions-20261016.evva2SX8`.
