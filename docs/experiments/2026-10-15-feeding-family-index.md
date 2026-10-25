# Correct the producer-lane identity before continuing

October15,2026. The [original complete-family declaration](2026-10-15-feeding-family.md)
remains sealed. Its CPU checks pass and the first five three-arm numerical
cohorts pass, but source review catches an unproven thread-to-subgroup mapping:
the producer branch uses subgroup IDs while its loader index uses local thread
ID modulo64. Full32-lane subgroups do **not** guarantee that relationship.
[Vulkan subgroup identity contract](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupLocalInvocationId.html).

Stop the original service without changing its sealed source. Five reviewed
cohorts/15 guards remain, as does the interrupted sixth native process and its
missing completion receipt. Systemd's successful stop is not experiment success.
No driver/API fault or wedge is observed. No original timing cohort starts.

The replacement derives producer index from `(subgroup_id % 2) * 32 + subgroup_lane`.
Exactly two full producer subgroups then own all64 indices regardless of local
invocation ordering. Keep32-lane/full-subgroup pipeline requirements and every
barrier, shape, tile, arithmetic operation, transform and gate unchanged. A CPU
permutation/negative test must expose the old assumption and verify the new one.
This is an indexing-contract repair, not a geometry or precision sweep.

Use fresh artifacts `runs/feeding-family-index-20261015.y5CaEr4O` and freshly
compiled shaders. Reuse unchanged native/C++ harness binaries only with source
and SHA verification; no backend rebuild. A fresh complete20-contract/23-occurrence
numerical cohort precedes the same conditional timing cohort. Original failures
and cancelled work remain extra compute, not replaced evidence.120s/process,
45min/service, the same GPU guard/headroom rules and CPU preparation limits.
Independent review remains between each three-arm cohort; no automatic retry
after another failure. Production, Naga and Atari remain unchanged.
