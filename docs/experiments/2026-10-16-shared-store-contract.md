# Shared-store ownership and lowering

October16,2026. CPU-only follow-up to the [native memory counters](../results/2026-10-16-kernel-memory.md).
Keep isolated58aff452, RGB46.199ms/JAX27.223ms; no backend, production or Atari change.

Audit the actual complete RGB/CDP matrix and convolution selections, their
generated WGSL and emitted SPIR-V. Distinguish scalar/vector operand stores,
cooperative result stores and explicit shared allocations. Match selected
sources, not every unused shader in a runtime catalog. Include the existing
F32/F16/BF16 generators as offline type coverage, not GPU qualification.

Use independent byte/element ownership checks for the existing layouts. A
conditional32-bank transaction model may explain the nine word kernels' affine
write-conflict counts, but must state its subgroup/transaction assumptions and
test alternative legal local-to-subgroup mappings. Vulkan does not guarantee
that mapping; cooperative matrix element ownership is also implementation-defined.
Do not label an inferred intercept as a measured output-store cost. SPIR-V is
not native assembly and does not expose driver-managed shared storage.

In particular, check whether dense products still use scalar shared writes
despite vector global loads. Reconcile their actual workload membership before
proposing another intervention. No padding/layout/geometry sweep, compiler
feature, source rewrite, GPU allocation or automatic promotion follows from this
declaration. Preserve existing arithmetic, precision, rounding and barriers.

One bounded CPU service: CPUQuota100%, MemoryMax2G, MemorySwapMax0, <=720s,
Restart=no and KillMode=control-group. Reuse already-built linked libraries;
compile only the source observer, not an unchanged backend. Preserve all failed
sources/logs and review before repair. Artifacts:
`runs/shared-store-contract-20261016.Mi1tGL3k`.
