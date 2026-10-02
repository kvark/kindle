# Matching external allocations, without a metadata API

The October 1 [review](https://github.com/kvark/blade/pull/402#pullrequestreview-5382327523)
is implemented. `Memory::External(Fd(Some(fd)))` is the only import path.
`ExternalMemoryAllocation`, UUID queries/checks and binding-offset metadata are
removed. Imports borrow/duplicate FDs. Acquire/release now belong to
`CommandEncoder`. [Compact results](2026-10-02-matching-external-allocations.json).

Import and export create the same buffer size/usages/flags, use the same
deterministic memory-type choice, allocate the Vulkan requirement size and bind
at zero. Both omit device-address/ray-tracing flags. Compatible device/driver
and matching recipes are caller preconditions, not properties inferred from an
arbitrary FD. Dedicated allocations, device groups and arbitrary foreign
allocation recipes remain unsupported. This supersedes the
[October 1 metadata design](2026-10-01-external-memory-api.md), not its evidence.

Dullahan GPU_SYNC v3 matches Blade's recipe. Its 44-byte packet carries frame
geometry and logical buffer size, not allocation metadata. A new tag rejects
older recipes. The fenced EXTERNAL ownership/ACK/STOP protocol is unchanged;
GPU_SYNC does not map pixels on the host. Legacy CUDA/SHM wire behavior remains.

## Validation

- Blade `74c407a1`: 9 CPU tests; strict Clippy, formatting, GLES and wasm32 pass.
  [Guarded native roundtrip](../../runs/external-memory-contract-20261002.t2K3Nz/blade-guard/):
  sizes 257/6208/16384 derive allocations 320/6208/16384 and memory type 3 on
  both sides; exact bytes, repeated imports, caller-FD lifetime and allocator
  cleanup pass. The unaligned size tests the distinction between logical and
  allocation bytes without exposing another API field.
- Meganeura `089031ae` only repins Blade; numerical code is unchanged.
- Kindle: 100 workspace CPU tests, strict workspace/Python-binding Clippy and
  formatting pass. The strengthened capture test is also compiled/Clippy-checked.
  The native tests are ignored by ordinary CPU testing (42 ignored total).
- [Two-context ring](../../runs/external-memory-contract-20261002.t2K3Nz/ring-guard/):
  three slots, twelve generations, exact bytes, split SCM_RIGHTS metadata,
  ring reuse and STOP/drop pass in 3.26s.
- [Actual Dullahan/vkcube producer](../../runs/external-memory-contract-20261002.t2K3Nz/producer-fixed-guard/):
  Dullahan `2721a8d3`, Vulkan 1.0 producer, twelve 160x120 frames over three
  slots, nonuniform rendered pixels, ownership handoff and owned-process/IPC
  cleanup pass in 3.23s. Both processes use the NVIDIA ICD. This is capture
  plumbing, not game competence, actor throughput or an exact image oracle.
- Passing native checks use RTX5080/580.178.04, require at least 2GiB sampled
  Vulkan estimated headroom and enable Khronos synchronization validation.
  No reported validation error, synchronization hazard or new kernel fault.
  All local GPU services are stopped/reaped. No NVML polling or host recovery.

## Retained failure and fix

The [first producer run](../../runs/external-memory-contract-20261002.t2K3Nz/producer/)
transferred frames and exited zero but emitted three validation errors, so it
is **not** accepted as a clean result. Dullahan enabled the external-memory
device extension without its Vulkan 1.0 instance dependency. The layer now
adds the required instance extensions without raising the application's API
version; the repeated test passes and now fails automatically on producer
validation errors. The dependency is required by the
[Vulkan extension specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_external_memory.html).
No driver fault occurred. Two CPU-only test-helper compile errors were fixed
before native execution; no failed native attempt is discarded.

## Integration and limits

[Blade #402](https://github.com/kvark/blade/pull/402),
[Meganeura #221](https://github.com/kvark/meganeura/pull/221) and
[Dullahan #15](https://github.com/kvark/dullahan/pull/15) carry the dependencies.
Mind-games `a177872` updates its Dullahan gitlink. The active
[Kindle PR31](https://github.com/kvark/kindle/pull/31) remains the CI/status
dashboard. Only the maintainer merges; repin chosen merged revisions when landing.
Blade CI1149 initially reports a Windows ray-tracing-test access violation;
Linux/macOS GPU checks pass. It is tracked separately, not hidden by the local
Linux validation. Phase 2 remains complete. No new training, performance claim,
utilization measurement, CPU learner workaround or Phase 3 campaign.
