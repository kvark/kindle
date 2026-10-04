# One external-memory import path

Blade [PR402](https://github.com/kvark/blade/pull/402) now extends
`Memory::External` rather than adding a Vulkan-specific buffer constructor.
Kindle and [Meganeura PR221](https://github.com/kvark/meganeura/pull/221) use
Blade `f54a55c3` / Meganeura `2c3130fb`. [Compact results](2026-10-01-external-memory-api.json).

`Fd(None)` exports. `Fd(Some((fd, allocation)))` imports through `create_buffer`
and the shared allocator. The existing external-source getter returns the
exporter's allocation size, binding offset, memory type and device/driver UUIDs
alongside the FD. The separate import constructor, import descriptor and
export-metadata accessor are removed. Imported FDs are borrowed and duplicated;
the caller retains ownership on both success and failure. Imported buffers no
longer expose a consumed FD. Queue-family acquire/release remain synchronization
operations, with producer completion and consumer completion before ring reuse.

## Validation

- Blade: 10 CPU tests; strict Clippy, formatting, GLES and wasm32 checks pass.
- Native allocation test: exact bytes at a nonzero 4096-byte binding offset,
  repeated imports from one caller-owned FD, rejection of wrong device/driver,
  short allocation, misalignment and out-of-range memory type, plus allocation
  cleanup. [Guard and output](../../runs/external-memory-rework-20261001.iAf7hk/blade-native-guard/).
- Kindle: 100 workspace CPU tests pass (41 hardware/fixture tests ignored);
  workspace and Python-binding strict Clippy and formatting pass.
- Native capture: two Vulkan contexts, three ring slots, twelve generations,
  exact bytes, split socket metadata/SCM_RIGHTS, reuse and STOP/drop cleanup.
  [Guard and output](../../runs/external-memory-rework-20261001.iAf7hk/kindle-guard/).
- Both passing native tests select RTX 5080, require at least 2 GiB sampled
  Vulkan estimated headroom, and enable Khronos synchronization validation.
  No reported validation error, synchronization hazard or new kernel fault.

The initial Blade test correctly rejected the default AMD integrated adapter
before import. Its [failed guard](../../runs/external-memory-rework-20261001.iAf7hk/blade-guard/)
is retained; the test now accepts explicit device selection. A CPU provenance
test also caught stale reported dependency revisions, fixed before final checks.
The failure snapshot's all-boot kernel log is truncated; the separate current-boot
check found no new GPU fault. No reset, driver change, retry of a quarantined
bundle or NVML polling occurred. CI remains tracked in the three PRs.

## Limits

These are import/integration tests, not learning or throughput measurements.
No new training, video pretraining or real-game competence run was needed.
Phase 2 evidence remains tied to its original backend revisions; Meganeura's
numerical code is unchanged. Opaque-FD imports require truthful Vulkan metadata;
dedicated allocations, device groups and imported-buffer device-address operations
remain unsupported. Allocation failure follows Blade's existing panicking resource
API. Budget headroom is not physical free/peak VRAM; utilization is unmeasured.
