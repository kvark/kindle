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
Linux validation. Its one Windows retry subsequently passes, making CI1149
green; this does not erase the initial failure. Phase 2 remains complete.
No new training, performance claim,
utilization measurement, CPU learner workaround or Phase 3 campaign.

A final upstream check found new Meganeura main `b947950` (`f05c1a0` moves
Adam/LaProp bias correction to the host; `b947950` adjusts optimizer-padding
test tolerance). That optimizer arithmetic change is a separate qualification
before the next learner experiment, not part of this capture-only backend
repin. A read-only merge-tree check finds no conflict with the current PR.

## Second review: whole-buffer ownership

The October 2 review is implemented in Blade `a7861806`: allocation failure
panics where it happens; one match constructs the import descriptor and
duplicates its FD; ownership methods are safe and take a whole `Buffer`.
Their barriers use `VK_WHOLE_SIZE`. Meganeura `4cbcd69b` only repins Blade.

Dullahan `30aa6d3e` and Kindle now use GPU_SYNC v4, rejecting v3's per-slot
ownership contract. Both release/acquire the whole ring buffer; the producer
skips its initial acquire once per **buffer**, not once per slot. Copy offsets
and the 44-byte geometry packet remain unchanged. Mind-games `589ed04` updates
its producer gitlink. There is no compatibility shim or new import path.

- Blade: 9 CPU tests, strict Clippy, formatting, GLES and wasm32 pass. Its
  [guarded allocation/ownership regression](../../runs/external-whole-buffer-20261002.yVEMhc/blade-guard/)
  passes in 2.57s with no new reported validation error or kernel warning.
- Dullahan: 4 CPU tests, strict Clippy and release build pass, including a new
  regression for first-use ownership across different ring slots. CI60 passes.
- Kindle: 100 workspace CPU tests (42 ignored), formatting and strict
  workspace/Python-binding Clippy pass with the new pins.
- The [whole-buffer ring test](../../runs/external-whole-buffer-20261002.yVEMhc/ring-guard/)
  passes its exact-byte/three-slot/twelve-generation checks in 1.99s, but is
  **not accepted as a clean GPU qualification**: its journal delta contains an
  NVIDIA `nvCheckOkFailedNoLog` / `NV_ERR_NO_MEMORY` warning from
  `_memdescAllocInternal`, received at 06:48:19.753 UTC. The guard returned true
  because its fault matcher does not include this warning. Preserve that raw
  result; do not equate it to clean hardware qualification.
- Further native work stopped. The declared v4 real-Dullahan-producer test
  **has not run**. The earlier v3 producer success is not evidence for v4.

The [host-only snapshot](../../runs/gpu-incident-whole-buffer-ae74xvpv/host/)
is sealed (21 files verified). No new Xid, hang or OOM-kill is recorded; the
test exited zero and its systemd service reports a 128MiB memory peak. The
same warning appears twice earlier in this boot. Its kernel source monotonic
time is 317192.876721s, journal receipt time 317197.270103s; those timestamps
do not establish the triggering process. Cause and severity are unresolved,
not proof of a new wedge or of harmlessness. No NVML polling, recovery, retry
or new training was performed. Review the warning before another local native
run; the producer qualification remains pending.
